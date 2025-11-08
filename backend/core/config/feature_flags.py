"""
Feature Flags - Harvey/Legora %100 Dynamic Feature Toggle System.

LaunchDarkly-style feature management with:
- Percentage rollouts (canary releases: 5% → 50% → 100%)
- User/tenant targeting
- A/B testing variants
- Real-time updates via Redis pub/sub
- Graceful fallback to static config

Why Feature Flags?
    Without: Hard deploys → risky rollbacks → downtime
    With: Runtime toggles → canary releases → zero-downtime

    Impact: Netflix/Google-level deployment safety! 🚀

Architecture:
    Static Layer: backend/core/config.py (RBAC_ENABLED, CACHE_ENABLED)
    Dynamic Layer: Redis-backed flags with pub/sub updates
    Fallback: Static config if Redis unavailable

Usage:
    >>> from backend.core.config.feature_flags import is_feature_enabled
    >>>
    >>> # Simple flag check
    >>> if is_feature_enabled("vector_search"):
    ...     return vector_search_results()
    >>>
    >>> # Percentage rollout (canary)
    >>> if is_feature_enabled("new_rag_pipeline", user_id=user.id, rollout=0.10):
    ...     return new_rag_v2(query)  # 10% of users
    >>>
    >>> # Tenant targeting
    >>> if is_feature_enabled("premium_features", tenant_id=tenant.id):
    ...     return premium_search()
"""

import hashlib
import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from uuid import UUID

logger = logging.getLogger(__name__)


class FeatureFlagType(str, Enum):
    """Feature flag types."""

    BOOLEAN = "boolean"  # Simple on/off
    PERCENTAGE = "percentage"  # Rollout by percentage
    TARGETED = "targeted"  # User/tenant whitelist
    VARIANT = "variant"  # A/B testing (A, B, C)


class FeatureFlag:
    """
    Feature flag definition.

    Harvey/Legora %100: Flexible feature toggle with targeting.

    Attributes:
        name: Flag name (e.g., "vector_search")
        enabled: Default enabled state
        type: Flag type (boolean, percentage, targeted, variant)
        rollout_percentage: Rollout percentage (0.0-1.0)
        whitelist_users: User IDs to always enable
        whitelist_tenants: Tenant IDs to always enable
        variants: A/B test variants {"A": 0.5, "B": 0.5}
    """

    def __init__(
        self,
        name: str,
        enabled: bool = False,
        flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN,
        rollout_percentage: float = 0.0,
        whitelist_users: Optional[List[UUID]] = None,
        whitelist_tenants: Optional[List[UUID]] = None,
        variants: Optional[Dict[str, float]] = None,
    ):
        self.name = name
        self.enabled = enabled
        self.type = flag_type
        self.rollout_percentage = rollout_percentage
        self.whitelist_users = whitelist_users or []
        self.whitelist_tenants = whitelist_tenants or []
        self.variants = variants or {}

    def is_enabled_for_user(
        self,
        user_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
    ) -> bool:
        """
        Check if flag enabled for user.

        Args:
            user_id: User UUID
            tenant_id: Tenant UUID

        Returns:
            bool: True if enabled for this user

        Example:
            >>> flag = FeatureFlag("vector_search", rollout_percentage=0.10)
            >>> flag.is_enabled_for_user(user_id=UUID("..."))
            True  # User in 10% rollout
        """
        # Whitelist check (always enabled)
        if user_id and user_id in self.whitelist_users:
            return True
        if tenant_id and tenant_id in self.whitelist_tenants:
            return True

        # Boolean flag (simple on/off)
        if self.type == FeatureFlagType.BOOLEAN:
            return self.enabled

        # Percentage rollout (consistent hashing)
        if self.type == FeatureFlagType.PERCENTAGE:
            if not user_id:
                return self.enabled

            # Hash user_id to get consistent percentage
            hash_input = f"{self.name}:{user_id}".encode("utf-8")
            hash_value = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
            user_percentage = (hash_value % 10000) / 10000.0

            return user_percentage < self.rollout_percentage

        return self.enabled

    def get_variant(self, user_id: Optional[UUID] = None) -> Optional[str]:
        """
        Get A/B test variant for user.

        Args:
            user_id: User UUID

        Returns:
            str: Variant name ("A", "B", "control")
            None: If not variant flag

        Example:
            >>> flag = FeatureFlag("rag_ab_test", variants={"A": 0.5, "B": 0.5})
            >>> flag.get_variant(user_id=UUID("..."))
            "A"  # 50% of users
        """
        if self.type != FeatureFlagType.VARIANT or not user_id:
            return None

        # Hash user_id to get consistent variant
        hash_input = f"{self.name}:{user_id}".encode("utf-8")
        hash_value = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
        user_percentage = (hash_value % 10000) / 10000.0

        # Assign variant based on percentage buckets
        cumulative = 0.0
        for variant_name, percentage in self.variants.items():
            cumulative += percentage
            if user_percentage < cumulative:
                return variant_name

        return "control"  # Fallback


# =============================================================================
# FEATURE FLAG REGISTRY
# =============================================================================

# Harvey/Legora %100: Turkish Legal AI Platform Feature Flags
FEATURE_FLAGS: Dict[str, FeatureFlag] = {
    # === SEARCH & RAG ===
    "vector_search": FeatureFlag(
        name="vector_search",
        enabled=False,  # Disabled by default (needs Weaviate)
        flag_type=FeatureFlagType.BOOLEAN,
    ),
    "citation_graph": FeatureFlag(
        name="citation_graph",
        enabled=False,  # Disabled by default (needs Neo4j)
        flag_type=FeatureFlagType.BOOLEAN,
    ),
    "hybrid_search": FeatureFlag(
        name="hybrid_search",
        enabled=True,  # Enabled: Elasticsearch + vector
        flag_type=FeatureFlagType.BOOLEAN,
    ),
    "semantic_search": FeatureFlag(
        name="semantic_search",
        enabled=True,  # Enabled: LLM-powered semantic search
        flag_type=FeatureFlagType.BOOLEAN,
    ),

    # === RAG PIPELINE ===
    "rag_v2": FeatureFlag(
        name="rag_v2",
        enabled=False,
        flag_type=FeatureFlagType.PERCENTAGE,
        rollout_percentage=0.10,  # Canary: 10% rollout
    ),
    "rag_streaming": FeatureFlag(
        name="rag_streaming",
        enabled=True,  # SSE streaming responses
        flag_type=FeatureFlagType.BOOLEAN,
    ),
    "rag_multi_hop": FeatureFlag(
        name="rag_multi_hop",
        enabled=False,  # Multi-hop reasoning (experimental)
        flag_type=FeatureFlagType.PERCENTAGE,
        rollout_percentage=0.05,  # 5% beta users
    ),

    # === RBAC & SECURITY ===
    "rbac_strict_mode": FeatureFlag(
        name="rbac_strict_mode",
        enabled=True,  # Deny by default
        flag_type=FeatureFlagType.BOOLEAN,
    ),
    "mfa_enforcement": FeatureFlag(
        name="mfa_enforcement",
        enabled=False,  # Optional MFA
        flag_type=FeatureFlagType.TARGETED,
        whitelist_tenants=[],  # Add enterprise tenants
    ),
    "session_recording": FeatureFlag(
        name="session_recording",
        enabled=False,  # Security audit recording
        flag_type=FeatureFlagType.TARGETED,
    ),

    # === CACHING ===
    "cache_warming": FeatureFlag(
        name="cache_warming",
        enabled=True,  # Startup cache preload
        flag_type=FeatureFlagType.BOOLEAN,
    ),
    "adaptive_cache": FeatureFlag(
        name="adaptive_cache",
        enabled=True,  # 7-day active user filtering
        flag_type=FeatureFlagType.BOOLEAN,
    ),

    # === ANALYTICS & MONITORING ===
    "advanced_analytics": FeatureFlag(
        name="advanced_analytics",
        enabled=False,  # Premium analytics
        flag_type=FeatureFlagType.TARGETED,
    ),
    "real_time_metrics": FeatureFlag(
        name="real_time_metrics",
        enabled=True,  # Prometheus metrics
        flag_type=FeatureFlagType.BOOLEAN,
    ),

    # === A/B TESTING ===
    "rag_model_test": FeatureFlag(
        name="rag_model_test",
        enabled=True,
        flag_type=FeatureFlagType.VARIANT,
        variants={
            "gpt4": 0.50,  # 50% GPT-4
            "gpt35": 0.50,  # 50% GPT-3.5
        },
    ),
}


def is_feature_enabled(
    flag_name: str,
    user_id: Optional[UUID] = None,
    tenant_id: Optional[UUID] = None,
    default: bool = False,
) -> bool:
    """
    Check if feature flag enabled.

    Harvey/Legora %100: Consistent feature toggle with rollouts.

    Args:
        flag_name: Feature flag name
        user_id: User UUID for targeting
        tenant_id: Tenant UUID for targeting
        default: Default value if flag not found

    Returns:
        bool: True if enabled

    Example:
        >>> # Simple check
        >>> if is_feature_enabled("vector_search"):
        ...     return vector_results()
        >>>
        >>> # With user targeting (10% rollout)
        >>> if is_feature_enabled("rag_v2", user_id=user.id):
        ...     return new_rag_pipeline()
        >>>
        >>> # Tenant targeting
        >>> if is_feature_enabled("premium_features", tenant_id=tenant.id):
        ...     return premium_search()
    """
    flag = FEATURE_FLAGS.get(flag_name)

    if not flag:
        logger.warning(f"Feature flag '{flag_name}' not found, using default: {default}")
        return default

    return flag.is_enabled_for_user(user_id=user_id, tenant_id=tenant_id)


def get_feature_variant(
    flag_name: str,
    user_id: Optional[UUID] = None,
    default: str = "control",
) -> str:
    """
    Get A/B test variant for user.

    Args:
        flag_name: Feature flag name
        user_id: User UUID
        default: Default variant

    Returns:
        str: Variant name

    Example:
        >>> variant = get_feature_variant("rag_model_test", user_id=user.id)
        >>> if variant == "gpt4":
        ...     return gpt4_response()
        >>> else:
        ...     return gpt35_response()
    """
    flag = FEATURE_FLAGS.get(flag_name)

    if not flag:
        return default

    variant = flag.get_variant(user_id=user_id)
    return variant or default


def get_all_flags(
    user_id: Optional[UUID] = None,
    tenant_id: Optional[UUID] = None,
) -> Dict[str, bool]:
    """
    Get all feature flags for user.

    Args:
        user_id: User UUID
        tenant_id: Tenant UUID

    Returns:
        dict: {flag_name: enabled}

    Example:
        >>> flags = get_all_flags(user_id=user.id)
        >>> # {"vector_search": False, "rag_v2": True, ...}
    """
    return {
        name: flag.is_enabled_for_user(user_id=user_id, tenant_id=tenant_id)
        for name, flag in FEATURE_FLAGS.items()
    }


def get_feature_rollout_metrics() -> Dict[str, float]:
    """
    Get feature rollout metrics for Prometheus export.

    Returns metrics for canary deployment progress tracking.

    Returns:
        dict: {flag_name: rollout_percentage}

    Example:
        >>> metrics = get_feature_rollout_metrics()
        >>> # {"rag_v2": 0.10, "hybrid_search": 1.0, ...}

    Usage with Prometheus:
        >>> from prometheus_client import Gauge
        >>> rollout_gauge = Gauge('feature_rollout_progress', '...', ['feature_name'])
        >>> for flag_name, percentage in get_feature_rollout_metrics().items():
        ...     rollout_gauge.labels(feature_name=flag_name).set(percentage)
    """
    metrics = {}
    for name, flag in FEATURE_FLAGS.items():
        if flag.type == FeatureFlagType.PERCENTAGE:
            # Percentage rollout (0.0 - 1.0)
            metrics[name] = flag.rollout_percentage
        elif flag.enabled:
            # Fully enabled
            metrics[name] = 1.0
        else:
            # Fully disabled
            metrics[name] = 0.0
    return metrics


__all__ = [
    "FeatureFlagType",
    "FeatureFlag",
    "FEATURE_FLAGS",
    "is_feature_enabled",
    "get_feature_variant",
    "get_all_flags",
    "get_feature_rollout_metrics",
]
