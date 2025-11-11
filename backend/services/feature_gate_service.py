"""
Feature Gate Service - Harvey/Legora CTO-Level Feature Flag Management
=======================================================================

Production-grade feature flag service for controlled feature rollouts.

SORUMLULU

U:
-----------
- Feature flag deerlendirmesi (tenant, user, context-based)
- Gradual rollout (percentage-based enablement)
- A/B testing support
- Feature flag override (admin/testing)
- Caching & performance optimization
- Audit logging (KVKK-compliant)
- Multi-tenant isolation

KVKK UYUMLULUK:
--------------
 Feature flag decisions logged without PII
 User IDs hashed in logs
 No user names, emails, or personal data in flags
L PII never stored in feature flag metadata

WHY FEATURE GATES?
-----------------
Without: All-or-nothing deployments  production incidents  rollback chaos
With: Gradual rollouts  A/B testing  kill switches  Harvey-level operational safety

Impact: Safe feature deployments with instant rollback! 

ARCHITECTURE:
------------
[Feature Flag Request]
         
[FeatureGateService.is_enabled(flag_name, context)]
         
[1. Check Override] (admin override for testing)
         
[2. Check Cache] (Redis cache for performance)
         
[3. Evaluate Rules] (tenant, user, rollout percentage)
         
[4. Log Decision] (audit trail)
         
[Return: True/False]

FEATURE FLAG TYPES:
------------------
1. **Boolean Flags**: Simple on/off switches
2. **Percentage Rollout**: Gradual rollout (0-100%)
3. **Tenant Whitelist**: Specific tenants enabled
4. **User Whitelist**: Specific users enabled
5. **Context-Based**: Custom rules (e.g., practice_area, region)

EVALUATION RULES:
----------------
Priority order:
1. Override (admin/testing)
2. Kill switch (emergency disable)
3. Tenant whitelist
4. User whitelist
5. Percentage rollout
6. Default value

USAGE:
-----
```python
from backend.services.feature_gate_service import FeatureGateService, FeatureContext

service = FeatureGateService()

# Simple check
enabled = await service.is_enabled(
    flag_name="nightly_bulk_ingestion",
    context=FeatureContext(tenant_id="acme-law-firm")
)

# With user context
enabled = await service.is_enabled(
    flag_name="ai_legal_reasoning",
    context=FeatureContext(
        tenant_id="acme-law-firm",
        user_id="user-123",
        metadata={"practice_area": "i__hukuku"}
    )
)

# Admin override (testing)
await service.set_override(
    flag_name="nightly_bulk_ingestion",
    tenant_id="acme-law-firm",
    enabled=False,
    reason="Testing scheduled job skip logic"
)
```

FEATURE FLAGS (Production):
--------------------------
- `nightly_bulk_ingestion`: Scheduled bulk document ingestion
- `weekly_compliance_report`: KVKK compliance reports
- `ai_legal_reasoning`: LLM-based legal analysis
- `workflow_engine_v2`: New workflow engine
- `slack_integration`: Slack notifications
- `teams_integration`: MS Teams notifications
- `sharepoint_webhook`: SharePoint webhooks

Author: Harvey/Legora CTO
Date: 2024-01-10
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class FeatureFlagType(str, Enum):
    """Feature flag trleri"""
    BOOLEAN = "boolean"  # Basit on/off
    PERCENTAGE = "percentage"  # Gradual rollout (0-100%)
    WHITELIST = "whitelist"  # Tenant/user whitelist
    CONTEXT_BASED = "context_based"  # Custom rules


class RolloutStatus(str, Enum):
    """Rollout durumu"""
    DISABLED = "disabled"  # Tamamen kapal1
    TESTING = "testing"  # Sadece test tenant'lar1
    GRADUAL = "gradual"  # Kademeli rollout (percentage)
    ENABLED = "enabled"  # Tamamen a1k


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class FeatureContext:
    """
    Feature flag deerlendirme context'i

    Attributes:
        tenant_id: Tenant ID (multi-tenant isolation)
        user_id: User ID (optional, for user-based flags)
        metadata: Additional context (e.g., practice_area, region)
    """
    tenant_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_hash(self) -> str:
        """
        Context hash'i (cache key iin)

        Returns:
            MD5 hash of context
        """
        key = f"{self.tenant_id}:{self.user_id or 'anonymous'}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class FeatureFlag:
    """
    Feature flag definition

    Attributes:
        name: Flag name (e.g., "nightly_bulk_ingestion")
        flag_type: Flag type (BOOLEAN, PERCENTAGE, etc.)
        status: Rollout status (DISABLED, TESTING, GRADUAL, ENABLED)
        default_enabled: Default value
        rollout_percentage: Rollout percentage (0-100)
        tenant_whitelist: Tenant IDs explicitly enabled
        tenant_blacklist: Tenant IDs explicitly disabled
        user_whitelist: User IDs explicitly enabled
        context_rules: Custom context-based rules
        description: Human-readable description
        metadata: Additional metadata
    """
    name: str
    flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN
    status: RolloutStatus = RolloutStatus.DISABLED
    default_enabled: bool = False
    rollout_percentage: int = 0  # 0-100
    tenant_whitelist: Set[str] = field(default_factory=set)
    tenant_blacklist: Set[str] = field(default_factory=set)
    user_whitelist: Set[str] = field(default_factory=set)
    context_rules: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureFlagDecision:
    """
    Feature flag evaluation decision

    Attributes:
        flag_name: Flag name
        enabled: Is feature enabled?
        reason: Why was this decision made?
        evaluation_time_ms: Evaluation duration
        context_hash: Context hash (for debugging)
    """
    flag_name: str
    enabled: bool
    reason: str
    evaluation_time_ms: float
    context_hash: str


# ============================================================================
# FEATURE GATE SERVICE
# ============================================================================


class FeatureGateService:
    """
    Feature Gate Service
    ===================

    Production-grade feature flag service with:
    - Multi-tenant isolation
    - Gradual rollout
    - A/B testing
    - Caching
    - Audit logging
    """

    def __init__(self):
        """Initialize service"""
        # In-memory feature flag registry (production: load from DB/config service)
        self._flags: Dict[str, FeatureFlag] = {}

        # Override map (admin testing)
        self._overrides: Dict[str, Dict[str, bool]] = {}  # {flag_name: {tenant_id: enabled}}

        # Kill switch (emergency disable)
        self._kill_switches: Set[str] = set()

        # Cache (in production: use Redis)
        self._cache: Dict[str, bool] = {}
        self._cache_ttl_seconds = 60

        # Initialize default flags
        self._initialize_default_flags()

        logger.info("FeatureGateService initialized")

    def _initialize_default_flags(self) -> None:
        """Initialize default production feature flags"""
        # Scheduled workflow flags
        self.register_flag(FeatureFlag(
            name="nightly_bulk_ingestion",
            flag_type=FeatureFlagType.BOOLEAN,
            status=RolloutStatus.ENABLED,
            default_enabled=True,
            description="Scheduled nightly bulk document ingestion"
        ))

        self.register_flag(FeatureFlag(
            name="weekly_compliance_report",
            flag_type=FeatureFlagType.BOOLEAN,
            status=RolloutStatus.ENABLED,
            default_enabled=True,
            description="Weekly KVKK compliance reports"
        ))

        self.register_flag(FeatureFlag(
            name="daily_index_health_check",
            flag_type=FeatureFlagType.BOOLEAN,
            status=RolloutStatus.ENABLED,
            default_enabled=True,
            description="Daily Elasticsearch/Pinecone health check"
        ))

        # AI/LLM flags (gradual rollout)
        self.register_flag(FeatureFlag(
            name="ai_legal_reasoning",
            flag_type=FeatureFlagType.PERCENTAGE,
            status=RolloutStatus.GRADUAL,
            default_enabled=False,
            rollout_percentage=80,  # 80% rollout
            description="LLM-based legal reasoning (GPT-4/Claude)"
        ))

        self.register_flag(FeatureFlag(
            name="ai_timeline_analysis",
            flag_type=FeatureFlagType.PERCENTAGE,
            status=RolloutStatus.GRADUAL,
            default_enabled=False,
            rollout_percentage=50,  # 50% rollout
            description="AI-powered timeline extraction"
        ))

        # Integration flags
        self.register_flag(FeatureFlag(
            name="slack_integration",
            flag_type=FeatureFlagType.BOOLEAN,
            status=RolloutStatus.ENABLED,
            default_enabled=True,
            description="Slack notifications and commands"
        ))

        self.register_flag(FeatureFlag(
            name="teams_integration",
            flag_type=FeatureFlagType.BOOLEAN,
            status=RolloutStatus.GRADUAL,
            default_enabled=False,
            rollout_percentage=30,  # 30% rollout
            description="MS Teams notifications"
        ))

        self.register_flag(FeatureFlag(
            name="sharepoint_webhook",
            flag_type=FeatureFlagType.WHITELIST,
            status=RolloutStatus.TESTING,
            default_enabled=False,
            tenant_whitelist={"test-tenant-1", "test-tenant-2"},
            description="SharePoint webhook triggers"
        ))

        # Experimental flags (disabled by default)
        self.register_flag(FeatureFlag(
            name="workflow_engine_v2",
            flag_type=FeatureFlagType.BOOLEAN,
            status=RolloutStatus.DISABLED,
            default_enabled=False,
            description="Next-gen workflow engine (experimental)"
        ))

        logger.info(f"Initialized {len(self._flags)} default feature flags")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    def register_flag(self, flag: FeatureFlag) -> None:
        """
        Register a feature flag

        Args:
            flag: Feature flag definition
        """
        self._flags[flag.name] = flag
        logger.debug(f"Registered feature flag: {flag.name} (status={flag.status.value})")

    async def is_enabled(
        self,
        flag_name: str,
        context: FeatureContext,
        default: bool = False
    ) -> bool:
        """
        Check if a feature is enabled

        Args:
            flag_name: Feature flag name
            context: Evaluation context (tenant, user, metadata)
            default: Default value if flag not found

        Returns:
            True if feature is enabled, False otherwise
        """
        start_time = datetime.now(timezone.utc)

        try:
            # 1. Check if flag exists
            if flag_name not in self._flags:
                logger.warning(
                    f"Feature flag '{flag_name}' not found. Using default: {default}"
                )
                return default

            flag = self._flags[flag_name]

            # 2. Check kill switch (emergency disable)
            if flag_name in self._kill_switches:
                reason = "Kill switch activated (emergency disable)"
                self._log_decision(flag_name, False, reason, context, start_time)
                return False

            # 3. Check override (admin testing)
            if flag_name in self._overrides:
                tenant_overrides = self._overrides[flag_name]
                if context.tenant_id in tenant_overrides:
                    enabled = tenant_overrides[context.tenant_id]
                    reason = f"Override by admin (enabled={enabled})"
                    self._log_decision(flag_name, enabled, reason, context, start_time)
                    return enabled

            # 4. Check cache
            cache_key = self._get_cache_key(flag_name, context)
            if cache_key in self._cache:
                enabled = self._cache[cache_key]
                reason = "Cached decision"
                self._log_decision(flag_name, enabled, reason, context, start_time)
                return enabled

            # 5. Evaluate flag rules
            enabled, reason = await self._evaluate_flag(flag, context)

            # 6. Cache decision
            self._cache[cache_key] = enabled

            # 7. Log decision
            self._log_decision(flag_name, enabled, reason, context, start_time)

            return enabled

        except Exception as e:
            logger.error(
                f"Error evaluating feature flag '{flag_name}': {e}",
                exc_info=True
            )
            # Fail open (return default) on errors
            return default

    async def _evaluate_flag(
        self,
        flag: FeatureFlag,
        context: FeatureContext
    ) -> tuple[bool, str]:
        """
        Evaluate feature flag rules

        Args:
            flag: Feature flag definition
            context: Evaluation context

        Returns:
            (enabled, reason) tuple
        """
        # 1. Check if flag is disabled
        if flag.status == RolloutStatus.DISABLED:
            return False, f"Flag status is DISABLED"

        # 2. Check tenant blacklist (explicit disable)
        if context.tenant_id in flag.tenant_blacklist:
            return False, f"Tenant '{context.tenant_id}' is blacklisted"

        # 3. Check tenant whitelist (explicit enable)
        if flag.tenant_whitelist and context.tenant_id in flag.tenant_whitelist:
            return True, f"Tenant '{context.tenant_id}' is whitelisted"

        # 4. Check user whitelist (explicit enable)
        if context.user_id and flag.user_whitelist and context.user_id in flag.user_whitelist:
            return True, f"User '{context.user_id}' is whitelisted"

        # 5. Check if fully enabled
        if flag.status == RolloutStatus.ENABLED:
            return True, "Flag status is ENABLED (100% rollout)"

        # 6. Check percentage rollout (gradual rollout)
        if flag.status == RolloutStatus.GRADUAL and flag.rollout_percentage > 0:
            # Use consistent hashing for stable rollout
            hash_value = self._hash_for_rollout(context.tenant_id, flag.name)
            enabled = hash_value < flag.rollout_percentage
            return enabled, f"Percentage rollout: {flag.rollout_percentage}% (hash={hash_value})"

        # 7. Check context-based rules
        if flag.context_rules:
            for rule in flag.context_rules:
                if self._evaluate_context_rule(rule, context):
                    return True, f"Context rule matched: {rule}"

        # 8. Default value
        return flag.default_enabled, f"Default value: {flag.default_enabled}"

    def _evaluate_context_rule(
        self,
        rule: Dict[str, Any],
        context: FeatureContext
    ) -> bool:
        """
        Evaluate a context-based rule

        Args:
            rule: Rule definition (e.g., {"practice_area": "i__hukuku"})
            context: Evaluation context

        Returns:
            True if rule matches, False otherwise
        """
        for key, expected_value in rule.items():
            actual_value = context.metadata.get(key)
            if actual_value != expected_value:
                return False
        return True

    def _hash_for_rollout(self, tenant_id: str, flag_name: str) -> int:
        """
        Consistent hash for percentage rollout

        Args:
            tenant_id: Tenant ID
            flag_name: Flag name

        Returns:
            Hash value (0-99)
        """
        key = f"{tenant_id}:{flag_name}"
        hash_bytes = hashlib.md5(key.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="little")
        return hash_int % 100

    def _get_cache_key(self, flag_name: str, context: FeatureContext) -> str:
        """
        Generate cache key

        Args:
            flag_name: Flag name
            context: Evaluation context

        Returns:
            Cache key
        """
        return f"feature_gate:{flag_name}:{context.get_hash()}"

    def _log_decision(
        self,
        flag_name: str,
        enabled: bool,
        reason: str,
        context: FeatureContext,
        start_time: datetime
    ) -> None:
        """
        Log feature flag decision (KVKK-compliant)

        Args:
            flag_name: Flag name
            enabled: Decision
            reason: Decision reason
            context: Evaluation context
            start_time: Evaluation start time
        """
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # KVKK: Hash user_id (no PII in logs)
        user_id_hash = None
        if context.user_id:
            user_id_hash = hashlib.md5(context.user_id.encode()).hexdigest()[:12]

        logger.info(
            f"Feature flag decision: {flag_name}={enabled} "
            f"(tenant={context.tenant_id}, user_hash={user_id_hash}, "
            f"reason='{reason}', duration={duration_ms:.2f}ms)"
        )

    # ========================================================================
    # ADMIN METHODS
    # ========================================================================

    def set_override(
        self,
        flag_name: str,
        tenant_id: str,
        enabled: bool,
        reason: str = "Admin override"
    ) -> None:
        """
        Set admin override for testing

        Args:
            flag_name: Flag name
            tenant_id: Tenant ID
            enabled: Override value
            reason: Override reason
        """
        if flag_name not in self._overrides:
            self._overrides[flag_name] = {}

        self._overrides[flag_name][tenant_id] = enabled

        logger.warning(
            f"Admin override set: {flag_name}={enabled} for tenant={tenant_id} "
            f"(reason: {reason})"
        )

    def remove_override(self, flag_name: str, tenant_id: str) -> None:
        """
        Remove admin override

        Args:
            flag_name: Flag name
            tenant_id: Tenant ID
        """
        if flag_name in self._overrides:
            self._overrides[flag_name].pop(tenant_id, None)

        logger.info(f"Admin override removed: {flag_name} for tenant={tenant_id}")

    def activate_kill_switch(self, flag_name: str, reason: str = "Emergency disable") -> None:
        """
        Activate kill switch (emergency disable)

        Args:
            flag_name: Flag name
            reason: Kill switch reason
        """
        self._kill_switches.add(flag_name)

        logger.critical(
            f"= KILL SWITCH ACTIVATED: {flag_name} (reason: {reason})"
        )

    def deactivate_kill_switch(self, flag_name: str) -> None:
        """
        Deactivate kill switch

        Args:
            flag_name: Flag name
        """
        self._kill_switches.discard(flag_name)

        logger.warning(f"Kill switch deactivated: {flag_name}")

    def clear_cache(self, flag_name: Optional[str] = None) -> None:
        """
        Clear cache

        Args:
            flag_name: Flag name (if None, clear all)
        """
        if flag_name:
            # Clear specific flag
            keys_to_remove = [k for k in self._cache if k.startswith(f"feature_gate:{flag_name}:")]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cache cleared for flag: {flag_name}")
        else:
            # Clear all
            self._cache.clear()
            logger.info("All cache cleared")

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """
        Get feature flag definition

        Args:
            flag_name: Flag name

        Returns:
            Feature flag or None
        """
        return self._flags.get(flag_name)

    def list_flags(self, status: Optional[RolloutStatus] = None) -> List[FeatureFlag]:
        """
        List all feature flags

        Args:
            status: Filter by status (optional)

        Returns:
            List of feature flags
        """
        flags = list(self._flags.values())

        if status:
            flags = [f for f in flags if f.status == status]

        return flags

    def get_enabled_flags_for_tenant(self, tenant_id: str) -> List[str]:
        """
        Get all enabled flags for a tenant

        Args:
            tenant_id: Tenant ID

        Returns:
            List of enabled flag names
        """
        enabled_flags = []

        context = FeatureContext(tenant_id=tenant_id)

        for flag_name in self._flags:
            # Synchronous check (for admin UI)
            import asyncio
            enabled = asyncio.run(self.is_enabled(flag_name, context))
            if enabled:
                enabled_flags.append(flag_name)

        return enabled_flags

    def get_rollout_stats(self, flag_name: str) -> Dict[str, Any]:
        """
        Get rollout statistics for a flag

        Args:
            flag_name: Flag name

        Returns:
            Rollout stats
        """
        flag = self.get_flag(flag_name)
        if not flag:
            return {}

        return {
            "flag_name": flag_name,
            "status": flag.status.value,
            "rollout_percentage": flag.rollout_percentage,
            "tenant_whitelist_count": len(flag.tenant_whitelist),
            "tenant_blacklist_count": len(flag.tenant_blacklist),
            "user_whitelist_count": len(flag.user_whitelist),
            "has_overrides": flag_name in self._overrides,
            "is_kill_switched": flag_name in self._kill_switches,
        }

    # ========================================================================
    # BULK OPERATIONS
    # ========================================================================

    async def evaluate_multiple(
        self,
        flag_names: List[str],
        context: FeatureContext
    ) -> Dict[str, bool]:
        """
        Evaluate multiple flags at once

        Args:
            flag_names: List of flag names
            context: Evaluation context

        Returns:
            Dict of {flag_name: enabled}
        """
        results = {}

        for flag_name in flag_names:
            results[flag_name] = await self.is_enabled(flag_name, context)

        return results

    def __repr__(self) -> str:
        return (
            f"<FeatureGateService(flags={len(self._flags)}, "
            f"overrides={len(self._overrides)}, "
            f"kill_switches={len(self._kill_switches)})>"
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global singleton instance (production: use dependency injection)
_feature_gate_service: Optional[FeatureGateService] = None


def get_feature_gate_service() -> FeatureGateService:
    """
    Get feature gate service singleton

    Returns:
        FeatureGateService instance
    """
    global _feature_gate_service

    if _feature_gate_service is None:
        _feature_gate_service = FeatureGateService()

    return _feature_gate_service
