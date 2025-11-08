"""
RBAC Policy Engine for Turkish Legal AI.

This module implements a flexible policy evaluation system:
- Context-aware authorization (ABAC-style)
- Ownership-based policies
- Team/organization membership policies
- Conditional policies (time, IP, feature flags)
- Policy chaining and composition
- Optional OPA (Open Policy Agent) integration

Policy Types:
    1. Role-Based: User has required role
    2. Permission-Based: User has required permission
    3. Ownership-Based: User owns the resource
    4. Team-Based: User is member of resource's team
    5. Conditional: Custom conditions (time, location, etc.)

Policy Evaluation Flow:
    Request → Context → Policies → Decision (ALLOW/DENY)

Example:
    >>> from backend.security.rbac.policies import PolicyEngine, PolicyContext
    >>>
    >>> # Create context
    >>> ctx = PolicyContext(
    ...     user_id=user_id,
    ...     resource_type="document",
    ...     resource_id=doc_id,
    ...     action="approve",
    ...     tenant_id=tenant_id
    ... )
    >>>
    >>> # Evaluate policy
    >>> engine = PolicyEngine(db_session)
    >>> allowed = await engine.evaluate(ctx)
    >>>
    >>> if allowed:
    ...     approve_document(doc_id)
"""

import datetime
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.logging import get_logger
from backend.security.rbac.permissions import PermissionService
from backend.security.rbac.roles import RoleService

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# POLICY AUDIT TRAIL (JSONL logging for compliance)
# =============================================================================

POLICY_AUDIT_ENABLED = True  # Can be disabled via config
POLICY_AUDIT_PATH = "/var/log/legalai/rbac_audit.jsonl"  # JSONL format for analysis


def _log_policy_evaluation(
    context: "PolicyContext",
    decision: bool,
    duration_ms: float,
    evaluation_details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log policy evaluation to JSONL audit trail.

    Format: One JSON object per line for easy parsing/analysis.
    """
    if not POLICY_AUDIT_ENABLED:
        return

    try:
        audit_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_id": str(context.user_id),
            "tenant_id": str(context.tenant_id),
            "resource_type": context.resource_type,
            "resource_id": str(context.resource_id) if context.resource_id else None,
            "action": context.action,
            "decision": "allow" if decision else "deny",
            "duration_ms": round(duration_ms, 2),
            "ip_address": context.ip_address,
            "user_agent": context.user_agent,
            "evaluation_details": evaluation_details or {},
        }

        # Append to JSONL file (one JSON per line)
        audit_path = Path(POLICY_AUDIT_PATH)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        with open(audit_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

    except Exception as e:
        logger.error(f"Failed to write policy audit log: {e}")


# =============================================================================
# OPENTELEMETRY (span injection for tracing)
# =============================================================================

# OpenTelemetry tracer (initialized externally)
otel_tracer = None  # TODO: Initialize from backend.core.tracing


def _create_policy_span(context: "PolicyContext"):
    """
    Create OpenTelemetry span for policy evaluation.

    Returns context manager for span.
    """
    if otel_tracer is None:
        # Return dummy context manager
        from contextlib import nullcontext
        return nullcontext()

    # Create span with context
    return otel_tracer.start_as_current_span(
        "rbac.policy.evaluate",
        attributes={
            "user_id": str(context.user_id),
            "tenant_id": str(context.tenant_id),
            "resource_type": context.resource_type,
            "action": context.action,
        }
    )


# =============================================================================
# POLICY ENUMS
# =============================================================================


class PolicyDecision(str, Enum):
    """Policy evaluation decision."""

    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"  # Policy doesn't apply


class PolicyEffect(str, Enum):
    """Policy effect when condition matches."""

    ALLOW = "allow"
    DENY = "deny"


# =============================================================================
# POLICY CONTEXT
# =============================================================================


@dataclass
class PolicyContext:
    """
    Context for policy evaluation.

    Contains all information needed to make authorization decision:
    - Who is making the request (user_id)
    - What resource (resource_type, resource_id)
    - What action (action)
    - Where (tenant_id, ip_address)
    - When (timestamp)
    - Additional metadata
    """

    # Core identity
    user_id: UUID
    tenant_id: UUID

    # Resource information
    resource_type: str  # e.g., "document", "contract"
    action: str  # e.g., "read", "write", "approve"
    resource_id: Optional[UUID] = None

    # Request metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    # Additional context
    organization_id: Optional[UUID] = None
    team_id: Optional[UUID] = None
    session_id: Optional[str] = None

    # Resource ownership
    resource_owner_id: Optional[UUID] = None
    resource_team_id: Optional[UUID] = None

    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate context after initialization."""
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.resource_type:
            raise ValueError("resource_type is required")
        if not self.action:
            raise ValueError("action is required")


# =============================================================================
# POLICY ENGINE
# =============================================================================


class PolicyEngine:
    """
    RBAC policy evaluation engine.

    Evaluates authorization requests using:
    - Role-based policies
    - Permission-based policies
    - Ownership policies
    - Team membership policies
    - Custom conditional policies
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize policy engine.

        Args:
            db: Async database session
        """
        self.db = db
        self.logger = logger
        self.permission_service = PermissionService(db)
        self.role_service = RoleService(db)

        # Custom policy handlers
        self.custom_policies: List[Callable] = []

    # =========================================================================
    # MAIN EVALUATION
    # =========================================================================

    async def evaluate(self, context: PolicyContext) -> bool:
        """
        Evaluate all policies for given context.

        Evaluation order (any DENY fails immediately):
        1. Superadmin check (wildcard permission)
        2. Permission-based policies
        3. Ownership policies
        4. Team membership policies
        5. Custom policies

        Logs to JSONL audit trail + OpenTelemetry span + Prometheus metrics.

        Args:
            context: Policy evaluation context

        Returns:
            True if allowed, False if denied
        """
        start_time = time.time()
        evaluation_details = {}

        # OpenTelemetry span for distributed tracing
        with _create_policy_span(context):
            self.logger.debug(
                f"Evaluating policy: user={context.user_id}, "
                f"resource={context.resource_type}:{context.resource_id}, "
                f"action={context.action}"
            )

            # 1. Check superadmin (wildcard permission)
            if await self._check_superadmin(context):
                self.logger.info(f"ALLOW: Superadmin access for user {context.user_id}")
                evaluation_details["reason"] = "superadmin"
                decision = True
            else:
                # 2. Permission-based check
                if await self._check_permission(context):
                    evaluation_details["has_permission"] = True
                else:
                    self.logger.info(
                        f"DENY: User {context.user_id} lacks permission "
                        f"{context.resource_type}:{context.action}"
                    )
                    evaluation_details["reason"] = "no_permission"
                    decision = False

                    # Early return on permission failure
                    duration_ms = (time.time() - start_time) * 1000

                    # Log to JSONL audit trail
                    _log_policy_evaluation(context, decision, duration_ms, evaluation_details)

                    # Prometheus metric: policy evaluation
                    # rbac_policy_evaluation_total.labels(
                    #     decision="deny",
                    #     resource_type=context.resource_type,
                    #     action=context.action
                    # ).inc()

                    return decision

                # 3. Ownership check (if resource has owner)
                if context.resource_owner_id:
                    if not await self._check_ownership(context):
                        self.logger.info(
                            f"DENY: User {context.user_id} doesn't own resource "
                            f"{context.resource_id}"
                        )
                        evaluation_details["reason"] = "not_owner"
                        decision = False

                        duration_ms = (time.time() - start_time) * 1000
                        _log_policy_evaluation(context, decision, duration_ms, evaluation_details)
                        return decision

                # 4. Team membership check (if resource belongs to team)
                if context.resource_team_id:
                    if not await self._check_team_membership(context):
                        self.logger.info(
                            f"DENY: User {context.user_id} not in team "
                            f"{context.resource_team_id}"
                        )
                        evaluation_details["reason"] = "not_team_member"
                        decision = False

                        duration_ms = (time.time() - start_time) * 1000
                        _log_policy_evaluation(context, decision, duration_ms, evaluation_details)
                        return decision

                # 5. Custom policies
                for custom_policy in self.custom_policies:
                    policy_decision = await custom_policy(context, self.db)
                    if policy_decision == PolicyDecision.DENY:
                        self.logger.info(
                            f"DENY: Custom policy denied access for user {context.user_id}"
                        )
                        evaluation_details["reason"] = "custom_policy"
                        decision = False

                        duration_ms = (time.time() - start_time) * 1000
                        _log_policy_evaluation(context, decision, duration_ms, evaluation_details)
                        return decision

                # All checks passed
                self.logger.info(f"ALLOW: User {context.user_id} granted access")
                evaluation_details["reason"] = "all_checks_passed"
                decision = True

            # Log successful evaluation
            duration_ms = (time.time() - start_time) * 1000
            _log_policy_evaluation(context, decision, duration_ms, evaluation_details)

            # Prometheus metric
            decision_label = "allow" if decision else "deny"
            # rbac_policy_evaluation_total.labels(
            #     decision=decision_label,
            #     resource_type=context.resource_type,
            #     action=context.action
            # ).inc()
            # rbac_policy_evaluation_duration_seconds.labels(
            #     resource_type=context.resource_type
            # ).observe(duration_ms / 1000)

            return decision

    # =========================================================================
    # POLICY CHECKS
    # =========================================================================

    async def _check_superadmin(self, context: PolicyContext) -> bool:
        """
        Check if user has superadmin (wildcard) permission.

        Args:
            context: Policy context

        Returns:
            True if user is superadmin
        """
        return await self.permission_service.user_has_permission(
            user_id=context.user_id,
            resource="*",
            action="*",
        )

    async def _check_permission(self, context: PolicyContext) -> bool:
        """
        Check if user has required permission.

        Args:
            context: Policy context

        Returns:
            True if user has permission
        """
        return await self.permission_service.user_has_permission(
            user_id=context.user_id,
            resource=context.resource_type,
            action=context.action,
        )

    async def _check_ownership(self, context: PolicyContext) -> bool:
        """
        Check if user owns the resource.

        Args:
            context: Policy context

        Returns:
            True if user owns resource
        """
        if not context.resource_owner_id:
            return True  # No ownership requirement

        return context.user_id == context.resource_owner_id

    async def _check_team_membership(self, context: PolicyContext) -> bool:
        """
        Check if user is member of resource's team.

        Args:
            context: Policy context

        Returns:
            True if user is team member
        """
        if not context.resource_team_id:
            return True  # No team requirement

        # TODO: Implement actual team membership check
        # This would query the team_members table
        # For now, allow if user's team_id matches
        return context.team_id == context.resource_team_id

    # =========================================================================
    # POLICY REGISTRATION
    # =========================================================================

    def register_policy(self, policy_fn: Callable) -> None:
        """
        Register custom policy function.

        Policy function signature:
            async def my_policy(context: PolicyContext, db: AsyncSession) -> PolicyDecision

        Args:
            policy_fn: Async policy evaluation function

        Example:
            >>> async def time_based_policy(ctx: PolicyContext, db: AsyncSession):
            ...     # Only allow access during business hours
            ...     if 9 <= ctx.timestamp.hour <= 17:
            ...         return PolicyDecision.ALLOW
            ...     return PolicyDecision.DENY
            >>>
            >>> engine.register_policy(time_based_policy)
        """
        self.custom_policies.append(policy_fn)
        self.logger.info(f"Registered custom policy: {policy_fn.__name__}")

    def clear_policies(self) -> None:
        """Clear all registered custom policies."""
        self.custom_policies.clear()

    # =========================================================================
    # BULK EVALUATION
    # =========================================================================

    async def evaluate_batch(
        self,
        contexts: List[PolicyContext],
    ) -> Dict[UUID, bool]:
        """
        Evaluate multiple policy contexts in batch.

        Args:
            contexts: List of policy contexts

        Returns:
            Dictionary mapping user_id to decision

        Example:
            >>> contexts = [
            ...     PolicyContext(user_id=user1, resource_type="doc", action="read"),
            ...     PolicyContext(user_id=user2, resource_type="doc", action="write"),
            ... ]
            >>> results = await engine.evaluate_batch(contexts)
            >>> # {user1: True, user2: False}
        """
        results = {}

        for context in contexts:
            decision = await self.evaluate(context)
            results[context.user_id] = decision

        return results


# =============================================================================
# PRE-DEFINED POLICIES
# =============================================================================


async def ownership_policy(
    context: PolicyContext,
    db: AsyncSession,
) -> PolicyDecision:
    """
    Policy: User must own the resource.

    Args:
        context: Policy context
        db: Database session

    Returns:
        ALLOW if user owns resource, DENY otherwise
    """
    if not context.resource_owner_id:
        return PolicyDecision.ABSTAIN

    if context.user_id == context.resource_owner_id:
        return PolicyDecision.ALLOW

    return PolicyDecision.DENY


async def business_hours_policy(
    context: PolicyContext,
    db: AsyncSession,
) -> PolicyDecision:
    """
    Policy: Only allow access during business hours (9 AM - 5 PM).

    Args:
        context: Policy context
        db: Database session

    Returns:
        ALLOW during business hours, DENY otherwise
    """
    hour = context.timestamp.hour

    if 9 <= hour <= 17:
        return PolicyDecision.ALLOW

    return PolicyDecision.DENY


async def ip_whitelist_policy(
    context: PolicyContext,
    db: AsyncSession,
    allowed_ips: Optional[Set[str]] = None,
) -> PolicyDecision:
    """
    Policy: Only allow access from whitelisted IPs.

    Args:
        context: Policy context
        db: Database session
        allowed_ips: Set of allowed IP addresses

    Returns:
        ALLOW if IP is whitelisted, DENY otherwise
    """
    if not allowed_ips:
        return PolicyDecision.ABSTAIN

    if not context.ip_address:
        return PolicyDecision.DENY

    if context.ip_address in allowed_ips:
        return PolicyDecision.ALLOW

    return PolicyDecision.DENY


# =============================================================================
# POLICY BUILDER (Fluent API)
# =============================================================================


class PolicyBuilder:
    """
    Fluent API for building complex policies.

    Example:
        >>> policy = (
        ...     PolicyBuilder()
        ...     .require_permission("document", "read")
        ...     .require_ownership()
        ...     .during_business_hours()
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize policy builder."""
        self.conditions: List[Callable] = []

    def require_permission(self, resource: str, action: str) -> "PolicyBuilder":
        """Add permission requirement."""

        async def check(ctx: PolicyContext, db: AsyncSession) -> PolicyDecision:
            perm_svc = PermissionService(db)
            has_perm = await perm_svc.user_has_permission(ctx.user_id, resource, action)
            return PolicyDecision.ALLOW if has_perm else PolicyDecision.DENY

        self.conditions.append(check)
        return self

    def require_ownership(self) -> "PolicyBuilder":
        """Add ownership requirement."""
        self.conditions.append(ownership_policy)
        return self

    def during_business_hours(self) -> "PolicyBuilder":
        """Add business hours requirement."""
        self.conditions.append(business_hours_policy)
        return self

    def build(self) -> Callable:
        """Build final policy function."""

        async def policy(ctx: PolicyContext, db: AsyncSession) -> PolicyDecision:
            for condition in self.conditions:
                decision = await condition(ctx, db)
                if decision == PolicyDecision.DENY:
                    return PolicyDecision.DENY
            return PolicyDecision.ALLOW

        return policy
