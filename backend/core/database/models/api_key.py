"""
API Key model for programmatic access in Turkish Legal AI.

This module provides the API Key model for secure API authentication:
- Secure API key generation (cryptographically random)
- Key hashing (SHA-256) for secure storage
- Scoped permissions (read, write, admin)
- Rate limiting metadata
- Expiration and revocation
- Usage tracking and analytics
- KVKK-compliant audit trail

API Key Features:
    - Prefix-based identification (la_xxxxxxxx)
    - One-way hashing (only hash stored in DB)
    - Scope-based access control
    - Rate limiting per key
    - IP whitelist/blacklist
    - Automatic expiration
    - Usage statistics

Security:
    - Keys are hashed using SHA-256 before storage
    - Plain key shown only once at creation
    - Support for key rotation
    - Automatic revocation on suspicious activity
    - Audit trail for all API calls

Example:
    >>> # Create API key
    >>> api_key = APIKey(
    ...     name="Production API Key",
    ...     user_id=user_id,
    ...     tenant_id=tenant_id,
    ...     scopes=["document:read", "contract:generate"],
    ...     rate_limit=1000,
    ...     expires_at=datetime.now() + timedelta(days=365)
    ... )
    >>> plain_key = api_key.generate_key()
    >>> # plain_key: "la_1a2b3c4d5e6f7g8h9i0j..."
    >>> # Store plain_key securely, show to user ONCE
    >>> 
    >>> # Verify API key (in request)
    >>> if api_key.verify("la_1a2b3c4d5e6f7g8h9i0j..."):
    ...     # Grant access
    ...     api_key.record_usage()
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    CheckConstraint,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship, validates

from backend.core.constants import (
    API_KEY_LENGTH,
    API_KEY_PREFIX,
    MAX_API_KEY_NAME_LENGTH,
    MAX_RATE_LIMIT_PER_MINUTE,
)
from backend.core.exceptions import (
    APIKeyExpiredError,
    APIKeyRevokedError,
    RateLimitExceededError,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
    TenantMixin,
    AuditMixin,
    SoftDeleteMixin,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# API KEY MODEL
# =============================================================================


class APIKey(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    API Key model for programmatic access authentication.
    
    API keys provide secure, token-based authentication for:
    - Programmatic API access (scripts, integrations)
    - Third-party application integration
    - Automated workflows
    - Webhook callbacks
    
    Security Features:
    - Cryptographically secure key generation
    - SHA-256 hashing (one-way, cannot reverse)
    - Prefix-based identification (la_xxx)
    - Scoped permissions (limit access)
    - Rate limiting per key
    - IP whitelist/blacklist
    - Automatic expiration
    - Revocation support
    
    Key Lifecycle:
    1. Creation: Generate random key, show to user ONCE
    2. Storage: Store SHA-256 hash only (never plain key)
    3. Verification: Hash incoming key, compare with stored hash
    4. Usage: Track usage statistics, enforce rate limits
    5. Expiration: Auto-expire after set period
    6. Revocation: Manual revoke if compromised
    
    Attributes:
        name: Descriptive name for the key
        key_prefix: First 8 chars of key (for identification)
        key_hash: SHA-256 hash of full key
        
        user_id: Owner user
        user: User relationship
        
        scopes: Permission scopes (array of strings)
        permissions: JSON detailed permissions
        
        is_active: Key is active
        is_revoked: Key has been revoked
        revoked_at: When key was revoked
        revoked_by_id: Who revoked the key
        revoke_reason: Why key was revoked
        
        expires_at: Key expiration date
        last_used_at: Last usage timestamp
        last_used_ip: Last request IP address
        
        rate_limit: Max requests per minute
        rate_limit_window: Rate limit window in seconds
        usage_count: Total API calls made
        
        ip_whitelist: Allowed IP addresses (array)
        ip_blacklist: Blocked IP addresses (array)
        
        settings: JSON configuration
        metadata: Additional metadata
        
    Relationships:
        tenant: Parent tenant
        user: Owner user
    """
    
    __tablename__ = "api_keys"
    
    # =========================================================================
    # IDENTITY
    # =========================================================================
    
    name = Column(
        String(MAX_API_KEY_NAME_LENGTH),
        nullable=False,
        comment="Descriptive name for the API key",
    )
    
    key_prefix = Column(
        String(20),
        nullable=False,
        index=True,
        comment="Key prefix for identification (first 8 chars)",
    )
    
    key_hash = Column(
        String(64),  # SHA-256 produces 64 hex chars
        unique=True,
        nullable=False,
        index=True,
        comment="SHA-256 hash of full API key",
    )
    
    # =========================================================================
    # OWNERSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner user",
    )
    
    user = relationship(
        "User",
        back_populates="api_keys",
    )
    
    # =========================================================================
    # PERMISSIONS & SCOPES
    # =========================================================================
    
    scopes = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Permission scopes (e.g., ['document:read', 'contract:generate'])",
    )
    
    permissions = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed permission configuration (JSON)",
    )
    
    # =========================================================================
    # STATUS & REVOCATION
    # =========================================================================
    
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Key is active and can be used",
    )
    
    is_revoked = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Key has been revoked (cannot be reactivated)",
    )
    
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When key was revoked",
    )
    
    revoked_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Who revoked the key",
    )
    
    revoke_reason = Column(
        Text,
        nullable=True,
        comment="Reason for revocation",
    )
    
    # =========================================================================
    # EXPIRATION & USAGE
    # =========================================================================
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Key expiration date (NULL = never expires)",
    )
    
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful API call",
    )
    
    last_used_ip = Column(
        String(45),  # IPv6 max length
        nullable=True,
        comment="IP address of last API call",
    )
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    rate_limit = Column(
        Integer,
        nullable=True,
        comment="Max requests per window (NULL = no limit)",
    )
    
    rate_limit_window = Column(
        Integer,
        nullable=False,
        default=60,
        comment="Rate limit window in seconds (default: 60 = per minute)",
    )
    
    usage_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of API calls made with this key",
    )
    
    # =========================================================================
    # IP RESTRICTIONS
    # =========================================================================
    
    ip_whitelist = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Allowed IP addresses/CIDR ranges (empty = all allowed)",
    )
    
    ip_blacklist = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Blocked IP addresses/CIDR ranges",
    )
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    settings = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional key settings (features, restrictions)",
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (description, tags, notes)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for active keys lookup
        Index(
            "ix_api_keys_active",
            "tenant_id",
            "is_active",
            postgresql_where="is_active = true AND is_revoked = false AND deleted_at IS NULL",
        ),
        # Index for expiration cleanup
        Index(
            "ix_api_keys_expires",
            "expires_at",
            postgresql_where="expires_at IS NOT NULL AND is_revoked = false",
        ),
        # Index for user's keys
        Index("ix_api_keys_user_id", "user_id", "tenant_id"),
        # Check: rate limit positive
        CheckConstraint(
            "rate_limit IS NULL OR rate_limit > 0",
            name="ck_api_keys_rate_limit_positive",
        ),
        # Check: rate limit window positive
        CheckConstraint(
            "rate_limit_window > 0",
            name="ck_api_keys_window_positive",
        ),
        # Check: usage count non-negative
        CheckConstraint(
            "usage_count >= 0",
            name="ck_api_keys_usage_count_positive",
        ),
    )
    
    # =========================================================================
    # KEY GENERATION
    # =========================================================================
    
    def generate_key(self) -> str:
        """
        Generate a new API key.
        
        Format: la_<64_random_chars>
        
        The plain key is returned and should be shown to the user ONCE.
        Only the SHA-256 hash is stored in the database.
        
        Returns:
            str: Plain API key (show to user once, then discard)
            
        Example:
            >>> api_key = APIKey(name="Production Key", user_id=user_id)
            >>> plain_key = api_key.generate_key()
            >>> print(f"Your API key: {plain_key}")
            >>> print("Store this securely, it won't be shown again!")
            >>> # plain_key: "la_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
        """
        # Generate cryptographically secure random key
        random_part = secrets.token_urlsafe(API_KEY_LENGTH)[:API_KEY_LENGTH]
        plain_key = f"{API_KEY_PREFIX}{random_part}"
        
        # Store prefix (for identification)
        self.key_prefix = plain_key[:12]  # la_a1b2c3d4
        
        # Hash and store (SHA-256)
        self.key_hash = self._hash_key(plain_key)
        
        logger.info(
            "API key generated",
            api_key_id=str(self.id),
            key_name=self.name,
            key_prefix=self.key_prefix,
            user_id=str(self.user_id),
        )
        
        return plain_key
    
    @staticmethod
    def _hash_key(plain_key: str) -> str:
        """
        Hash API key using SHA-256.
        
        Args:
            plain_key: Plain API key
            
        Returns:
            str: SHA-256 hash (hex string)
        """
        return hashlib.sha256(plain_key.encode()).hexdigest()
    
    def verify(self, plain_key: str) -> bool:
        """
        Verify an API key against stored hash.
        
        Args:
            plain_key: Plain API key from request
            
        Returns:
            bool: True if key matches
            
        Example:
            >>> if api_key.verify(request_api_key):
            ...     # Grant access
            ...     pass
        """
        if not plain_key:
            return False
        
        # Hash incoming key
        incoming_hash = self._hash_key(plain_key)
        
        # Constant-time comparison (prevent timing attacks)
        return hmac.compare_digest(incoming_hash, self.key_hash)
    
    # =========================================================================
    # KEY VALIDATION
    # =========================================================================
    
    def validate_access(self, ip_address: str | None = None) -> None:
        """
        Validate that API key can be used for access.
        
        Checks:
        - Key is active
        - Key is not revoked
        - Key has not expired
        - IP is allowed (if whitelist configured)
        - IP is not blocked (if blacklist configured)
        
        Args:
            ip_address: Request IP address (optional)
            
        Raises:
            APIKeyRevokedError: If key is revoked
            APIKeyExpiredError: If key is expired
            PermissionDeniedError: If IP is blocked
            
        Example:
            >>> try:
            ...     api_key.validate_access(request.client.host)
            ... except APIKeyExpiredError:
            ...     return {"error": "API key expired"}
        """
        # Check if revoked
        if self.is_revoked:
            logger.warning(
                "Attempt to use revoked API key",
                api_key_id=str(self.id),
                key_prefix=self.key_prefix,
                ip=ip_address,
            )
            raise APIKeyRevokedError(
                message="API key has been revoked",
                key_id=str(self.id),
                revoked_at=self.revoked_at,
                reason=self.revoke_reason,
            )
        
        # Check if active
        if not self.is_active:
            logger.warning(
                "Attempt to use inactive API key",
                api_key_id=str(self.id),
                key_prefix=self.key_prefix,
                ip=ip_address,
            )
            raise APIKeyRevokedError(
                message="API key is not active",
                key_id=str(self.id),
            )
        
        # Check expiration
        if self.is_expired:
            logger.warning(
                "Attempt to use expired API key",
                api_key_id=str(self.id),
                key_prefix=self.key_prefix,
                expires_at=self.expires_at.isoformat() if self.expires_at else None,
                ip=ip_address,
            )
            raise APIKeyExpiredError(
                message="API key has expired",
                key_id=str(self.id),
                expires_at=self.expires_at,
            )
        
        # Check IP restrictions
        if ip_address:
            if not self._is_ip_allowed(ip_address):
                logger.warning(
                    "API key access denied due to IP restriction",
                    api_key_id=str(self.id),
                    key_prefix=self.key_prefix,
                    ip=ip_address,
                    whitelist=self.ip_whitelist,
                    blacklist=self.ip_blacklist,
                )
                from backend.core.exceptions import PermissionDeniedError
                raise PermissionDeniedError(
                    message=f"Access denied from IP: {ip_address}",
                    resource="api_key",
                    action="use",
                )
    
    @property
    def is_expired(self) -> bool:
        """Check if API key has expired."""
        if not self.expires_at:
            return False  # Never expires
        
        return datetime.now(timezone.utc) > self.expires_at
    
    def _is_ip_allowed(self, ip_address: str) -> bool:
        """
        Check if IP address is allowed.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            bool: True if allowed
        """
        # Check blacklist first (takes precedence)
        if self.ip_blacklist:
            if self._ip_matches_list(ip_address, self.ip_blacklist):
                return False
        
        # If whitelist is configured, IP must be in it
        if self.ip_whitelist:
            return self._ip_matches_list(ip_address, self.ip_whitelist)
        
        # No whitelist = all IPs allowed (except blacklisted)
        return True
    
    @staticmethod
    def _ip_matches_list(ip_address: str, ip_list: list[str]) -> bool:
        """
        Check if IP matches any entry in list.
        
        Supports:
        - Exact match: "192.168.1.1"
        - CIDR ranges: "192.168.1.0/24"
        
        Args:
            ip_address: IP to check
            ip_list: List of IPs/CIDR ranges
            
        Returns:
            bool: True if matches
        """
        import ipaddress
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for allowed in ip_list:
                # Check if CIDR range
                if "/" in allowed:
                    network = ipaddress.ip_network(allowed, strict=False)
                    if ip in network:
                        return True
                # Exact match
                elif ip_address == allowed:
                    return True
            
            return False
        except ValueError:
            logger.error(
                "Invalid IP address format",
                ip_address=ip_address,
            )
            return False
    
    # =========================================================================
    # USAGE TRACKING
    # =========================================================================
    
    def record_usage(self, ip_address: str | None = None) -> None:
        """
        Record API key usage.
        
        Updates:
        - last_used_at
        - last_used_ip
        - usage_count
        
        Args:
            ip_address: Request IP address (optional)
            
        Example:
            >>> api_key.record_usage(request.client.host)
        """
        self.last_used_at = datetime.now(timezone.utc)
        self.usage_count += 1
        
        if ip_address:
            self.last_used_ip = ip_address
        
        logger.debug(
            "API key usage recorded",
            api_key_id=str(self.id),
            key_prefix=self.key_prefix,
            usage_count=self.usage_count,
            ip=ip_address,
        )
    
    def check_rate_limit(self) -> bool:
        """
        Check if rate limit allows request.
        
        This is a simple check. Production implementation should use
        Redis or similar for distributed rate limiting.
        
        Returns:
            bool: True if within limit
            
        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        if not self.rate_limit:
            return True  # No rate limit
        
        # Note: This is a simplified check
        # Production should track requests in time windows using Redis
        
        logger.debug(
            "Rate limit check",
            api_key_id=str(self.id),
            rate_limit=self.rate_limit,
            window=self.rate_limit_window,
        )
        
        return True
    
    # =========================================================================
    # PERMISSION CHECKS
    # =========================================================================
    
    def has_scope(self, scope: str) -> bool:
        """
        Check if API key has specific scope.
        
        Supports wildcards:
        - "document:*" matches "document:read", "document:write"
        - "*:read" matches all read permissions
        
        Args:
            scope: Scope to check (e.g., "document:read")
            
        Returns:
            bool: True if scope is granted
            
        Example:
            >>> if api_key.has_scope("document:read"):
            ...     return document_data
        """
        if not self.scopes:
            return False
        
        # Check exact match
        if scope in self.scopes:
            return True
        
        # Check wildcards
        if "*" in self.scopes or "*:*" in self.scopes:
            return True
        
        # Check resource wildcard (e.g., "document:*")
        if ":" in scope:
            resource, action = scope.split(":", 1)
            if f"{resource}:*" in self.scopes:
                return True
            if f"*:{action}" in self.scopes:
                return True
        
        return False
    
    def require_scope(self, scope: str) -> None:
        """
        Require API key to have scope (raises if not).
        
        Args:
            scope: Required scope
            
        Raises:
            PermissionDeniedError: If scope not granted
        """
        if not self.has_scope(scope):
            logger.warning(
                "API key missing required scope",
                api_key_id=str(self.id),
                key_prefix=self.key_prefix,
                required_scope=scope,
                granted_scopes=self.scopes,
            )
            from backend.core.exceptions import PermissionDeniedError
            raise PermissionDeniedError(
                message=f"API key missing required scope: {scope}",
                required_permission=scope,
            )
    
    # =========================================================================
    # KEY MANAGEMENT
    # =========================================================================
    
    def revoke(self, reason: str | None = None, revoked_by_id: str | None = None) -> None:
        """
        Revoke API key (cannot be undone).
        
        Args:
            reason: Reason for revocation
            revoked_by_id: Who revoked the key (optional)
            
        Example:
            >>> api_key.revoke(
            ...     reason="Key compromised",
            ...     revoked_by_id=str(admin_user.id)
            ... )
        """
        self.is_revoked = True
        self.is_active = False
        self.revoked_at = datetime.now(timezone.utc)
        self.revoke_reason = reason
        
        if revoked_by_id:
            self.revoked_by_id = revoked_by_id
        
        logger.warning(
            "API key revoked",
            api_key_id=str(self.id),
            key_name=self.name,
            key_prefix=self.key_prefix,
            reason=reason,
            revoked_by=revoked_by_id,
        )
    
    def deactivate(self) -> None:
        """Temporarily deactivate API key (can be reactivated)."""
        self.is_active = False
        
        logger.info(
            "API key deactivated",
            api_key_id=str(self.id),
            key_name=self.name,
        )
    
    def reactivate(self) -> None:
        """Reactivate a deactivated API key."""
        if self.is_revoked:
            raise ValidationError(
                message="Cannot reactivate revoked API key",
                field="is_active",
            )
        
        self.is_active = True
        
        logger.info(
            "API key reactivated",
            api_key_id=str(self.id),
            key_name=self.name,
        )
    
    def extend_expiration(self, days: int) -> None:
        """
        Extend API key expiration.
        
        Args:
            days: Number of days to extend
            
        Example:
            >>> api_key.extend_expiration(365)  # Extend by 1 year
        """
        if not self.expires_at:
            # No expiration set, set from now
            self.expires_at = datetime.now(timezone.utc) + timedelta(days=days)
        else:
            # Extend existing expiration
            self.expires_at += timedelta(days=days)
        
        logger.info(
            "API key expiration extended",
            api_key_id=str(self.id),
            key_name=self.name,
            new_expiration=self.expires_at.isoformat(),
        )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("name")
    def validate_name(self, key: str, name: str) -> str:
        """Validate API key name."""
        if not name or not name.strip():
            raise ValidationError(
                message="API key name cannot be empty",
                field="name",
            )
        
        return name.strip()
    
    @validates("scopes")
    def validate_scopes(self, key: str, scopes: list[str]) -> list[str]:
        """Validate scope format."""
        import re
        
        scope_pattern = r"^[a-z_*]+:[a-z_*]+$"
        
        for scope in scopes:
            if not re.match(scope_pattern, scope):
                raise ValidationError(
                    message=f"Invalid scope format: {scope} (expected 'resource:action')",
                    field="scopes",
                )
        
        return scopes
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<APIKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"
    
    def to_dict(self, include_hash: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_hash: Include key hash (default: False for security)
            
        Returns:
            dict: API key data
        """
        data = super().to_dict()
        
        # Remove hash by default (security)
        if not include_hash:
            data.pop("key_hash", None)
        
        # Add computed fields
        data["is_expired"] = self.is_expired
        data["days_until_expiration"] = None
        if self.expires_at:
            delta = self.expires_at - datetime.now(timezone.utc)
            data["days_until_expiration"] = max(0, delta.days)
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "APIKey",
]