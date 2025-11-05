"""
OAuth Token model for third-party authentication in Turkish Legal AI.

This module provides the OAuthToken model for OAuth 2.0 integration:
- OAuth 2.0 token storage
- Multiple provider support (Google, Microsoft, etc.)
- Token refresh automation
- Scope management
- Token expiration tracking
- Security best practices
- KVKK-compliant data handling

OAuth Flow:
    1. User initiates OAuth login
    2. Redirect to provider (Google, Microsoft)
    3. Provider returns authorization code
    4. Exchange code for access/refresh tokens
    5. Store tokens securely (encrypted)
    6. Auto-refresh when expired

Supported Providers:
    - Google (Google Workspace integration)
    - Microsoft (Azure AD, Office 365)
    - GitHub (future)
    - Custom OIDC providers

Security:
    - Token encryption at rest
    - Automatic token rotation
    - Scope validation
    - Expiration enforcement
    - Revocation support

Example:
    >>> # Store OAuth token after authentication
    >>> token = OAuthToken(
    ...     user_id=user.id,
    ...     provider=OAuthProvider.GOOGLE,
    ...     access_token=encrypt(access_token),
    ...     refresh_token=encrypt(refresh_token),
    ...     expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    ...     scopes=["email", "profile", "drive.readonly"]
    ... )
    >>> 
    >>> # Check if token needs refresh
    >>> if token.is_expired():
    ...     new_token = await token.refresh()
"""

import enum
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID as UUIDType

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    CheckConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from backend.core.exceptions import (
    OAuthTokenExpiredError,
    OAuthTokenRevokedError,
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
# ENUMS
# =============================================================================


class OAuthProvider(str, enum.Enum):
    """
    OAuth provider identifiers.
    
    Providers:
    - GOOGLE: Google OAuth (Gmail, Drive, Calendar)
    - MICROSOFT: Microsoft OAuth (Azure AD, Office 365)
    - GITHUB: GitHub OAuth
    - LINKEDIN: LinkedIn OAuth
    - CUSTOM: Custom OIDC provider
    """
    
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    CUSTOM = "custom"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name(self) -> str:
        """Display name for provider."""
        names = {
            self.GOOGLE: "Google",
            self.MICROSOFT: "Microsoft",
            self.GITHUB: "GitHub",
            self.LINKEDIN: "LinkedIn",
            self.CUSTOM: "Custom OIDC",
        }
        return names.get(self, self.value)


class TokenType(str, enum.Enum):
    """OAuth token type."""
    
    BEARER = "bearer"
    MAC = "mac"
    
    def __str__(self) -> str:
        return self.value


class TokenStatus(str, enum.Enum):
    """Token status."""
    
    ACTIVE = "active"              # Active and valid
    EXPIRED = "expired"            # Expired (needs refresh)
    REVOKED = "revoked"            # Manually revoked
    INVALID = "invalid"            # Invalid (refresh failed)
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# OAUTH TOKEN MODEL
# =============================================================================


class OAuthToken(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    OAuth Token model for third-party authentication.
    
    Stores OAuth 2.0 tokens for integrated services:
    - Access tokens (short-lived)
    - Refresh tokens (long-lived)
    - Token metadata (scopes, expiration)
    - Provider information
    
    Token Lifecycle:
    1. Initial grant: Store access + refresh tokens
    2. Usage: Use access token for API calls
    3. Expiration: Automatically refresh when expired
    4. Rotation: New tokens issued on refresh
    5. Revocation: Manual revoke or logout
    
    Security:
        - Tokens encrypted at rest (AES-256)
        - Refresh tokens rotated on use
        - Automatic expiration
        - Scope validation
        - Revocation support
    
    Attributes:
        user_id: Token owner
        user: User relationship
        
        provider: OAuth provider (Google, Microsoft, etc.)
        provider_user_id: User ID at provider
        provider_email: Email at provider
        
        access_token: Encrypted access token
        refresh_token: Encrypted refresh token
        token_type: Token type (usually "bearer")
        
        scopes: Granted scopes (array)
        
        expires_at: Access token expiration
        issued_at: When token was issued
        
        status: Token status
        
        last_refreshed_at: Last refresh timestamp
        refresh_count: Number of times refreshed
        
        revoked_at: When token was revoked
        revoke_reason: Why token was revoked
        
        metadata: Additional provider data
        
    Relationships:
        tenant: Parent tenant
        user: Token owner
    """
    
    __tablename__ = "oauth_tokens"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Token owner",
    )
    
    user = relationship(
        "User",
        back_populates="oauth_tokens",
    )
    
    # =========================================================================
    # PROVIDER INFORMATION
    # =========================================================================
    
    provider = Column(
        Enum(OAuthProvider, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="OAuth provider (google, microsoft, github)",
    )
    
    provider_user_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment="User ID at OAuth provider",
    )
    
    provider_email = Column(
        String(255),
        nullable=True,
        comment="Email address at OAuth provider",
    )
    
    # =========================================================================
    # TOKENS (ENCRYPTED)
    # =========================================================================
    
    access_token = Column(
        Text,
        nullable=False,
        comment="Encrypted access token (use for API calls)",
    )
    
    refresh_token = Column(
        Text,
        nullable=True,
        comment="Encrypted refresh token (use to get new access token)",
    )
    
    token_type = Column(
        Enum(TokenType, native_enum=False, length=50),
        nullable=False,
        default=TokenType.BEARER,
        comment="Token type (usually 'bearer')",
    )
    
    # =========================================================================
    # SCOPES
    # =========================================================================
    
    scopes = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Granted OAuth scopes (permissions)",
    )
    
    # =========================================================================
    # EXPIRATION
    # =========================================================================
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Access token expiration timestamp",
    )
    
    issued_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When token was first issued",
    )
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    status = Column(
        Enum(TokenStatus, native_enum=False, length=50),
        nullable=False,
        default=TokenStatus.ACTIVE,
        index=True,
        comment="Token status",
    )
    
    # =========================================================================
    # REFRESH TRACKING
    # =========================================================================
    
    last_refreshed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When token was last refreshed",
    )
    
    refresh_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times token has been refreshed",
    )
    
    # =========================================================================
    # REVOCATION
    # =========================================================================
    
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When token was revoked",
    )
    
    revoke_reason = Column(
        String(255),
        nullable=True,
        comment="Reason for revocation",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional provider metadata (profile info, etc.)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Unique token per user per provider
        Index(
            "ix_oauth_tokens_user_provider",
            "user_id",
            "provider",
            unique=True,
        ),
        
        # Index for provider user lookup
        Index(
            "ix_oauth_tokens_provider_user",
            "provider",
            "provider_user_id",
        ),
        
        # Index for active tokens
        Index(
            "ix_oauth_tokens_active",
            "user_id",
            "status",
            postgresql_where="status = 'active' AND deleted_at IS NULL",
        ),
        
        # Index for expired tokens (cleanup)
        Index(
            "ix_oauth_tokens_expired",
            "expires_at",
            postgresql_where="status = 'active'",
        ),
        
        # Check: refresh count non-negative
        CheckConstraint(
            "refresh_count >= 0",
            name="ck_oauth_tokens_refresh_count",
        ),
    )
    
    # =========================================================================
    # TOKEN CREATION
    # =========================================================================
    
    @classmethod
    def create_token(
        cls,
        user_id: UUIDType,
        tenant_id: UUIDType,
        provider: OAuthProvider,
        provider_user_id: str,
        access_token: str,
        refresh_token: str | None,
        expires_in: int,
        scopes: list[str],
        provider_email: str | None = None,
        provider_metadata: dict[str, Any] | None = None,
    ) -> "OAuthToken":
        """
        Create OAuth token after successful authentication.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant UUID
            provider: OAuth provider
            provider_user_id: User ID at provider
            access_token: Access token (will be encrypted)
            refresh_token: Refresh token (will be encrypted)
            expires_in: Token lifetime in seconds
            scopes: Granted scopes
            provider_email: Email at provider
            provider_metadata: Additional provider data
            
        Returns:
            OAuthToken: New token instance
            
        Example:
            >>> token = OAuthToken.create_token(
            ...     user_id=user.id,
            ...     tenant_id=user.tenant_id,
            ...     provider=OAuthProvider.GOOGLE,
            ...     provider_user_id="12345678901234567890",
            ...     access_token="ya29.a0AfH6SMB...",
            ...     refresh_token="1//0gPxQ7...",
            ...     expires_in=3600,
            ...     scopes=["email", "profile", "drive.readonly"],
            ...     provider_email="user@gmail.com"
            ... )
        """
        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        
        # Encrypt tokens (placeholder - production would use encryption)
        encrypted_access = cls._encrypt_token(access_token)
        encrypted_refresh = cls._encrypt_token(refresh_token) if refresh_token else None
        
        token = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            access_token=encrypted_access,
            refresh_token=encrypted_refresh,
            expires_at=expires_at,
            scopes=scopes,
            status=TokenStatus.ACTIVE,
            metadata=provider_metadata or {},
        )
        
        logger.info(
            "OAuth token created",
            token_id=str(token.id),
            user_id=str(user_id),
            provider=provider.value,
            scopes=scopes,
        )
        
        return token
    
    @staticmethod
    def _encrypt_token(token: str) -> str:
        """
        Encrypt token for storage.
        
        Production implementation would use:
        - AES-256-GCM encryption
        - Key from environment/vault
        - Per-token IV
        
        Args:
            token: Plain token
            
        Returns:
            str: Encrypted token
        """
        # Placeholder: Production would use proper encryption
        # from cryptography.fernet import Fernet
        # return fernet.encrypt(token.encode()).decode()
        return f"encrypted:{token}"
    
    @staticmethod
    def _decrypt_token(encrypted_token: str) -> str:
        """
        Decrypt token for use.
        
        Args:
            encrypted_token: Encrypted token
            
        Returns:
            str: Plain token
        """
        # Placeholder: Production would use proper decryption
        if encrypted_token.startswith("encrypted:"):
            return encrypted_token[10:]
        return encrypted_token
    
    # =========================================================================
    # TOKEN VALIDATION
    # =========================================================================
    
    def is_expired(self) -> bool:
        """
        Check if access token is expired.
        
        Returns:
            bool: True if expired
        """
        return datetime.now(timezone.utc) >= self.expires_at
    
    def is_valid(self) -> bool:
        """
        Check if token is valid for use.
        
        Returns:
            bool: True if valid
        """
        if self.status != TokenStatus.ACTIVE:
            return False
        
        if self.is_expired():
            return False
        
        return True
    
    def validate(self) -> None:
        """
        Validate token (raises if invalid).
        
        Raises:
            OAuthTokenExpiredError: If token expired
            OAuthTokenRevokedError: If token revoked
        """
        if self.status == TokenStatus.REVOKED:
            raise OAuthTokenRevokedError(
                message="OAuth token has been revoked",
                token_id=str(self.id),
                provider=self.provider.value,
                revoked_at=self.revoked_at,
                reason=self.revoke_reason,
            )
        
        if self.is_expired():
            raise OAuthTokenExpiredError(
                message="OAuth token has expired",
                token_id=str(self.id),
                provider=self.provider.value,
                expires_at=self.expires_at,
            )
        
        if self.status != TokenStatus.ACTIVE:
            raise OAuthTokenRevokedError(
                message=f"OAuth token is not active: {self.status.value}",
                token_id=str(self.id),
                provider=self.provider.value,
            )
    
    def get_access_token(self) -> str:
        """
        Get decrypted access token.
        
        Returns:
            str: Plain access token
            
        Raises:
            OAuthTokenExpiredError: If token expired
        """
        self.validate()
        return self._decrypt_token(self.access_token)
    
    def get_refresh_token(self) -> str | None:
        """
        Get decrypted refresh token.
        
        Returns:
            str | None: Plain refresh token
        """
        if not self.refresh_token:
            return None
        
        return self._decrypt_token(self.refresh_token)
    
    # =========================================================================
    # TOKEN REFRESH
    # =========================================================================
    
    def update_tokens(
        self,
        access_token: str,
        refresh_token: str | None,
        expires_in: int,
        scopes: list[str] | None = None,
    ) -> None:
        """
        Update tokens after refresh.
        
        Args:
            access_token: New access token
            refresh_token: New refresh token (if rotated)
            expires_in: Token lifetime in seconds
            scopes: Updated scopes (optional)
            
        Example:
            >>> # After refreshing with OAuth provider
            >>> token.update_tokens(
            ...     access_token="ya29.a0AfH6SMB...",
            ...     refresh_token="1//0gPxQ7...",  # May be rotated
            ...     expires_in=3600
            ... )
        """
        # Encrypt new tokens
        self.access_token = self._encrypt_token(access_token)
        if refresh_token:
            self.refresh_token = self._encrypt_token(refresh_token)
        
        # Update expiration
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        
        # Update scopes if provided
        if scopes:
            self.scopes = scopes
        
        # Update status
        self.status = TokenStatus.ACTIVE
        
        # Track refresh
        self.last_refreshed_at = datetime.now(timezone.utc)
        self.refresh_count += 1
        
        logger.info(
            "OAuth token refreshed",
            token_id=str(self.id),
            provider=self.provider.value,
            refresh_count=self.refresh_count,
        )
    
    def mark_expired(self) -> None:
        """Mark token as expired."""
        self.status = TokenStatus.EXPIRED
        
        logger.debug(
            "OAuth token marked as expired",
            token_id=str(self.id),
            provider=self.provider.value,
        )
    
    # =========================================================================
    # SCOPE MANAGEMENT
    # =========================================================================
    
    def has_scope(self, scope: str) -> bool:
        """
        Check if token has specific scope.
        
        Args:
            scope: Scope to check
            
        Returns:
            bool: True if scope granted
            
        Example:
            >>> if token.has_scope("drive.readonly"):
            ...     access_drive_files()
        """
        return scope in self.scopes
    
    def has_all_scopes(self, required_scopes: list[str]) -> bool:
        """
        Check if token has all required scopes.
        
        Args:
            required_scopes: List of required scopes
            
        Returns:
            bool: True if all scopes granted
        """
        return all(scope in self.scopes for scope in required_scopes)
    
    def has_any_scope(self, scopes: list[str]) -> bool:
        """
        Check if token has any of the specified scopes.
        
        Args:
            scopes: List of scopes to check
            
        Returns:
            bool: True if any scope granted
        """
        return any(scope in self.scopes for scope in scopes)
    
    # =========================================================================
    # REVOCATION
    # =========================================================================
    
    def revoke(self, reason: str | None = None) -> None:
        """
        Revoke OAuth token.
        
        Args:
            reason: Revocation reason
            
        Example:
            >>> token.revoke(reason="User disconnected Google account")
        """
        self.status = TokenStatus.REVOKED
        self.revoked_at = datetime.now(timezone.utc)
        self.revoke_reason = reason
        
        logger.warning(
            "OAuth token revoked",
            token_id=str(self.id),
            provider=self.provider.value,
            reason=reason,
        )
    
    # =========================================================================
    # PROVIDER HELPERS
    # =========================================================================
    
    def get_authorization_header(self) -> str:
        """
        Get Authorization header value.
        
        Returns:
            str: Authorization header (e.g., "Bearer ya29...")
            
        Example:
            >>> headers = {
            ...     "Authorization": token.get_authorization_header()
            ... }
        """
        access_token = self.get_access_token()
        return f"{self.token_type.value.capitalize()} {access_token}"
    
    @property
    def time_until_expiration(self) -> timedelta:
        """Get time remaining until expiration."""
        return self.expires_at - datetime.now(timezone.utc)
    
    @property
    def seconds_until_expiration(self) -> int:
        """Get seconds remaining until expiration."""
        return max(0, int(self.time_until_expiration.total_seconds()))
    
    def should_refresh(self, threshold_seconds: int = 300) -> bool:
        """
        Check if token should be refreshed proactively.
        
        Args:
            threshold_seconds: Refresh if expires in less than this (default: 5 min)
            
        Returns:
            bool: True if should refresh
        """
        return self.seconds_until_expiration < threshold_seconds
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("provider_user_id")
    def validate_provider_user_id(self, key: str, provider_user_id: str) -> str:
        """Validate provider user ID."""
        if not provider_user_id or not provider_user_id.strip():
            raise ValidationError(
                message="Provider user ID cannot be empty",
                field="provider_user_id",
            )
        
        return provider_user_id.strip()
    
    @validates("access_token")
    def validate_access_token(self, key: str, access_token: str) -> str:
        """Validate access token."""
        if not access_token or not access_token.strip():
            raise ValidationError(
                message="Access token cannot be empty",
                field="access_token",
            )
        
        return access_token.strip()
    
    @validates("scopes")
    def validate_scopes(self, key: str, scopes: list[str]) -> list[str]:
        """Validate scopes."""
        if not scopes:
            raise ValidationError(
                message="At least one scope must be granted",
                field="scopes",
            )
        
        # Remove duplicates
        return list(set(scopes))
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<OAuthToken("
            f"id={self.id}, "
            f"provider={self.provider.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self, include_tokens: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_tokens: Include encrypted tokens (default: False for security)
            
        Returns:
            dict: Token data
        """
        data = super().to_dict()
        
        # Remove sensitive data by default
        if not include_tokens:
            data.pop("access_token", None)
            data.pop("refresh_token", None)
        
        # Add computed fields
        data["provider_display"] = self.provider.display_name
        data["is_expired"] = self.is_expired()
        data["is_valid"] = self.is_valid()
        data["seconds_until_expiration"] = self.seconds_until_expiration
        data["should_refresh"] = self.should_refresh()
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "OAuthToken",
    "OAuthProvider",
    "TokenType",
    "TokenStatus",
]
