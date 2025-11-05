"""
Session model for user authentication and session management in Turkish Legal AI.

This module provides the Session model for secure user session tracking:
- JWT-based session management
- Device fingerprinting and tracking
- Geographic location tracking (IP-based)
- Concurrent session limits
- Automatic session expiration
- Force logout/revocation support
- KVKK-compliant session audit

Session Architecture:
    - Access token (short-lived, 15-60 min)
    - Refresh token (long-lived, 7-30 days)
    - Session fingerprint (device + browser)
    - IP tracking and geolocation
    - Activity timestamps
    - Revocation support (logout, security)

Security Features:
    - Token rotation on refresh
    - Device fingerprinting
    - Suspicious activity detection
    - Geographic anomaly detection
    - Concurrent session limits
    - Automatic cleanup of expired sessions
    - Force logout capability

Example:
    >>> # Create session on login
    >>> session = Session(
    ...     user_id=user_id,
    ...     tenant_id=tenant_id,
    ...     device_fingerprint="Chrome-Mac-192.168.1.1",
    ...     ip_address="192.168.1.100",
    ...     user_agent="Mozilla/5.0...",
    ...     expires_at=datetime.now() + timedelta(days=7)
    ... )
    >>> access_token = session.generate_access_token()
    >>> refresh_token = session.generate_refresh_token()
    >>> 
    >>> # Verify session on request
    >>> if session.is_valid():
    ...     session.update_activity()
    ...     # Grant access
"""

import hashlib
import secrets
import uuid
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
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, validates

from backend.core.config.security import security_config
from backend.core.constants import (
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
    MAX_CONCURRENT_SESSIONS,
    SESSION_INACTIVITY_TIMEOUT_MINUTES,
)
from backend.core.exceptions import (
    SessionExpiredError,
    SessionRevokedError,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
    TenantMixin,
    SoftDeleteMixin,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# SESSION MODEL
# =============================================================================


class Session(Base, BaseModelMixin, TenantMixin, SoftDeleteMixin):
    """
    Session model for user authentication and tracking.
    
    Sessions represent authenticated user connections:
    - Created on successful login
    - Tracked per device/browser
    - Contains access and refresh tokens
    - Expires after inactivity or explicit logout
    - Can be revoked for security reasons
    
    Session Lifecycle:
    1. Login: Create session with access + refresh tokens
    2. Activity: Update last_activity on each API call
    3. Refresh: Generate new access token using refresh token
    4. Expiration: Auto-expire after inactivity or max lifetime
    5. Logout: Revoke session and invalidate tokens
    
    Token Strategy:
    - Access Token: Short-lived (15-60 min), used for API calls
    - Refresh Token: Long-lived (7-30 days), used to get new access token
    - Token Rotation: New refresh token issued on each refresh
    
    Attributes:
        user_id: Authenticated user
        user: User relationship
        
        session_token: Unique session identifier
        access_token_jti: JWT ID of current access token
        refresh_token_jti: JWT ID of current refresh token
        
        device_fingerprint: Device/browser fingerprint
        device_name: User-friendly device name
        device_type: Device type (desktop, mobile, tablet)
        
        ip_address: Client IP address
        ip_location: Geographic location (city, country)
        user_agent: HTTP User-Agent header
        
        is_active: Session is active
        is_revoked: Session has been revoked
        revoked_at: When session was revoked
        revoke_reason: Why session was revoked
        
        last_activity_at: Last API call timestamp
        expires_at: Session expiration timestamp
        
        login_method: How user logged in (password, sso, mfa)
        metadata: Additional session data
        
    Relationships:
        tenant: Parent tenant
        user: Authenticated user
    """
    
    __tablename__ = "sessions"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Authenticated user",
    )
    
    user = relationship(
        "User",
        back_populates="sessions",
    )
    
    # =========================================================================
    # SESSION TOKENS
    # =========================================================================
    
    session_token = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique session identifier (for tracking)",
    )
    
    access_token_jti = Column(
        String(36),  # UUID length
        nullable=True,
        index=True,
        comment="JWT ID (jti) of current access token",
    )
    
    refresh_token_jti = Column(
        String(36),  # UUID length
        nullable=True,
        index=True,
        comment="JWT ID (jti) of current refresh token",
    )
    
    # =========================================================================
    # DEVICE INFORMATION
    # =========================================================================
    
    device_fingerprint = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Device/browser fingerprint hash",
    )
    
    device_name = Column(
        String(255),
        nullable=True,
        comment="User-friendly device name (e.g., 'Chrome on MacBook Pro')",
    )
    
    device_type = Column(
        String(50),
        nullable=True,
        comment="Device type (desktop, mobile, tablet)",
    )
    
    # =========================================================================
    # NETWORK INFORMATION
    # =========================================================================
    
    ip_address = Column(
        String(45),  # IPv6 max length
        nullable=False,
        index=True,
        comment="Client IP address",
    )
    
    ip_location = Column(
        JSONB,
        nullable=True,
        comment="Geographic location (city, country, lat/lon)",
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="HTTP User-Agent header",
    )
    
    # =========================================================================
    # SESSION STATUS
    # =========================================================================
    
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Session is active",
    )
    
    is_revoked = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Session has been revoked (logout or security)",
    )
    
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When session was revoked",
    )
    
    revoke_reason = Column(
        String(255),
        nullable=True,
        comment="Reason for revocation (logout, security, admin)",
    )
    
    # =========================================================================
    # ACTIVITY TRACKING
    # =========================================================================
    
    last_activity_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="Last activity timestamp (updated on each API call)",
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Session expiration timestamp",
    )
    
    # =========================================================================
    # LOGIN INFORMATION
    # =========================================================================
    
    login_method = Column(
        String(50),
        nullable=False,
        default="password",
        comment="Login method (password, sso, oauth, mfa)",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional session metadata (browser, OS, screen resolution)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for active sessions lookup
        Index(
            "ix_sessions_active",
            "user_id",
            "is_active",
            postgresql_where="is_active = true AND is_revoked = false AND deleted_at IS NULL",
        ),
        # Index for session cleanup (expired sessions)
        Index(
            "ix_sessions_expired",
            "expires_at",
            postgresql_where="is_revoked = false",
        ),
        # Index for device tracking
        Index("ix_sessions_device", "user_id", "device_fingerprint"),
        # Index for IP tracking
        Index("ix_sessions_ip", "ip_address", "user_id"),
        # Composite index for token lookup
        Index("ix_sessions_tokens", "access_token_jti", "refresh_token_jti"),
    )
    
    # =========================================================================
    # SESSION CREATION
    # =========================================================================
    
    @classmethod
    def create_session(
        cls,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID,
        ip_address: str,
        user_agent: str | None = None,
        device_info: dict[str, Any] | None = None,
        login_method: str = "password",
    ) -> "Session":
        """
        Create a new session on user login.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant UUID
            ip_address: Client IP address
            user_agent: HTTP User-Agent header
            device_info: Additional device information
            login_method: How user logged in
            
        Returns:
            Session: New session instance
            
        Example:
            >>> session = Session.create_session(
            ...     user_id=user.id,
            ...     tenant_id=user.tenant_id,
            ...     ip_address="192.168.1.100",
            ...     user_agent=request.headers.get("User-Agent"),
            ...     device_info={
            ...         "browser": "Chrome",
            ...         "os": "macOS",
            ...         "device_type": "desktop"
            ...     },
            ...     login_method="password"
            ... )
        """
        device_info = device_info or {}
        
        # Generate session token
        session_token = secrets.token_urlsafe(48)
        
        # Generate device fingerprint
        fingerprint_data = f"{user_agent or ''}-{device_info.get('browser', '')}-{device_info.get('os', '')}"
        device_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32]
        
        # Parse device info
        device_name = cls._parse_device_name(user_agent, device_info)
        device_type = device_info.get("device_type", "unknown")
        
        # Set expiration (default: 7 days)
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        
        # Create session
        session = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            session_token=session_token,
            device_fingerprint=device_fingerprint,
            device_name=device_name,
            device_type=device_type,
            ip_address=ip_address,
            user_agent=user_agent,
            login_method=login_method,
            expires_at=expires_at,
            metadata=device_info,
        )
        
        logger.info(
            "Session created",
            session_id=str(session.id),
            user_id=str(user_id),
            ip_address=ip_address,
            device_type=device_type,
            login_method=login_method,
        )
        
        return session
    
    @staticmethod
    def _parse_device_name(user_agent: str | None, device_info: dict[str, Any]) -> str:
        """
        Parse user-friendly device name from User-Agent and device info.
        
        Args:
            user_agent: HTTP User-Agent string
            device_info: Additional device information
            
        Returns:
            str: Device name (e.g., "Chrome on MacBook Pro")
        """
        browser = device_info.get("browser", "Unknown Browser")
        os = device_info.get("os", "Unknown OS")
        
        # Simple parsing (production should use user-agents library)
        if user_agent:
            if "Chrome" in user_agent:
                browser = "Chrome"
            elif "Firefox" in user_agent:
                browser = "Firefox"
            elif "Safari" in user_agent and "Chrome" not in user_agent:
                browser = "Safari"
            elif "Edge" in user_agent:
                browser = "Edge"
            
            if "Mac" in user_agent:
                os = "macOS"
            elif "Windows" in user_agent:
                os = "Windows"
            elif "Linux" in user_agent:
                os = "Linux"
            elif "Android" in user_agent:
                os = "Android"
            elif "iOS" in user_agent or "iPhone" in user_agent:
                os = "iOS"
        
        return f"{browser} on {os}"
    
    # =========================================================================
    # TOKEN MANAGEMENT
    # =========================================================================
    
    def generate_access_token(self) -> str:
        """
        Generate JWT access token for this session.
        
        Returns:
            str: JWT access token
            
        Example:
            >>> access_token = session.generate_access_token()
            >>> # Use in Authorization header: Bearer <access_token>
        """
        # Generate unique JWT ID
        jti = str(uuid.uuid4())
        self.access_token_jti = jti
        
        # Create access token
        token = security_config.create_access_token(
            subject=str(self.user_id),
            additional_claims={
                "session_id": str(self.id),
                "tenant_id": str(self.tenant_id),
                "jti": jti,
                "type": "access",
            },
            expires_delta=timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        
        logger.debug(
            "Access token generated",
            session_id=str(self.id),
            user_id=str(self.user_id),
            jti=jti,
        )
        
        return token
    
    def generate_refresh_token(self) -> str:
        """
        Generate JWT refresh token for this session.
        
        Returns:
            str: JWT refresh token
            
        Example:
            >>> refresh_token = session.generate_refresh_token()
            >>> # Store securely, use to get new access token
        """
        # Generate unique JWT ID
        jti = str(uuid.uuid4())
        self.refresh_token_jti = jti
        
        # Create refresh token
        token = security_config.create_refresh_token(
            subject=str(self.user_id),
            additional_claims={
                "session_id": str(self.id),
                "tenant_id": str(self.tenant_id),
                "jti": jti,
                "type": "refresh",
            },
        )
        
        logger.debug(
            "Refresh token generated",
            session_id=str(self.id),
            user_id=str(self.user_id),
            jti=jti,
        )
        
        return token
    
    def rotate_tokens(self) -> tuple[str, str]:
        """
        Rotate access and refresh tokens (security best practice).
        
        Called when:
        - User refreshes access token
        - Suspicious activity detected
        - Regular rotation interval
        
        Returns:
            tuple: (new_access_token, new_refresh_token)
            
        Example:
            >>> access_token, refresh_token = session.rotate_tokens()
            >>> # Return new tokens to client
        """
        access_token = self.generate_access_token()
        refresh_token = self.generate_refresh_token()
        
        logger.info(
            "Session tokens rotated",
            session_id=str(self.id),
            user_id=str(self.user_id),
        )
        
        return access_token, refresh_token
    
    # =========================================================================
    # SESSION VALIDATION
    # =========================================================================
    
    def is_valid(self) -> bool:
        """
        Check if session is valid for use.
        
        Checks:
        - Session is active
        - Session is not revoked
        - Session has not expired
        - Session has not been inactive too long
        
        Returns:
            bool: True if session is valid
        """
        if not self.is_active:
            return False
        
        if self.is_revoked:
            return False
        
        now = datetime.now(timezone.utc)
        
        # Check expiration
        if now > self.expires_at:
            return False
        
        # Check inactivity timeout
        inactivity_limit = timedelta(minutes=SESSION_INACTIVITY_TIMEOUT_MINUTES)
        if now - self.last_activity_at > inactivity_limit:
            return False
        
        return True
    
    def validate(self) -> None:
        """
        Validate session (raises if invalid).
        
        Raises:
            SessionExpiredError: If session expired
            SessionRevokedError: If session revoked
            
        Example:
            >>> try:
            ...     session.validate()
            ... except SessionExpiredError:
            ...     return {"error": "Session expired, please login again"}
        """
        if self.is_revoked:
            logger.warning(
                "Attempt to use revoked session",
                session_id=str(self.id),
                user_id=str(self.user_id),
            )
            raise SessionRevokedError(
                message="Session has been revoked",
                session_id=str(self.id),
                revoked_at=self.revoked_at,
                reason=self.revoke_reason,
            )
        
        if not self.is_active:
            raise SessionExpiredError(
                message="Session is not active",
                session_id=str(self.id),
            )
        
        now = datetime.now(timezone.utc)
        
        # Check expiration
        if now > self.expires_at:
            logger.info(
                "Session expired",
                session_id=str(self.id),
                user_id=str(self.user_id),
                expires_at=self.expires_at.isoformat(),
            )
            raise SessionExpiredError(
                message="Session has expired",
                session_id=str(self.id),
                expires_at=self.expires_at,
            )
        
        # Check inactivity
        inactivity_limit = timedelta(minutes=SESSION_INACTIVITY_TIMEOUT_MINUTES)
        if now - self.last_activity_at > inactivity_limit:
            logger.info(
                "Session expired due to inactivity",
                session_id=str(self.id),
                user_id=str(self.user_id),
                last_activity=self.last_activity_at.isoformat(),
            )
            raise SessionExpiredError(
                message="Session expired due to inactivity",
                session_id=str(self.id),
            )
    
    def verify_token_jti(self, jti: str, token_type: str = "access") -> bool:
        """
        Verify that token JTI matches current session token.
        
        Args:
            jti: JWT ID from token
            token_type: Type of token (access or refresh)
            
        Returns:
            bool: True if JTI matches
        """
        if token_type == "access":
            return self.access_token_jti == jti
        elif token_type == "refresh":
            return self.refresh_token_jti == jti
        else:
            return False
    
    # =========================================================================
    # ACTIVITY TRACKING
    # =========================================================================
    
    def update_activity(self, ip_address: str | None = None) -> None:
        """
        Update session activity timestamp.
        
        Called on each API request to keep session alive.
        
        Args:
            ip_address: Current IP address (for anomaly detection)
            
        Example:
            >>> @app.middleware("http")
            ... async def track_activity(request, call_next):
            ...     if session := get_session(request):
            ...         session.update_activity(request.client.host)
            ...     return await call_next(request)
        """
        self.last_activity_at = datetime.now(timezone.utc)
        
        # Check for IP change (potential security issue)
        if ip_address and ip_address != self.ip_address:
            logger.warning(
                "Session IP address changed",
                session_id=str(self.id),
                user_id=str(self.user_id),
                old_ip=self.ip_address,
                new_ip=ip_address,
            )
            
            # Update IP for tracking
            self.ip_address = ip_address
            
            # Could trigger additional security checks here
            # (e.g., require MFA, send alert email)
        
        logger.debug(
            "Session activity updated",
            session_id=str(self.id),
            user_id=str(self.user_id),
        )
    
    def extend_expiration(self, days: int = 7) -> None:
        """
        Extend session expiration.
        
        Args:
            days: Number of days to extend
            
        Example:
            >>> session.extend_expiration(30)  # Extend by 30 days
        """
        self.expires_at += timedelta(days=days)
        
        logger.info(
            "Session expiration extended",
            session_id=str(self.id),
            user_id=str(self.user_id),
            new_expiration=self.expires_at.isoformat(),
        )
    
    # =========================================================================
    # SESSION TERMINATION
    # =========================================================================
    
    def revoke(self, reason: str = "logout") -> None:
        """
        Revoke session (logout).
        
        Args:
            reason: Reason for revocation
            
        Example:
            >>> # User logout
            >>> session.revoke(reason="logout")
            >>> 
            >>> # Security revocation
            >>> session.revoke(reason="suspicious_activity")
        """
        self.is_revoked = True
        self.is_active = False
        self.revoked_at = datetime.now(timezone.utc)
        self.revoke_reason = reason
        
        logger.info(
            "Session revoked",
            session_id=str(self.id),
            user_id=str(self.user_id),
            reason=reason,
        )
    
    @classmethod
    def revoke_user_sessions(
        cls,
        user_id: uuid.UUID,
        reason: str = "security",
        except_session_id: uuid.UUID | None = None,
    ) -> int:
        """
        Revoke all sessions for a user (e.g., password change).
        
        Args:
            user_id: User UUID
            reason: Revocation reason
            except_session_id: Keep this session active (optional)
            
        Returns:
            int: Number of sessions revoked
            
        Example:
            >>> # Revoke all sessions after password change
            >>> count = Session.revoke_user_sessions(
            ...     user_id=user.id,
            ...     reason="password_changed",
            ...     except_session_id=current_session.id  # Keep current
            ... )
            >>> print(f"Revoked {count} other sessions")
        """
        # This would be implemented in a service layer with DB session
        # Placeholder for demonstration
        logger.info(
            "Revoking all user sessions",
            user_id=str(user_id),
            reason=reason,
            except_session_id=str(except_session_id) if except_session_id else None,
        )
        return 0
    
    # =========================================================================
    # SECURITY CHECKS
    # =========================================================================
    
    def check_concurrent_sessions(self, max_sessions: int = MAX_CONCURRENT_SESSIONS) -> bool:
        """
        Check if user has too many concurrent sessions.
        
        Args:
            max_sessions: Maximum allowed concurrent sessions
            
        Returns:
            bool: True if within limit
        """
        # This would query active sessions for user
        # Placeholder for demonstration
        return True
    
    def detect_suspicious_activity(self) -> bool:
        """
        Detect suspicious session activity.
        
        Checks:
        - Rapid IP changes
        - Geographic anomalies
        - Unusual access patterns
        
        Returns:
            bool: True if suspicious activity detected
        """
        # Placeholder for sophisticated anomaly detection
        # Production would check:
        # - IP geolocation changes
        # - Access pattern anomalies
        # - Impossible travel scenarios
        return False
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("ip_address")
    def validate_ip_address(self, key: str, ip_address: str) -> str:
        """Validate IP address format."""
        import ipaddress
        
        try:
            ipaddress.ip_address(ip_address)
            return ip_address
        except ValueError:
            raise ValidationError(
                message=f"Invalid IP address format: {ip_address}",
                field="ip_address",
            )
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Session(id={self.id}, user_id={self.user_id}, device={self.device_name})>"
    
    def to_dict(self, include_tokens: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_tokens: Include token JTIs (default: False for security)
            
        Returns:
            dict: Session data
        """
        data = super().to_dict()
        
        # Remove sensitive data by default
        if not include_tokens:
            data.pop("access_token_jti", None)
            data.pop("refresh_token_jti", None)
            data.pop("session_token", None)
        
        # Add computed fields
        data["is_valid"] = self.is_valid()
        data["time_since_activity"] = (
            datetime.now(timezone.utc) - self.last_activity_at
        ).total_seconds()
        data["time_until_expiration"] = (
            self.expires_at - datetime.now(timezone.utc)
        ).total_seconds()
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Session",
]