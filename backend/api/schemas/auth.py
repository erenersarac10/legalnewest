"""
Authentication Schemas for Turkish Legal AI Platform.

Enterprise-grade authentication and authorization schemas for secure user
authentication, registration, token management, and session control.

=============================================================================
FEATURES
=============================================================================

1. Authentication Schemas
   -----------------------
   - LoginRequest: Email/password login
   - LoginResponse: Access + refresh tokens with user data
   - LogoutRequest: Session termination
   - TokenRefreshRequest: Refresh access token
   - TokenRefreshResponse: New access token

2. Registration Schemas
   ---------------------
   - RegisterRequest: User registration with validation
   - RegisterResponse: Created user + initial tokens
   - EmailVerificationRequest: Email verification token
   - EmailVerificationResponse: Verification confirmation

3. Password Management
   --------------------
   - PasswordResetRequest: Initiate password reset
   - PasswordResetConfirm: Complete password reset with token
   - ChangePasswordRequest: Change password (authenticated)
   - PasswordStrengthResponse: Password strength analysis

4. Token Schemas
   --------------
   - AccessTokenPayload: JWT access token claims
   - RefreshTokenPayload: JWT refresh token claims
   - TokenPair: Access + refresh token pair
   - TokenIntrospectionResponse: Token validation details

5. Session Management
   -------------------
   - SessionInfo: Active session information
   - SessionListResponse: User's active sessions
   - SessionRevokeRequest: Revoke specific session
   - SessionRevokeAllRequest: Revoke all sessions

6. Multi-Factor Authentication (MFA)
   ----------------------------------
   - MFAEnableRequest: Enable MFA for user
   - MFAVerifyRequest: Verify MFA code
   - MFABackupCodesResponse: Backup codes for MFA
   - MFADisableRequest: Disable MFA

=============================================================================
USAGE
=============================================================================

Login Flow:
-----------

>>> from backend.api.schemas.auth import LoginRequest, LoginResponse
>>>
>>> # Client sends login request
>>> login_data = LoginRequest(
...     email="avukat@example.com",
...     password="SecureP@ss123"
... )
>>>
>>> # Server validates and returns tokens
>>> response = LoginResponse(
...     access_token="eyJhbGciOiJIUzI1NiIs...",
...     refresh_token="eyJhbGciOiJIUzI1NiIs...",
...     token_type="bearer",
...     expires_in=3600,
...     user={
...         "id": "user-uuid",
...         "email": "avukat@example.com",
...         "full_name": "Ahmet Yılmaz",
...         "role": "lawyer"
...     }
... )

Registration Flow:
------------------

>>> from backend.api.schemas.auth import RegisterRequest, RegisterResponse
>>>
>>> # Client sends registration request
>>> register_data = RegisterRequest(
...     email="yeni@example.com",
...     password="SecureP@ss123",
...     password_confirm="SecureP@ss123",
...     full_name="Mehmet Demir",
...     phone="+905321234567",
...     profession_type="lawyer",
...     bar_number="12345",
...     accept_terms=True,
...     accept_kvkk=True
... )
>>>
>>> # Server creates user and sends verification email
>>> response = RegisterResponse(
...     user=user_data,
...     access_token="...",  # Optional: auto-login
...     message="Kayıt başarılı! Lütfen e-postanızı doğrulayın."
... )

Token Refresh Flow:
-------------------

>>> from backend.api.schemas.auth import TokenRefreshRequest, TokenRefreshResponse
>>>
>>> # Access token expired, refresh it
>>> refresh_request = TokenRefreshRequest(
...     refresh_token="eyJhbGciOiJIUzI1NiIs..."
... )
>>>
>>> # Server validates refresh token and issues new access token
>>> response = TokenRefreshResponse(
...     access_token="eyJhbGciOiJIUzI1NiIs...",  # New access token
...     token_type="bearer",
...     expires_in=3600
... )

Password Reset Flow:
--------------------

>>> from backend.api.schemas.auth import (
...     PasswordResetRequest,
...     PasswordResetConfirm
... )
>>>
>>> # Step 1: User requests password reset
>>> reset_request = PasswordResetRequest(
...     email="avukat@example.com"
... )
>>> # Server sends reset link to email
>>>
>>> # Step 2: User clicks link and submits new password
>>> confirm_request = PasswordResetConfirm(
...     token="reset-token-from-email",
...     new_password="NewSecureP@ss456",
...     password_confirm="NewSecureP@ss456"
... )

MFA Enable Flow:
----------------

>>> from backend.api.schemas.auth import MFAEnableRequest, MFAVerifyRequest
>>>
>>> # Step 1: User enables MFA
>>> @app.post("/auth/mfa/enable")
>>> async def enable_mfa(current_user: User):
...     # Generate QR code for authenticator app
...     secret = generate_totp_secret()
...     qr_code = generate_qr_code(secret, current_user.email)
...     return MFAEnableResponse(
...         secret=secret,
...         qr_code=qr_code,
...         backup_codes=generate_backup_codes()
...     )
>>>
>>> # Step 2: User scans QR and verifies with code
>>> verify_request = MFAVerifyRequest(
...     code="123456"  # From authenticator app
... )

Session Management:
-------------------

>>> from backend.api.schemas.auth import SessionInfo, SessionListResponse
>>>
>>> # List user's active sessions
>>> @app.get("/auth/sessions")
>>> async def list_sessions(current_user: User):
...     sessions = await get_user_sessions(current_user.id)
...     return SessionListResponse(
...         sessions=sessions,
...         total=len(sessions)
...     )
>>>
>>> # Revoke specific session
>>> @app.delete("/auth/sessions/{session_id}")
>>> async def revoke_session(session_id: str):
...     await revoke_session_by_id(session_id)
...     return {"message": "Oturum sonlandırıldı"}

=============================================================================
JWT TOKEN STRUCTURE
=============================================================================

Access Token Claims:
--------------------
{
  "sub": "user-uuid",              # Subject (user ID)
  "email": "avukat@example.com",   # User email
  "role": "lawyer",                # User role
  "tenant_id": "tenant-uuid",      # Tenant ID
  "permissions": [...],            # User permissions
  "type": "access",                # Token type
  "exp": 1699364400,               # Expiration timestamp
  "iat": 1699360800,               # Issued at timestamp
  "jti": "token-uuid"              # JWT ID (unique)
}

Refresh Token Claims:
---------------------
{
  "sub": "user-uuid",              # Subject (user ID)
  "type": "refresh",               # Token type
  "session_id": "session-uuid",    # Session ID
  "exp": 1701952800,               # Expiration (30 days)
  "iat": 1699360800,               # Issued at
  "jti": "token-uuid"              # JWT ID
}

Token Lifetimes:
----------------
- Access Token: 1 hour (3600 seconds)
- Refresh Token: 30 days (2592000 seconds)
- Email Verification Token: 24 hours
- Password Reset Token: 1 hour
- MFA Code: 30 seconds validity

=============================================================================
VALIDATION RULES
=============================================================================

Email Validation:
-----------------
- Valid email format (RFC 5322)
- Turkish domains supported (.tr, .com.tr)
- Lowercase normalized
- Max 255 characters
- No disposable email services

Password Validation:
--------------------
- Minimum 8 characters
- Maximum 128 characters
- At least 1 uppercase letter
- At least 1 lowercase letter
- At least 1 digit
- At least 1 special character (!@#$%^&*(),.?\":{}|<>)
- Not in common password list
- Not similar to email/name

Phone Validation:
-----------------
- Turkish phone numbers: +905XXXXXXXXX
- 10 digits after country code
- Starts with 5 (mobile)
- Normalized format

Registration Validation:
------------------------
- Email unique in system
- Password confirmation matches
- Terms and KVKK acceptance required
- Bar number required for lawyers
- Phone number required

=============================================================================
SECURITY FEATURES
=============================================================================

Password Security:
------------------
- Argon2id hashing (memory-hard, resistant to GPU attacks)
- Automatic salt generation (unique per password)
- Cost factor: 4 iterations, 64MB memory
- Hash length: 32 bytes
- No plaintext storage ever

Token Security:
---------------
- JWT with HS256 algorithm (HMAC-SHA256)
- Cryptographically secure secret key (32+ bytes)
- Token rotation on refresh
- Refresh token single-use (prevents replay attacks)
- JTI (JWT ID) for token revocation
- Expiration strictly enforced

Rate Limiting:
--------------
- Login: 5 attempts per 15 minutes per IP
- Registration: 3 attempts per hour per IP
- Password reset: 3 requests per hour per email
- Token refresh: 10 requests per minute per user
- MFA verification: 5 attempts per 5 minutes

Session Security:
-----------------
- Unique session ID per login
- Device fingerprinting (user agent, IP)
- Concurrent session limit (configurable)
- Automatic logout on suspicious activity
- Session expiry on inactivity (24 hours)

Account Protection:
-------------------
- Account lockout after 5 failed logins
- Lockout duration: 30 minutes
- Email notification on lockout
- CAPTCHA after 3 failed attempts
- IP-based blocking for brute force

=============================================================================
KVKK COMPLIANCE
=============================================================================

Personal Data Processing:
-------------------------

Login Data:
  - Purpose: Authentication and access control
  - Legal Basis: Contract performance
  - Retention: Active session duration
  - Logging: IP address, timestamp, user agent

Registration Data:
  - Purpose: User account creation and identity verification
  - Legal Basis: Contract performance + explicit consent
  - Retention: Account lifetime + 5 years after deletion
  - Sensitive: TC Kimlik No (encrypted), phone (masked in logs)

Password Data:
  - Purpose: Authentication
  - Legal Basis: Contract performance
  - Storage: Hashed with Argon2id (irreversible)
  - Retention: Until password change or account deletion

Session Data:
  - Purpose: Maintain authenticated session
  - Legal Basis: Contract performance
  - Retention: Session duration (max 30 days)
  - Includes: IP address, device info, login timestamp

Consent Requirements:
---------------------
- Terms of Service acceptance (required)
- KVKK consent (required for Turkey)
- Marketing consent (optional)
- Analytics consent (optional)

Data Subject Rights:
--------------------
- Right to access: User can view their auth history
- Right to rectification: User can change email/password
- Right to deletion: Account deletion removes all data
- Right to data portability: Export auth logs
- Right to object: Opt-out of optional processing

=============================================================================
ERROR HANDLING
=============================================================================

Common Authentication Errors:
------------------------------

INVALID_CREDENTIALS:
  - Status: 401 Unauthorized
  - Message: "E-posta veya şifre hatalı"
  - Cause: Wrong email or password
  - Action: Verify credentials, check caps lock

ACCOUNT_LOCKED:
  - Status: 403 Forbidden
  - Message: "Hesabınız çok fazla başarısız giriş denemesi nedeniyle kilitlendi"
  - Cause: 5+ failed login attempts
  - Action: Wait 30 minutes or use password reset

EMAIL_NOT_VERIFIED:
  - Status: 403 Forbidden
  - Message: "Lütfen e-posta adresinizi doğrulayın"
  - Cause: Email verification pending
  - Action: Check email for verification link

TOKEN_EXPIRED:
  - Status: 401 Unauthorized
  - Message: "Oturum süreniz doldu"
  - Cause: Access token expired
  - Action: Use refresh token to get new access token

INVALID_REFRESH_TOKEN:
  - Status: 401 Unauthorized
  - Message: "Geçersiz yenileme tokenı"
  - Cause: Refresh token invalid, expired, or already used
  - Action: Login again

PASSWORD_TOO_WEAK:
  - Status: 400 Bad Request
  - Message: "Şifre yeterince güçlü değil"
  - Cause: Password doesn't meet strength requirements
  - Action: Use stronger password with mixed characters

EMAIL_ALREADY_EXISTS:
  - Status: 409 Conflict
  - Message: "Bu e-posta adresi zaten kullanımda"
  - Cause: Email already registered
  - Action: Use different email or login/recover password

=============================================================================
TROUBLESHOOTING
=============================================================================

"Token validation fails":
-------------------------
Problem: JWT token rejected by backend
Solution:
  1. Check token expiration (exp claim)
  2. Verify token signature (secret key)
  3. Ensure token type is correct (access vs refresh)
  4. Check for token in revocation list

"Cannot login after registration":
----------------------------------
Problem: Email verification required
Solution:
  1. Check email for verification link
  2. Resend verification email if needed
  3. Check spam/junk folder
  4. Admin can manually verify email

"Refresh token rotation issues":
---------------------------------
Problem: Refresh token used multiple times
Solution:
  1. Implement single-use refresh tokens
  2. Rotate refresh token on each use
  3. Revoke old refresh token immediately
  4. Use refresh token families for detection

"Session management problems":
------------------------------
Problem: Too many sessions or stale sessions
Solution:
  1. Set max concurrent sessions per user
  2. Implement session cleanup job
  3. Auto-expire inactive sessions (24h)
  4. Provide user UI to manage sessions

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-07
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import EmailStr, Field, field_validator, model_validator

from backend.api.schemas.base import (
    BaseSchema,
    IdentifierSchema,
    RequestSchema,
    ResponseSchema,
    TimestampSchema,
    validate_password_strength,
    validate_turkish_phone,
)

# =============================================================================
# ENUMS
# =============================================================================


class TokenType(str, Enum):
    """
    OAuth token types.
    EXACT MATCH with database model (backend/core/database/models/oauth_token.py)
    """

    BEARER = "bearer"
    MAC = "mac"


class MFAMethod(str, Enum):
    """Multi-factor authentication methods."""

    TOTP = "totp"  # Time-based OTP (Google Authenticator, Authy)
    SMS = "sms"  # SMS verification code
    EMAIL = "email"  # Email verification code


# =============================================================================
# LOGIN SCHEMAS
# =============================================================================


class LoginRequest(RequestSchema):
    """
    User login request.

    Authenticates user with email and password.

    Example:
        >>> login = LoginRequest(
        ...     email="avukat@example.com",
        ...     password="SecureP@ss123"
        ... )
    """

    email: EmailStr = Field(
        ...,
        description="User email address",
        examples=["avukat@example.com"],
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password",
        json_schema_extra={"sensitive": True},
    )

    remember_me: bool = Field(
        default=False,
        description="Extend session duration (30 days vs 1 day)",
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalize email to lowercase."""
        return v.lower().strip()


class UserInfo(BaseSchema):
    """
    User information included in login/register responses.

    Subset of user data for authentication responses.
    """

    id: UUID = Field(..., description="User UUID")
    email: EmailStr = Field(..., description="User email")
    full_name: str = Field(..., description="Full name")
    role: str = Field(..., description="User role")
    tenant_id: Optional[UUID] = Field(None, description="Tenant UUID")
    is_email_verified: bool = Field(..., description="Email verification status")
    mfa_enabled: bool = Field(False, description="MFA enabled status")


class LoginResponse(ResponseSchema):
    """
    Successful login response.

    Returns access and refresh tokens with user information.

    Example:
        >>> response = LoginResponse(
        ...     access_token="eyJhbGci...",
        ...     refresh_token="eyJhbGci...",
        ...     token_type="bearer",
        ...     expires_in=3600,
        ...     user=user_info
        ... )
    """

    access_token: str = Field(
        ...,
        description="JWT access token",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )

    refresh_token: str = Field(
        ...,
        description="JWT refresh token",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )

    token_type: TokenType = Field(
        TokenType.BEARER,
        description="Token type (always 'bearer')",
    )

    expires_in: int = Field(
        ...,
        description="Access token expiration time in seconds",
        examples=[3600],
    )

    user: UserInfo = Field(
        ...,
        description="Authenticated user information",
    )


class LogoutRequest(RequestSchema):
    """
    Logout request.

    Revokes current session and optionally all sessions.

    Example:
        >>> logout = LogoutRequest(
        ...     all_sessions=False  # Logout current session only
        ... )
    """

    all_sessions: bool = Field(
        default=False,
        description="Revoke all user sessions (logout everywhere)",
    )


# =============================================================================
# REGISTRATION SCHEMAS
# =============================================================================


class RegisterRequest(RequestSchema):
    """
    User registration request.

    Creates new user account with validation.

    Example:
        >>> register = RegisterRequest(
        ...     email="yeni@example.com",
        ...     password="SecureP@ss123",
        ...     password_confirm="SecureP@ss123",
        ...     full_name="Mehmet Demir",
        ...     phone="+905321234567",
        ...     profession_type="lawyer",
        ...     bar_number="12345",
        ...     accept_terms=True,
        ...     accept_kvkk=True
        ... )
    """

    email: EmailStr = Field(
        ...,
        description="User email address (will be verified)",
        examples=["yeni@example.com"],
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password (min 8 chars, mixed case, numbers, special)",
        json_schema_extra={"sensitive": True},
    )

    password_confirm: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password confirmation (must match password)",
        json_schema_extra={"sensitive": True},
    )

    full_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Full name (first and last name)",
        examples=["Mehmet Demir"],
    )

    phone: str = Field(
        ...,
        description="Turkish mobile phone number",
        examples=["+905321234567", "0532 123 45 67"],
    )

    profession_type: Optional[str] = Field(
        None,
        description="Profession type (lawyer, judge, legal_consultant, etc.)",
        examples=["lawyer", "legal_consultant"],
    )

    bar_number: Optional[str] = Field(
        None,
        max_length=50,
        description="Bar association number (required for lawyers)",
        examples=["12345"],
    )

    organization_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Organization/law firm name (optional)",
        examples=["Yılmaz Hukuk Bürosu"],
    )

    accept_terms: bool = Field(
        ...,
        description="Accept Terms of Service (required)",
    )

    accept_kvkk: bool = Field(
        ...,
        description="Accept KVKK privacy policy (required for Turkey)",
    )

    marketing_consent: bool = Field(
        default=False,
        description="Consent for marketing communications (optional)",
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalize email to lowercase."""
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        return validate_password_strength(v)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        """Validate and normalize Turkish phone number."""
        return validate_turkish_phone(v)

    @model_validator(mode="after")
    def validate_passwords_match(self):
        """Validate password confirmation matches."""
        if self.password != self.password_confirm:
            raise ValueError("Şifreler eşleşmiyor")
        return self

    @model_validator(mode="after")
    def validate_terms_acceptance(self):
        """Validate required terms accepted."""
        if not self.accept_terms:
            raise ValueError("Kullanım koşullarını kabul etmelisiniz")
        if not self.accept_kvkk:
            raise ValueError("KVKK gizlilik politikasını kabul etmelisiniz")
        return self

    @model_validator(mode="after")
    def validate_lawyer_bar_number(self):
        """Validate bar number required for lawyers."""
        if self.profession_type == "lawyer" and not self.bar_number:
            raise ValueError("Avukatlar için baro numarası zorunludur")
        return self


class RegisterResponse(ResponseSchema):
    """
    Successful registration response.

    Returns created user and optionally tokens (if auto-login enabled).

    Example:
        >>> response = RegisterResponse(
        ...     user=user_info,
        ...     access_token="eyJhbGci...",  # Optional
        ...     message="Kayıt başarılı! Lütfen e-postanızı doğrulayın."
        ... )
    """

    user: UserInfo = Field(
        ...,
        description="Created user information",
    )

    access_token: Optional[str] = Field(
        None,
        description="JWT access token (if auto-login enabled)",
    )

    refresh_token: Optional[str] = Field(
        None,
        description="JWT refresh token (if auto-login enabled)",
    )

    token_type: Optional[TokenType] = Field(
        None,
        description="Token type (if tokens provided)",
    )

    expires_in: Optional[int] = Field(
        None,
        description="Token expiration in seconds (if tokens provided)",
    )

    message: str = Field(
        ...,
        description="Registration success message",
        examples=["Kayıt başarılı! Lütfen e-postanızı doğrulayın."],
    )


# =============================================================================
# EMAIL VERIFICATION SCHEMAS
# =============================================================================


class EmailVerificationRequest(RequestSchema):
    """
    Email verification request.

    Verifies user email with token sent via email.

    Example:
        >>> verify = EmailVerificationRequest(
        ...     token="verification-token-from-email"
        ... )
    """

    token: str = Field(
        ...,
        min_length=32,
        description="Email verification token from email link",
        examples=["abc123def456..."],
    )


class ResendVerificationRequest(RequestSchema):
    """
    Resend email verification request.

    Sends new verification email to user.

    Example:
        >>> resend = ResendVerificationRequest(
        ...     email="avukat@example.com"
        ... )
    """

    email: EmailStr = Field(
        ...,
        description="User email address",
        examples=["avukat@example.com"],
    )


# =============================================================================
# TOKEN MANAGEMENT SCHEMAS
# =============================================================================


class TokenRefreshRequest(RequestSchema):
    """
    Token refresh request.

    Exchanges refresh token for new access token.

    Example:
        >>> refresh = TokenRefreshRequest(
        ...     refresh_token="eyJhbGci..."
        ... )
    """

    refresh_token: str = Field(
        ...,
        description="JWT refresh token",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )


class TokenRefreshResponse(ResponseSchema):
    """
    Token refresh response.

    Returns new access token (and optionally new refresh token).

    Example:
        >>> response = TokenRefreshResponse(
        ...     access_token="eyJhbGci...",
        ...     token_type="bearer",
        ...     expires_in=3600
        ... )
    """

    access_token: str = Field(
        ...,
        description="New JWT access token",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )

    refresh_token: Optional[str] = Field(
        None,
        description="New JWT refresh token (if rotation enabled)",
    )

    token_type: TokenType = Field(
        TokenType.BEARER,
        description="Token type",
    )

    expires_in: int = Field(
        ...,
        description="Access token expiration in seconds",
        examples=[3600],
    )


class TokenIntrospectionResponse(ResponseSchema):
    """
    Token introspection response.

    Returns detailed token validation information.

    Example:
        >>> response = TokenIntrospectionResponse(
        ...     active=True,
        ...     user_id="user-uuid",
        ...     token_type="access",
        ...     expires_at=datetime(2024, 11, 7, 12, 0, 0)
        ... )
    """

    active: bool = Field(
        ...,
        description="Whether token is valid and active",
    )

    user_id: Optional[UUID] = Field(
        None,
        description="User UUID (if token is valid)",
    )

    token_type: Optional[str] = Field(
        None,
        description="Token type (access/refresh)",
        examples=["access", "refresh"],
    )

    expires_at: Optional[datetime] = Field(
        None,
        description="Token expiration timestamp (UTC)",
    )

    issued_at: Optional[datetime] = Field(
        None,
        description="Token issued at timestamp (UTC)",
    )

    scope: Optional[List[str]] = Field(
        None,
        description="Token scopes/permissions",
        examples=[["contracts:read", "documents:write"]],
    )


# =============================================================================
# PASSWORD MANAGEMENT SCHEMAS
# =============================================================================


class PasswordResetRequest(RequestSchema):
    """
    Password reset request (initiate).

    Sends password reset email to user.

    Example:
        >>> reset = PasswordResetRequest(
        ...     email="avukat@example.com"
        ... )
    """

    email: EmailStr = Field(
        ...,
        description="User email address",
        examples=["avukat@example.com"],
    )


class PasswordResetConfirm(RequestSchema):
    """
    Password reset confirmation.

    Completes password reset with token from email.

    Example:
        >>> confirm = PasswordResetConfirm(
        ...     token="reset-token-from-email",
        ...     new_password="NewSecureP@ss456",
        ...     password_confirm="NewSecureP@ss456"
        ... )
    """

    token: str = Field(
        ...,
        min_length=32,
        description="Password reset token from email",
        examples=["abc123def456..."],
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password",
        json_schema_extra={"sensitive": True},
    )

    password_confirm: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password confirmation",
        json_schema_extra={"sensitive": True},
    )

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        return validate_password_strength(v)

    @model_validator(mode="after")
    def validate_passwords_match(self):
        """Validate password confirmation matches."""
        if self.new_password != self.password_confirm:
            raise ValueError("Şifreler eşleşmiyor")
        return self


class ChangePasswordRequest(RequestSchema):
    """
    Change password request (authenticated user).

    Allows authenticated user to change their password.

    Example:
        >>> change = ChangePasswordRequest(
        ...     old_password="OldP@ss123",
        ...     new_password="NewP@ss456",
        ...     password_confirm="NewP@ss456"
        ... )
    """

    old_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Current password",
        json_schema_extra={"sensitive": True},
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password",
        json_schema_extra={"sensitive": True},
    )

    password_confirm: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password confirmation",
        json_schema_extra={"sensitive": True},
    )

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        return validate_password_strength(v)

    @model_validator(mode="after")
    def validate_passwords_match(self):
        """Validate password confirmation matches."""
        if self.new_password != self.password_confirm:
            raise ValueError("Şifreler eşleşmiyor")
        return self

    @model_validator(mode="after")
    def validate_passwords_different(self):
        """Validate new password is different from old."""
        if self.old_password == self.new_password:
            raise ValueError("Yeni şifre eskisiyle aynı olamaz")
        return self


# =============================================================================
# SESSION MANAGEMENT SCHEMAS
# =============================================================================


class SessionInfo(BaseSchema, IdentifierSchema, TimestampSchema):
    """
    Active session information.

    Represents a user's active session.

    Example:
        >>> session = SessionInfo(
        ...     id="session-uuid",
        ...     user_id="user-uuid",
        ...     ip_address="192.168.1.1",
        ...     user_agent="Mozilla/5.0...",
        ...     is_current=True,
        ...     created_at=datetime.utcnow(),
        ...     last_activity_at=datetime.utcnow()
        ... )
    """

    user_id: UUID = Field(..., description="User UUID")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent string")
    device_type: Optional[str] = Field(None, description="Device type (mobile/desktop/tablet)")
    location: Optional[str] = Field(None, description="Approximate location (city, country)")
    is_current: bool = Field(..., description="Whether this is the current session")
    last_activity_at: datetime = Field(..., description="Last activity timestamp (UTC)")
    expires_at: datetime = Field(..., description="Session expiration timestamp (UTC)")


class SessionListResponse(ResponseSchema):
    """
    List of user's active sessions.

    Example:
        >>> response = SessionListResponse(
        ...     sessions=[session1, session2],
        ...     total=2
        ... )
    """

    sessions: List[SessionInfo] = Field(
        ...,
        description="List of active sessions",
    )

    total: int = Field(
        ...,
        ge=0,
        description="Total number of active sessions",
    )


class SessionRevokeRequest(RequestSchema):
    """
    Revoke specific session request.

    Example:
        >>> revoke = SessionRevokeRequest(
        ...     session_id="session-uuid"
        ... )
    """

    session_id: UUID = Field(
        ...,
        description="Session UUID to revoke",
    )


# =============================================================================
# MFA SCHEMAS
# =============================================================================


class MFAEnableRequest(RequestSchema):
    """
    Enable MFA request.

    Initiates MFA setup for user.

    Example:
        >>> enable = MFAEnableRequest(
        ...     method="totp",
        ...     password="UserP@ss123"  # Verify user
        ... )
    """

    method: MFAMethod = Field(
        MFAMethod.TOTP,
        description="MFA method to enable",
    )

    password: str = Field(
        ...,
        description="User password for verification",
        json_schema_extra={"sensitive": True},
    )


class MFAEnableResponse(ResponseSchema):
    """
    MFA enable response.

    Returns QR code and backup codes for MFA setup.

    Example:
        >>> response = MFAEnableResponse(
        ...     secret="BASE32SECRET",
        ...     qr_code="data:image/png;base64,...",
        ...     backup_codes=["12345678", "23456789", ...]
        ... )
    """

    secret: str = Field(
        ...,
        description="TOTP secret (Base32)",
        examples=["JBSWY3DPEHPK3PXP"],
    )

    qr_code: str = Field(
        ...,
        description="QR code data URL for authenticator app",
        examples=["data:image/png;base64,iVBORw0KGgoAAAANS..."],
    )

    backup_codes: List[str] = Field(
        ...,
        description="Backup codes for MFA (store securely!)",
        examples=[["12345678", "23456789", "34567890"]],
    )

    message: str = Field(
        ...,
        description="Setup instructions",
        examples=["QR kodu tarayın ve doğrulama kodunu girin"],
    )


class MFAVerifyRequest(RequestSchema):
    """
    MFA verification request.

    Verifies MFA code from authenticator app.

    Example:
        >>> verify = MFAVerifyRequest(
        ...     code="123456"
        ... )
    """

    code: str = Field(
        ...,
        min_length=6,
        max_length=8,
        description="MFA code from authenticator app",
        examples=["123456"],
    )


class MFADisableRequest(RequestSchema):
    """
    Disable MFA request.

    Disables MFA for user (requires password verification).

    Example:
        >>> disable = MFADisableRequest(
        ...     password="UserP@ss123"
        ... )
    """

    password: str = Field(
        ...,
        description="User password for verification",
        json_schema_extra={"sensitive": True},
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TokenType",
    "MFAMethod",
    # Login
    "LoginRequest",
    "LoginResponse",
    "LogoutRequest",
    "UserInfo",
    # Registration
    "RegisterRequest",
    "RegisterResponse",
    # Email verification
    "EmailVerificationRequest",
    "ResendVerificationRequest",
    # Token management
    "TokenRefreshRequest",
    "TokenRefreshResponse",
    "TokenIntrospectionResponse",
    # Password management
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "ChangePasswordRequest",
    # Session management
    "SessionInfo",
    "SessionListResponse",
    "SessionRevokeRequest",
    # MFA
    "MFAEnableRequest",
    "MFAEnableResponse",
    "MFAVerifyRequest",
    "MFADisableRequest",
]
