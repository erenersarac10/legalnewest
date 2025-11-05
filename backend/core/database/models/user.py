"""
User model for Turkish Legal AI authentication and authorization.

This module provides the core User model with:
- Email/password authentication (Argon2 hashing)
- Role-based access control (RBAC)
- Multi-tenant support
- Turkish-specific fields (TC No, bar number)
- KVKK compliance (encrypted PII)
- Session management
- API key management
- Account security (lockout, MFA)

Security Features:
- Password hashing with Argon2id
- TC Kimlik No encryption at rest
- Failed login tracking
- Account lockout mechanism
- Email verification required
- Audit trail for all changes

Example:
    >>> user = User(
    ...     email="avukat@example.com",
    ...     full_name="Ahmet Yılmaz",
    ...     role=UserRole.LAWYER,
    ...     bar_number="12345",
    ...     tenant_id=tenant_id
    ... )
    >>> user.set_password("SecureP@ss123")
    >>> user.verify_password("SecureP@ss123")
    True
"""

import enum
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Enum,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates

from backend.core.config.security import security_config
from backend.core.constants import (
    MAX_EMAIL_LENGTH,
    MAX_FAILED_LOGIN_ATTEMPTS,
    MAX_NAME_LENGTH,
    MAX_PHONE_LENGTH,
    MIN_PASSWORD_LENGTH,
    TC_KIMLIK_LENGTH,
)
from backend.core.exceptions import (
    AccountLockedError,
    InvalidCredentialsError,
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


class UserRole(str, enum.Enum):
    """
    User roles with hierarchical permissions.
    
    Roles (ascending privilege):
    - CITIZEN: Basic user, can use chat and document upload
    - LAWYER: Licensed lawyer, can access legal research and templates
    - JUDGE: Judge/prosecutor, read-only access to precedents
    - ADMIN: Organization admin, manage users and settings
    - SUPERADMIN: Platform admin, manage all tenants
    """
    
    CITIZEN = "citizen"
    LAWYER = "lawyer"
    JUDGE = "judge"
    PROSECUTOR = "prosecutor"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name(self) -> str:
        """Human-readable role name in Turkish."""
        names = {
            self.CITIZEN: "Vatandaş",
            self.LAWYER: "Avukat",
            self.JUDGE: "Hakim",
            self.PROSECUTOR: "Savcı",
            self.ADMIN: "Yönetici",
            self.SUPERADMIN: "Sistem Yöneticisi",
        }
        return names.get(self, self.value)
    
    @property
    def permissions(self) -> list[str]:
        """Get permissions for this role."""
        role_permissions = {
            self.CITIZEN: [
                "chat:use",
                "document:upload",
                "document:view_own",
            ],
            self.LAWYER: [
                "chat:use",
                "document:upload",
                "document:view_own",
                "document:analyze",
                "legal_research:use",
                "contract:generate",
                "template:use",
            ],
            self.JUDGE: [
                "chat:use",
                "legal_research:use",
                "precedent:view",
            ],
            self.PROSECUTOR: [
                "chat:use",
                "legal_research:use",
                "precedent:view",
            ],
            self.ADMIN: [
                "chat:use",
                "document:*",
                "legal_research:use",
                "contract:generate",
                "template:*",
                "user:manage",
                "organization:manage",
                "analytics:view",
            ],
            self.SUPERADMIN: ["*"],  # All permissions
        }
        return role_permissions.get(self, [])


class AccountStatus(str, enum.Enum):
    """Account status for user lifecycle management."""
    
    PENDING = "pending"  # Email verification pending
    ACTIVE = "active"    # Active and verified
    LOCKED = "locked"    # Locked due to security (failed logins)
    SUSPENDED = "suspended"  # Admin suspended
    DISABLED = "disabled"    # Soft disabled (can be reactivated)
    
    def __str__(self) -> str:
        return self.value


class ProfessionType(str, enum.Enum):
    """Turkish legal profession types."""
    
    LAWYER = "lawyer"              # Avukat
    JUDGE = "judge"                # Hakim
    PROSECUTOR = "prosecutor"      # Savcı
    LEGAL_CONSULTANT = "legal_consultant"  # Hukuk Danışmanı
    NOTARY = "notary"              # Noter
    ENFORCEMENT_OFFICER = "enforcement_officer"  # İcra Müdürü
    LEGAL_EXPERT = "legal_expert"  # Bilirkişi
    LAW_STUDENT = "law_student"    # Hukuk Öğrencisi
    OTHER = "other"                # Diğer
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# USER MODEL
# =============================================================================


class User(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    User model with Turkish legal professional support.
    
    Attributes:
        email: Unique email address (login identifier)
        password_hash: Argon2 hashed password
        full_name: User's full name
        phone: Turkish phone number (+90XXXXXXXXXX)
        tc_no: Turkish ID number (encrypted, KVKK compliant)
        bar_number: Baro sicil numarası (for lawyers)
        profession: Professional type
        role: User role (RBAC)
        status: Account status
        is_email_verified: Email verification status
        email_verified_at: Email verification timestamp
        is_mfa_enabled: Multi-factor authentication enabled
        mfa_secret: TOTP secret (encrypted)
        failed_login_attempts: Failed login counter
        locked_until: Account lock expiration
        last_login_at: Last successful login
        last_login_ip: Last login IP address
        login_count: Total successful logins
        password_changed_at: Last password change
        preferences: User preferences (JSON)
        metadata: Additional metadata (JSON)
    
    Relationships:
        tenant: Organization/tenant
        created_by: User who created this account
        updated_by: User who last updated
        sessions: Active sessions
        api_keys: User's API keys
        documents: Uploaded documents
        chat_sessions: Chat history
    """
    
    __tablename__ = "users"
    
    # =========================================================================
    # AUTHENTICATION FIELDS
    # =========================================================================
    
    email = Column(
        String(MAX_EMAIL_LENGTH),
        unique=True,
        nullable=False,
        index=True,
        comment="Email address (login identifier)",
    )
    
    password_hash = Column(
        String(255),
        nullable=False,
        comment="Argon2 hashed password",
    )
    
    # =========================================================================
    # PROFILE FIELDS
    # =========================================================================
    
    full_name = Column(
        String(MAX_NAME_LENGTH),
        nullable=False,
        comment="User's full name",
    )
    
    phone = Column(
        String(MAX_PHONE_LENGTH),
        nullable=True,
        index=True,
        comment="Turkish phone number (+90XXXXXXXXXX)",
    )
    
    # Turkish-specific: TC Kimlik No (encrypted for KVKK compliance)
    tc_no_encrypted = Column(
        Text,
        nullable=True,
        comment="Encrypted Turkish ID number (KVKK compliant)",
    )
    
    # Turkish-specific: Baro numarası (for lawyers)
    bar_number = Column(
        String(20),
        nullable=True,
        index=True,
        comment="Turkish bar association number (avukat için)",
    )
    
    profession = Column(
        Enum(ProfessionType, native_enum=False, length=50),
        nullable=True,
        comment="Professional type",
    )
    
    # =========================================================================
    # AUTHORIZATION FIELDS
    # =========================================================================
    
    role = Column(
        Enum(UserRole, native_enum=False, length=50),
        nullable=False,
        default=UserRole.CITIZEN,
        index=True,
        comment="User role (RBAC)",
    )
    
    status = Column(
        Enum(AccountStatus, native_enum=False, length=50),
        nullable=False,
        default=AccountStatus.PENDING,
        index=True,
        comment="Account status",
    )
    
    # =========================================================================
    # VERIFICATION FIELDS
    # =========================================================================
    
    is_email_verified = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Email verification status",
    )
    
    email_verified_at = Column(
        "email_verified_at",
        nullable=True,
        comment="Email verification timestamp",
    )
    
    # =========================================================================
    # MFA FIELDS
    # =========================================================================
    
    is_mfa_enabled = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Multi-factor authentication enabled",
    )
    
    mfa_secret = Column(
        Text,
        nullable=True,
        comment="Encrypted TOTP secret for MFA",
    )
    
    # =========================================================================
    # SECURITY FIELDS
    # =========================================================================
    
    failed_login_attempts = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Failed login attempt counter",
    )
    
    locked_until = Column(
        "locked_until",
        nullable=True,
        comment="Account lock expiration (auto-unlock after this time)",
    )
    
    last_login_at = Column(
        "last_login_at",
        nullable=True,
        comment="Last successful login timestamp",
    )
    
    last_login_ip = Column(
        String(45),  # IPv6 max length
        nullable=True,
        comment="Last login IP address",
    )
    
    login_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total successful login count",
    )
    
    password_changed_at = Column(
        "password_changed_at",
        nullable=True,
        comment="Last password change timestamp",
    )
    
    # =========================================================================
    # JSON FIELDS
    # =========================================================================
    
    preferences = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="User preferences (language, theme, notifications)",
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (extensible)",
    )
    
    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================
    
    # Sessions relationship (one-to-many)
    sessions = relationship(
        "Session",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # API keys relationship (one-to-many)
    api_keys = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # Documents relationship (one-to-many)
    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # Chat sessions relationship (one-to-many)
    chat_sessions = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Unique email per tenant (allow same email across tenants)
        UniqueConstraint(
            "email",
            "tenant_id",
            name="uq_users_email_tenant",
        ),
        # Index for active users (common query)
        Index(
            "ix_users_active",
            "tenant_id",
            "status",
            postgresql_where="status = 'active' AND deleted_at IS NULL",
        ),
        # Index for email verification
        Index(
            "ix_users_unverified",
            "email",
            postgresql_where="is_email_verified = false",
        ),
        # Check: email format
        CheckConstraint(
            "email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$'",
            name="ck_users_email_format",
        ),
        # Check: phone format (Turkish)
        CheckConstraint(
            "phone IS NULL OR phone ~ '^\\+90[0-9]{10}$'",
            name="ck_users_phone_format",
        ),
    )
    
    # =========================================================================
    # PASSWORD MANAGEMENT
    # =========================================================================
    
    def set_password(self, plain_password: str) -> None:
        """
        Hash and set user password.
        
        Uses Argon2id for secure password hashing.
        
        Args:
            plain_password: Plain text password
            
        Raises:
            ValidationError: If password doesn't meet requirements
            
        Example:
            >>> user.set_password("MySecureP@ss123")
        """
        # Validate password strength
        security_config.validate_password_strength(plain_password)
        
        # Hash password
        self.password_hash = security_config.hash_password(plain_password)
        self.password_changed_at = datetime.now(timezone.utc)
        
        logger.info(
            "Password changed",
            user_id=str(self.id),
            email=self.email,
        )
    
    def verify_password(self, plain_password: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            plain_password: Plain text password to verify
            
        Returns:
            bool: True if password matches
            
        Example:
            >>> if user.verify_password("MySecureP@ss123"):
            ...     print("Login successful")
        """
        is_valid = security_config.verify_password(
            plain_password,
            self.password_hash,
        )
        
        if is_valid:
            # Check if password needs rehashing (algorithm updated)
            if security_config.needs_rehash(self.password_hash):
                logger.info(
                    "Password needs rehashing",
                    user_id=str(self.id),
                )
                # Rehash with new parameters
                self.password_hash = security_config.hash_password(plain_password)
        
        return is_valid
    
    def needs_password_change(self, max_age_days: int = 90) -> bool:
        """
        Check if password change is required.
        
        Args:
            max_age_days: Maximum password age in days
            
        Returns:
            bool: True if password should be changed
        """
        if not self.password_changed_at:
            return True  # Never changed
        
        age = datetime.now(timezone.utc) - self.password_changed_at
        return age.days > max_age_days
    
    # =========================================================================
    # TC KIMLIK NO (ENCRYPTED)
    # =========================================================================
    
    @property
    def tc_no(self) -> str | None:
        """
        Decrypt and return TC Kimlik No.
        
        Returns:
            str | None: Decrypted TC No or None
        """
        if not self.tc_no_encrypted:
            return None
        
        try:
            return security_config.decrypt(self.tc_no_encrypted)
        except Exception as e:
            logger.error(
                "TC No decryption failed",
                user_id=str(self.id),
                error=str(e),
            )
            return None
    
    @tc_no.setter
    def tc_no(self, value: str | None) -> None:
        """
        Encrypt and set TC Kimlik No.
        
        Args:
            value: TC Kimlik No (11 digits)
            
        Raises:
            ValidationError: If TC No is invalid
        """
        if not value:
            self.tc_no_encrypted = None
            return
        
        # Validate TC No
        if not self._validate_tc_no(value):
            raise ValidationError(
                message="Geçersiz TC Kimlik Numarası",
                field="tc_no",
            )
        
        # Encrypt for KVKK compliance
        self.tc_no_encrypted = security_config.encrypt(value)
        
        logger.info(
            "TC No encrypted and stored",
            user_id=str(self.id),
        )
    
    @staticmethod
    def _validate_tc_no(tc_no: str) -> bool:
        """
        Validate Turkish ID number using official algorithm.
        
        TC Kimlik No rules:
        - Exactly 11 digits
        - First digit cannot be 0
        - 10th digit = (sum(odd positions) * 7 - sum(even positions)) mod 10
        - 11th digit = sum(first 10 digits) mod 10
        
        Args:
            tc_no: TC Kimlik No to validate
            
        Returns:
            bool: True if valid
        """
        if not tc_no or len(tc_no) != TC_KIMLIK_LENGTH:
            return False
        
        if not tc_no.isdigit():
            return False
        
        if tc_no[0] == '0':
            return False
        
        # Convert to integers
        digits = [int(d) for d in tc_no]
        
        # Check 10th digit
        odd_sum = sum(digits[0:9:2])  # 1st, 3rd, 5th, 7th, 9th
        even_sum = sum(digits[1:8:2])  # 2nd, 4th, 6th, 8th
        check_10 = (odd_sum * 7 - even_sum) % 10
        
        if digits[9] != check_10:
            return False
        
        # Check 11th digit
        check_11 = sum(digits[0:10]) % 10
        
        return digits[10] == check_11
    
    # =========================================================================
    # ACCOUNT SECURITY
    # =========================================================================
    
    def record_failed_login(self) -> None:
        """
        Record a failed login attempt.
        
        Locks account after MAX_FAILED_LOGIN_ATTEMPTS.
        
        Example:
            >>> user.record_failed_login()
            >>> if user.is_locked:
            ...     raise AccountLockedError()
        """
        self.failed_login_attempts += 1
        
        if self.failed_login_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
            # Lock account for 30 minutes
            from datetime import timedelta
            self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            self.status = AccountStatus.LOCKED
            
            logger.warning(
                "Account locked due to failed login attempts",
                user_id=str(self.id),
                email=self.email,
                attempts=self.failed_login_attempts,
            )
    
    def record_successful_login(self, ip_address: str) -> None:
        """
        Record a successful login.
        
        Resets failed attempts and updates login metadata.
        
        Args:
            ip_address: Client IP address
        """
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_login_at = datetime.now(timezone.utc)
        self.last_login_ip = ip_address
        self.login_count += 1
        
        if self.status == AccountStatus.LOCKED:
            self.status = AccountStatus.ACTIVE
        
        logger.info(
            "Successful login",
            user_id=str(self.id),
            email=self.email,
            ip=ip_address,
            login_count=self.login_count,
        )
    
    @property
    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if self.status == AccountStatus.LOCKED:
            # Check if lock has expired
            if self.locked_until and datetime.now(timezone.utc) > self.locked_until:
                return False  # Auto-unlock
            return True
        return False
    
    def unlock_account(self) -> None:
        """Unlock account (admin action)."""
        self.status = AccountStatus.ACTIVE
        self.failed_login_attempts = 0
        self.locked_until = None
        
        logger.info(
            "Account unlocked",
            user_id=str(self.id),
            email=self.email,
        )
    
    # =========================================================================
    # AUTHORIZATION
    # =========================================================================
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            permission: Permission string (e.g., "document:upload")
            
        Returns:
            bool: True if user has permission
            
        Example:
            >>> if user.has_permission("contract:generate"):
            ...     generate_contract()
        """
        # Superadmin has all permissions
        if self.role == UserRole.SUPERADMIN:
            return True
        
        # Check role permissions
        permissions = self.role.permissions
        
        # Wildcard check
        if "*" in permissions:
            return True
        
        # Exact match
        if permission in permissions:
            return True
        
        # Wildcard prefix (e.g., "document:*" matches "document:upload")
        permission_prefix = permission.split(":")[0]
        if f"{permission_prefix}:*" in permissions:
            return True
        
        return False
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin or superadmin."""
        return self.role in (UserRole.ADMIN, UserRole.SUPERADMIN)
    
    @property
    def is_lawyer(self) -> bool:
        """Check if user is a lawyer."""
        return self.role == UserRole.LAWYER and self.bar_number is not None
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("email")
    def validate_email(self, key: str, email: str) -> str:
        """Validate email format."""
        if not email:
            raise ValidationError(message="Email gereklidir", field="email")
        
        # Basic email validation
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email.lower()):
            raise ValidationError(
                message="Geçersiz email formatı",
                field="email",
            )
        
        return email.lower()
    
    @validates("phone")
    def validate_phone(self, key: str, phone: str | None) -> str | None:
        """Validate Turkish phone format."""
        if not phone:
            return None
        
        # Turkish phone: +90XXXXXXXXXX (13 chars)
        phone_pattern = r"^\+90\d{10}$"
        if not re.match(phone_pattern, phone):
            raise ValidationError(
                message="Geçersiz telefon formatı. Örnek: +905551234567",
                field="phone",
            )
        
        return phone
    
    @validates("bar_number")
    def validate_bar_number(self, key: str, bar_number: str | None) -> str | None:
        """Validate bar number format."""
        if not bar_number:
            return None
        
        # Bar number: numeric, 3-20 digits
        if not bar_number.isdigit() or len(bar_number) < 3 or len(bar_number) > 20:
            raise ValidationError(
                message="Geçersiz baro sicil numarası",
                field="bar_number",
            )
        
        return bar_number
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"
    
    def to_dict(self, include_sensitive: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_sensitive: Include password_hash and encrypted fields
            
        Returns:
            dict: User data
        """
        data = super().to_dict()
        
        # Remove sensitive fields by default
        if not include_sensitive:
            data.pop("password_hash", None)
            data.pop("tc_no_encrypted", None)
            data.pop("mfa_secret", None)
        
        # Add computed fields
        data["is_locked"] = self.is_locked
        data["is_admin"] = self.is_admin
        data["is_lawyer"] = self.is_lawyer
        data["role_display"] = self.role.display_name
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "User",
    "UserRole",
    "AccountStatus",
    "ProfessionType",
]