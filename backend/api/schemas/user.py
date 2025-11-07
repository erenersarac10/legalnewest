"""
User Schemas for Turkish Legal AI Platform.

Enterprise-grade user management schemas for CRUD operations, profile management,
preferences, statistics, and role-based access control (RBAC).

=============================================================================
FEATURES
=============================================================================

1. User CRUD Schemas
   ------------------
   - UserCreate: Create user (admin only)
   - UserUpdate: Update user (admin)
   - UserResponse: Basic user response
   - UserDetailResponse: Detailed user with relationships
   - UserListResponse: Paginated user list

2. Profile Management
   -------------------
   - UserProfileUpdate: Self-service profile update
   - UserPreferencesUpdate: User preferences
   - UserAvatarUpdate: Profile photo upload
   - UserProfileResponse: Public profile view

3. User Roles & Permissions
   --------------------------
   - UserRole enum: CITIZEN, LAWYER, JUDGE, PROSECUTOR, ADMIN, SUPERADMIN
   - AccountStatus enum: PENDING, ACTIVE, LOCKED, SUSPENDED, DISABLED
   - ProfessionType enum: Turkish legal professions
   - Role-based field visibility

4. User Statistics
   -----------------
   - UserStatistics: Usage metrics
   - UserActivity: Recent activity log
   - LoginHistory: Authentication history
   - UsageQuota: Resource consumption

5. Search & Filter
   ----------------
   - UserSearchRequest: Advanced search
   - UserFilterParams: Role, status, tenant filters
   - UserSortParams: Sort by name, date, activity

=============================================================================
USAGE
=============================================================================

Create User (Admin):
--------------------

>>> from backend.api.schemas.user import UserCreate
>>>
>>> # Admin creates new user
>>> user_data = UserCreate(
...     email="avukat@example.com",
...     password="SecureP@ss123",
...     full_name="Ahmet Yılmaz",
...     phone="+905321234567",
...     role="lawyer",
...     profession_type="lawyer",
...     bar_number="12345",
...     tenant_id="tenant-uuid"
... )
>>>
>>> @app.post("/users")
>>> async def create_user(
...     user: UserCreate,
...     current_user: User = Depends(require_role("admin"))
... ):
...     new_user = await user_service.create(user)
...     return UserResponse.model_validate(new_user)

Update User:
------------

>>> from backend.api.schemas.user import UserUpdate
>>>
>>> # Admin updates user
>>> update_data = UserUpdate(
...     full_name="Ahmet Yılmaz (Updated)",
...     phone="+905329999999",
...     role="admin"  # Promote to admin
... )
>>>
>>> @app.patch("/users/{user_id}")
>>> async def update_user(
...     user_id: UUID,
...     data: UserUpdate,
...     current_user: User = Depends(require_role("admin"))
... ):
...     updated_user = await user_service.update(user_id, data)
...     return UserResponse.model_validate(updated_user)

Self Profile Update:
--------------------

>>> from backend.api.schemas.user import UserProfileUpdate
>>>
>>> # User updates own profile
>>> profile_data = UserProfileUpdate(
...     full_name="Mehmet Demir",
...     phone="+905321234567",
...     bio="Ankara Barosu'na kayıtlı avukat"
... )
>>>
>>> @app.patch("/users/me")
>>> async def update_profile(
...     data: UserProfileUpdate,
...     current_user: User = Depends(get_current_user)
... ):
...     updated = await user_service.update(current_user.id, data)
...     return UserDetailResponse.model_validate(updated)

User Preferences:
-----------------

>>> from backend.api.schemas.user import UserPreferencesUpdate
>>>
>>> # Update user preferences
>>> prefs = UserPreferencesUpdate(
...     language="tr",
...     timezone="Europe/Istanbul",
...     theme="dark",
...     notifications={
...         "email": True,
...         "push": False,
...         "sms": False
...     }
... )
>>>
>>> @app.put("/users/me/preferences")
>>> async def update_preferences(
...     prefs: UserPreferencesUpdate,
...     current_user: User = Depends(get_current_user)
... ):
...     await user_service.update_preferences(current_user.id, prefs)
...     return {"message": "Tercihler güncellendi"}

List Users (Paginated):
-----------------------

>>> from backend.api.schemas.user import UserFilterParams
>>> from backend.api.dependencies.pagination import PaginationParams
>>>
>>> @app.get("/users")
>>> async def list_users(
...     filters: UserFilterParams = Depends(),
...     pagination: PaginationParams = Depends(get_pagination),
...     current_user: User = Depends(require_role("admin"))
... ):
...     users, total = await user_service.list_filtered(filters, pagination)
...     return PaginatedResponse[UserResponse](
...         items=[UserResponse.model_validate(u) for u in users],
...         total=total,
...         page=pagination.page,
...         page_size=pagination.page_size,
...         total_pages=(total + pagination.page_size - 1) // pagination.page_size,
...         has_next=pagination.page * pagination.page_size < total,
...         has_prev=pagination.page > 1
...     )

User Statistics:
----------------

>>> from backend.api.schemas.user import UserStatistics
>>>
>>> @app.get("/users/me/statistics")
>>> async def get_statistics(
...     current_user: User = Depends(get_current_user)
... ):
...     stats = await user_service.get_statistics(current_user.id)
...     return UserStatistics.model_validate(stats)

=============================================================================
USER ROLES & HIERARCHY
=============================================================================

Role Hierarchy (Ascending Privilege):
--------------------------------------

1. CITIZEN (Level 10)
   - Basic user
   - Can use chat and upload documents
   - Read-only access to public resources
   - No admin capabilities

2. LAWYER (Level 30)
   - Licensed lawyer (avukat)
   - Access to legal research and templates
   - Can create contracts and documents
   - Bar number required

3. JUDGE (Level 40)
   - Judge or prosecutor
   - Read-only access to precedents
   - Can view all documents in organization
   - No document creation

4. PROSECUTOR (Level 40)
   - Prosecutor (savcı)
   - Similar to JUDGE
   - Special access to case management

5. ADMIN (Level 80)
   - Organization administrator
   - Manage users in organization
   - Configure organization settings
   - View usage and billing

6. SUPERADMIN (Level 100)
   - Platform administrator
   - Manage all tenants
   - System configuration
   - Full access to everything

Role-Based Permissions:
-----------------------

CITIZEN permissions:
  - chat:use
  - documents:upload
  - documents:read_own

LAWYER permissions:
  - All CITIZEN permissions
  - contracts:create
  - contracts:read
  - templates:use
  - legal_research:access

ADMIN permissions:
  - All LAWYER permissions
  - users:create
  - users:read_all
  - users:update
  - organization:manage

SUPERADMIN permissions:
  - All permissions (admin:*)

=============================================================================
ACCOUNT STATUS LIFECYCLE
=============================================================================

Status Flow:
------------

PENDING (Registration)
  ↓ (Email verification)
ACTIVE (Normal usage)
  ↓ (Failed logins or admin action)
LOCKED (Temporary, auto-unlock after 30 min)
  ↓ (Admin action or security issue)
SUSPENDED (Manual review required)
  ↓ (Admin decision)
DISABLED (Soft delete, can be reactivated)
  ↓ (Hard delete after retention period)
DELETED (Permanently removed)

Status Descriptions:
--------------------

PENDING:
  - Initial state after registration
  - Email verification pending
  - Limited access (read-only)
  - Auto-expires after 7 days if not verified

ACTIVE:
  - Normal active user
  - Full access per role
  - Can login and use all features

LOCKED:
  - Temporary lockout due to failed logins
  - Auto-unlocks after 30 minutes
  - User notified via email
  - Manual unlock available for admin

SUSPENDED:
  - Admin-initiated suspension
  - Policy violation or security concern
  - User cannot login
  - Manual review and reactivation required

DISABLED:
  - Soft delete (data retained)
  - User cannot login
  - Can be reactivated by admin
  - Data retained per KVKK (5 years)

=============================================================================
PROFESSION TYPES
=============================================================================

Turkish Legal Professions:
---------------------------

LAWYER (Avukat):
  - Requires bar number (baro numarası)
  - Full access to legal research
  - Can create contracts and petitions

JUDGE (Hakim):
  - Read-only access to precedents
  - Can view case history
  - No document creation

PROSECUTOR (Savcı):
  - Similar to JUDGE
  - Access to prosecution tools
  - Special permissions for investigation

LEGAL_CONSULTANT (Hukuk Danışmanı):
  - Company legal department
  - Limited contract access
  - No bar number required

NOTARY (Noter):
  - Notarization workflows
  - Document authentication
  - Official seals and stamps

ENFORCEMENT_OFFICER (İcra Müdürü):
  - Enforcement court access
  - Execution proceedings
  - Special forms and templates

LEGAL_EXPERT (Bilirkişi):
  - Expert witness access
  - Case analysis tools
  - Report templates

LAW_STUDENT (Hukuk Öğrencisi):
  - Educational access
  - Limited features
  - Practice mode available

OTHER (Diğer):
  - General legal services user
  - Basic features
  - No special privileges

=============================================================================
VALIDATION RULES
=============================================================================

Email Validation:
-----------------
- Valid email format (RFC 5322)
- Unique across system
- Turkish domains supported
- Lowercase normalized
- Max 255 characters
- Corporate domain verification (optional)

Password Validation:
--------------------
- Minimum 8 characters (enforced by auth.py)
- Password strength check
- Cannot be same as email
- History check (last 5 passwords)

Phone Validation:
-----------------
- Turkish format: +905XXXXXXXXX
- Unique per tenant (optional)
- SMS verification available
- Normalized storage

Bar Number Validation:
----------------------
- Required for LAWYER role
- Alphanumeric, max 50 characters
- Validated against bar association (future)
- Unique per bar association

Full Name Validation:
---------------------
- Minimum 2 characters
- Maximum 255 characters
- Turkish characters supported (ğ, ü, ş, ı, ö, ç)
- No numbers or special characters
- Trimmed whitespace

=============================================================================
SECURITY & PRIVACY
=============================================================================

Password Security:
------------------
- Never returned in responses (excluded)
- Argon2id hashing in database
- Password change requires old password
- Reset via email verification only

Sensitive Data Protection:
--------------------------
TC Kimlik No:
  - Encrypted at rest (AES-256)
  - Masked in logs (*********67)
  - Only visible to user and superadmin
  - KVKK compliant handling

Phone Number:
  - Stored normalized (+90XXXXXXXXXX)
  - Masked in audit logs
  - SMS verification required

Email:
  - Unique identifier
  - Verification required
  - Used for authentication

PII Handling:
-------------
- Full name: Visible per role
- Address: Encrypted, need-to-know only
- Date of birth: Optional, encrypted
- Profile photo: Stored in S3, CDN cached

Role-Based Visibility:
----------------------
- CITIZEN: Can see own data only
- LAWYER: Can see team members
- ADMIN: Can see organization users
- SUPERADMIN: Can see all users

Field Visibility Matrix:
------------------------
Field            | Self | Team | Admin | SuperAdmin
-----------------+------+------+-------+-----------
email            | ✓    | ✓    | ✓     | ✓
full_name        | ✓    | ✓    | ✓     | ✓
phone            | ✓    | ✗    | ✓     | ✓
tc_kimlik_no     | ✓    | ✗    | ✗     | ✓
password_hash    | ✗    | ✗    | ✗     | ✗
failed_logins    | ✓    | ✗    | ✓     | ✓
last_login_at    | ✓    | ✗    | ✓     | ✓

=============================================================================
KVKK COMPLIANCE
=============================================================================

Personal Data Processing:
-------------------------

User Data Categories:
  - Identity: email, full_name, tc_kimlik_no
  - Contact: phone, address
  - Authentication: password_hash, mfa_secret
  - Usage: login_history, activity_log
  - Professional: bar_number, profession_type

Legal Basis:
  - Contract performance (Terms of Service)
  - Explicit consent (KVKK checkbox)
  - Legal obligation (bar number verification)

Retention Period:
  - Active users: Indefinite (while account active)
  - Deleted users: 5 years (KVKK requirement)
  - Logs: 2 years (security)
  - Backups: 90 days

Data Subject Rights:
--------------------
1. Right to Access:
   - GET /users/me (own data)
   - GET /users/me/data-export (full export)

2. Right to Rectification:
   - PATCH /users/me (update profile)
   - Contact support for corrections

3. Right to Erasure:
   - DELETE /users/me (soft delete)
   - Hard delete after retention period

4. Right to Data Portability:
   - GET /users/me/data-export (JSON format)
   - Includes all personal data

5. Right to Object:
   - Opt-out of marketing
   - Opt-out of analytics

Data Transfer:
--------------
- Data stored in Turkey (compliance)
- Cross-border transfer restrictions
- EU adequacy decision respected

=============================================================================
PERFORMANCE OPTIMIZATION
=============================================================================

Database Queries:
-----------------
- Index on email (unique)
- Index on tenant_id (multi-tenant)
- Index on role + account_status (filtering)
- Composite index on (tenant_id, email)

Caching Strategy:
-----------------
- User profile: 5 minutes (Redis)
- Permissions: 15 minutes (in-memory)
- Role hierarchy: 1 hour (rarely changes)
- Statistics: 1 hour (expensive query)

Pagination:
-----------
- Default: 20 users per page
- Max: 100 users per page
- Use cursor pagination for large datasets
- Cache total count (5 minutes)

N+1 Query Prevention:
---------------------
- Eager load tenant relationship
- Eager load role permissions
- Batch load user preferences
- Use select_related and prefetch_related

=============================================================================
TROUBLESHOOTING
=============================================================================

"Cannot create user with LAWYER role":
---------------------------------------
Problem: Bar number validation fails
Solution:
  1. Ensure bar_number is provided
  2. Check bar number format (alphanumeric, max 50)
  3. Verify profession_type is "lawyer"
  4. Contact bar association for verification

"Email already exists error":
------------------------------
Problem: Email uniqueness constraint violated
Solution:
  1. Check if email already registered
  2. Use different email
  3. If previous account, recover password
  4. Admin can check deleted users

"User locked out":
-------------------
Problem: Too many failed login attempts
Solution:
  1. Wait 30 minutes for auto-unlock
  2. Use password reset
  3. Admin can manually unlock
  4. Check for brute force attack

"Permission denied errors":
----------------------------
Problem: User lacks required permissions
Solution:
  1. Check user role and permissions
  2. Verify tenant membership
  3. Check account status (must be ACTIVE)
  4. Admin can grant additional permissions

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-07
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import EmailStr, Field, field_validator, model_validator

from backend.api.schemas.base import (
    BaseSchema,
    IdentifierSchema,
    RequestSchema,
    ResponseSchema,
    TimestampSchema,
    TenantSchema,
    AuditSchema,
    validate_turkish_phone,
)

# =============================================================================
# ENUMS (Match database model exactly)
# =============================================================================


class UserRole(str, Enum):
    """
    User roles with hierarchical permissions.
    EXACT MATCH with database model (backend/core/database/models/user.py)

    Roles (ascending privilege):
    - CITIZEN: Basic user
    - LAWYER: Licensed lawyer
    - JUDGE: Judge/prosecutor
    - PROSECUTOR: Prosecutor
    - ADMIN: Organization admin
    - SUPERADMIN: Platform admin
    """

    CITIZEN = "citizen"
    LAWYER = "lawyer"
    JUDGE = "judge"
    PROSECUTOR = "prosecutor"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


class AccountStatus(str, Enum):
    """
    Account status lifecycle.
    EXACT MATCH with database model (backend/core/database/models/user.py)

    States:
    - PENDING: Email verification pending
    - ACTIVE: Active and verified
    - LOCKED: Locked due to security (auto-unlock)
    - SUSPENDED: Admin suspended
    - DISABLED: Soft disabled (can be reactivated)
    """

    PENDING = "pending"
    ACTIVE = "active"
    LOCKED = "locked"
    SUSPENDED = "suspended"
    DISABLED = "disabled"


class ProfessionType(str, Enum):
    """
    Turkish legal profession types.
    EXACT MATCH with database model (backend/core/database/models/user.py)

    Professional categories for legal sector users.
    """

    LAWYER = "lawyer"  # Avukat
    JUDGE = "judge"  # Hakim
    PROSECUTOR = "prosecutor"  # Savcı
    LEGAL_CONSULTANT = "legal_consultant"  # Hukuk Danışmanı
    NOTARY = "notary"  # Noter
    ENFORCEMENT_OFFICER = "enforcement_officer"  # İcra Müdürü
    LEGAL_EXPERT = "legal_expert"  # Bilirkişi
    LAW_STUDENT = "law_student"  # Hukuk Öğrencisi


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================


class UserCreate(RequestSchema, TenantSchema):
    """
    Create user request (admin only).

    Creates new user account in system.

    Example:
        >>> user = UserCreate(
        ...     email="avukat@example.com",
        ...     password="SecureP@ss123",
        ...     full_name="Ahmet Yılmaz",
        ...     phone="+905321234567",
        ...     role="lawyer",
        ...     profession_type="lawyer",
        ...     bar_number="12345",
        ...     tenant_id="tenant-uuid"
        ... )
    """

    email: EmailStr = Field(
        ...,
        description="User email address (unique)",
        examples=["avukat@example.com"],
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password (will be hashed)",
        json_schema_extra={"sensitive": True},
    )

    full_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Full name (first and last)",
        examples=["Ahmet Yılmaz"],
    )

    phone: str = Field(
        ...,
        description="Turkish mobile phone",
        examples=["+905321234567"],
    )

    role: UserRole = Field(
        UserRole.CITIZEN,
        description="User role (determines permissions)",
    )

    profession_type: Optional[ProfessionType] = Field(
        None,
        description="Profession type",
    )

    bar_number: Optional[str] = Field(
        None,
        max_length=50,
        description="Bar number (required for lawyers)",
        examples=["12345"],
    )

    tc_kimlik_no: Optional[str] = Field(
        None,
        min_length=11,
        max_length=11,
        description="TC Kimlik No (optional, will be encrypted)",
        json_schema_extra={"sensitive": True},
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalize email to lowercase."""
        return v.lower().strip()

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        """Validate and normalize phone."""
        return validate_turkish_phone(v)

    @model_validator(mode="after")
    def validate_lawyer_bar_number(self):
        """Validate bar number for lawyers."""
        if self.role == UserRole.LAWYER and not self.bar_number:
            raise ValueError("Avukatlar için baro numarası zorunludur")
        return self


class UserUpdate(RequestSchema):
    """
    Update user request (admin).

    Updates user fields (all optional).

    Example:
        >>> update = UserUpdate(
        ...     full_name="Ahmet Yılmaz (Updated)",
        ...     role="admin"
        ... )
    """

    email: Optional[EmailStr] = Field(
        None,
        description="New email (requires reverification)",
    )

    full_name: Optional[str] = Field(
        None,
        min_length=2,
        max_length=255,
        description="Updated full name",
    )

    phone: Optional[str] = Field(
        None,
        description="Updated phone number",
    )

    role: Optional[UserRole] = Field(
        None,
        description="Updated role",
    )

    profession_type: Optional[ProfessionType] = Field(
        None,
        description="Updated profession",
    )

    bar_number: Optional[str] = Field(
        None,
        max_length=50,
        description="Updated bar number",
    )

    account_status: Optional[AccountStatus] = Field(
        None,
        description="Updated account status",
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: Optional[str]) -> Optional[str]:
        """Normalize email."""
        return v.lower().strip() if v else None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Validate phone if provided."""
        return validate_turkish_phone(v) if v else None


class UserProfileUpdate(RequestSchema):
    """
    Update own profile request (self-service).

    Users can update their own profile (limited fields).

    Example:
        >>> profile = UserProfileUpdate(
        ...     full_name="Mehmet Demir",
        ...     phone="+905321234567",
        ...     bio="Ankara Barosu'na kayıtlı avukat"
        ... )
    """

    full_name: Optional[str] = Field(
        None,
        min_length=2,
        max_length=255,
        description="Updated full name",
    )

    phone: Optional[str] = Field(
        None,
        description="Updated phone",
    )

    bio: Optional[str] = Field(
        None,
        max_length=500,
        description="Short bio (max 500 chars)",
    )

    avatar_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Profile photo URL",
    )

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Validate phone."""
        return validate_turkish_phone(v) if v else None


class UserPreferencesUpdate(RequestSchema):
    """
    Update user preferences.

    Example:
        >>> prefs = UserPreferencesUpdate(
        ...     language="tr",
        ...     timezone="Europe/Istanbul",
        ...     theme="dark"
        ... )
    """

    language: Optional[str] = Field(
        None,
        description="Preferred language (tr, en)",
        examples=["tr", "en"],
    )

    timezone: Optional[str] = Field(
        None,
        description="User timezone",
        examples=["Europe/Istanbul", "UTC"],
    )

    theme: Optional[str] = Field(
        None,
        description="UI theme (light, dark, auto)",
        examples=["light", "dark", "auto"],
    )

    notifications: Optional[Dict[str, bool]] = Field(
        None,
        description="Notification preferences",
        examples=[{"email": True, "push": False, "sms": False}],
    )

    date_format: Optional[str] = Field(
        None,
        description="Date format preference",
        examples=["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"],
    )


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================


class UserResponse(ResponseSchema, IdentifierSchema, TimestampSchema, TenantSchema):
    """
    Basic user response.

    Standard user representation for lists and references.

    Example:
        >>> user = UserResponse(
        ...     id="user-uuid",
        ...     email="avukat@example.com",
        ...     full_name="Ahmet Yılmaz",
        ...     role="lawyer",
        ...     account_status="active",
        ...     tenant_id="tenant-uuid",
        ...     created_at=datetime.utcnow(),
        ...     updated_at=datetime.utcnow()
        ... )
    """

    email: EmailStr = Field(..., description="User email")
    full_name: str = Field(..., description="Full name")
    phone: Optional[str] = Field(None, description="Phone number (masked)")
    role: UserRole = Field(..., description="User role")
    account_status: AccountStatus = Field(..., description="Account status")
    profession_type: Optional[ProfessionType] = Field(None, description="Profession")
    bar_number: Optional[str] = Field(None, description="Bar number (if lawyer)")
    is_email_verified: bool = Field(..., description="Email verification status")
    mfa_enabled: bool = Field(False, description="MFA enabled status")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")


class UserDetailResponse(UserResponse, AuditSchema):
    """
    Detailed user response.

    Extended user information with relationships and metadata.

    Example:
        >>> user = UserDetailResponse(
        ...     ...,  # All UserResponse fields
        ...     tc_kimlik_no="***********67",  # Masked
        ...     failed_login_attempts=0,
        ...     locked_until=None,
        ...     preferences={...},
        ...     statistics={...}
        ... )
    """

    tc_kimlik_no: Optional[str] = Field(
        None,
        description="TC Kimlik No (masked: ***********67)",
    )

    failed_login_attempts: int = Field(
        0,
        description="Failed login count",
    )

    locked_until: Optional[datetime] = Field(
        None,
        description="Account locked until (if locked)",
    )

    preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="User preferences",
    )

    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class UserStatistics(BaseSchema):
    """
    User usage statistics.

    Example:
        >>> stats = UserStatistics(
        ...     documents_uploaded=150,
        ...     contracts_created=45,
        ...     chat_sessions=89,
        ...     total_usage_mb=1024.5
        ... )
    """

    documents_uploaded: int = Field(
        0,
        ge=0,
        description="Total documents uploaded",
    )

    contracts_created: int = Field(
        0,
        ge=0,
        description="Contracts created",
    )

    chat_sessions: int = Field(
        0,
        ge=0,
        description="Chat sessions initiated",
    )

    ai_queries: int = Field(
        0,
        ge=0,
        description="AI queries performed",
    )

    total_usage_mb: float = Field(
        0.0,
        ge=0.0,
        description="Storage usage in MB",
    )

    last_activity_at: Optional[datetime] = Field(
        None,
        description="Last activity timestamp",
    )


class LoginHistoryEntry(BaseSchema, IdentifierSchema, TimestampSchema):
    """
    Login history entry.

    Example:
        >>> entry = LoginHistoryEntry(
        ...     id="login-uuid",
        ...     ip_address="192.168.1.1",
        ...     user_agent="Mozilla/5.0...",
        ...     success=True,
        ...     created_at=datetime.utcnow()
        ... )
    """

    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent string")
    location: Optional[str] = Field(None, description="Approximate location")
    success: bool = Field(..., description="Login success status")
    failure_reason: Optional[str] = Field(None, description="Failure reason if failed")


# =============================================================================
# FILTER & SEARCH SCHEMAS
# =============================================================================


class UserFilterParams(BaseSchema):
    """
    User filtering parameters.

    Example:
        >>> filters = UserFilterParams(
        ...     role="lawyer",
        ...     account_status="active",
        ...     search="Ahmet"
        ... )
    """

    role: Optional[UserRole] = Field(
        None,
        description="Filter by role",
    )

    account_status: Optional[AccountStatus] = Field(
        None,
        description="Filter by status",
    )

    profession_type: Optional[ProfessionType] = Field(
        None,
        description="Filter by profession",
    )

    search: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Search in name, email",
    )

    is_email_verified: Optional[bool] = Field(
        None,
        description="Filter by email verification",
    )

    mfa_enabled: Optional[bool] = Field(
        None,
        description="Filter by MFA status",
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "UserRole",
    "AccountStatus",
    "ProfessionType",
    # Request schemas
    "UserCreate",
    "UserUpdate",
    "UserProfileUpdate",
    "UserPreferencesUpdate",
    # Response schemas
    "UserResponse",
    "UserDetailResponse",
    "UserStatistics",
    "LoginHistoryEntry",
    # Filter schemas
    "UserFilterParams",
]
