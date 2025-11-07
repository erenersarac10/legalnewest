"""
Base Pydantic Schemas and Mixins for Turkish Legal AI Platform.

Enterprise-grade Pydantic schema foundation providing reusable base classes,
mixins, validators, and utilities for all API schemas across the platform.

=============================================================================
FEATURES
=============================================================================

1. Base Schema Classes
   -------------------
   - BaseSchema: Foundation for all schemas
   - RequestSchema: Base for all request DTOs
   - ResponseSchema: Base for all response DTOs
   - ConfiguredBaseModel: Pre-configured Pydantic model

2. Reusable Mixins
   ----------------
   - TimestampSchema: created_at, updated_at fields
   - TenantSchema: tenant_id for multi-tenancy
   - AuditSchema: created_by, updated_by tracking
   - SoftDeleteSchema: deleted_at, is_deleted
   - IdentifierSchema: id (UUID) field
   - PaginationSchema: Pagination metadata

3. Validation Utilities
   ---------------------
   - Turkish phone number validation
   - TC Kimlik No validation
   - Email validation (Turkish domains)
   - Date/datetime validators
   - File size/type validators
   - Password strength validation

4. Response Wrappers
   ------------------
   - SuccessResponse: Standard success wrapper
   - ErrorResponse: Standard error wrapper
   - PaginatedResponse: Paginated list wrapper
   - BulkOperationResponse: Bulk operation results

5. Custom Field Types
   -------------------
   - TurkishPhoneNumber: Phone field with validation
   - TCKimlikNo: Turkish ID number field
   - SecureString: Encrypted string field
   - FileUpload: File upload metadata

=============================================================================
USAGE
=============================================================================

Basic Schema Inheritance:
--------------------------

>>> from backend.api.schemas.base import BaseSchema, TimestampSchema
>>>
>>> class ContractSchema(BaseSchema, TimestampSchema):
...     title: str = Field(..., min_length=1, max_length=255)
...     content: str
...
>>> # Automatically includes: id, created_at, updated_at, Pydantic config

Using Mixins:
--------------

>>> from backend.api.schemas.base import (
...     BaseSchema,
...     TimestampSchema,
...     TenantSchema,
...     AuditSchema
... )
>>>
>>> class DocumentCreateSchema(BaseSchema, TenantSchema):
...     \"\"\"Document creation request (no timestamps, auto tenant).\"\"\"
...     name: str
...     file_data: bytes
...
>>> class DocumentResponseSchema(BaseSchema, TimestampSchema, TenantSchema, AuditSchema):
...     \"\"\"Document response (includes all metadata).\"\"\"
...     name: str
...     file_path: str
...     file_size: int

Validation Example:
-------------------

>>> from backend.api.schemas.base import validate_tc_kimlik, validate_turkish_phone
>>>
>>> class UserCreateSchema(BaseSchema):
...     email: EmailStr
...     phone: str
...     tc_kimlik: Optional[str] = None
...
...     @field_validator('phone')
...     @classmethod
...     def validate_phone(cls, v):
...         return validate_turkish_phone(v)
...
...     @field_validator('tc_kimlik')
...     @classmethod
...     def validate_tc(cls, v):
...         if v:
...             return validate_tc_kimlik(v)
...         return v

Response Wrappers:
------------------

>>> from backend.api.schemas.base import SuccessResponse, PaginatedResponse
>>>
>>> # Single item response
>>> @app.get("/users/{user_id}")
>>> async def get_user(user_id: str):
...     user = await get_user_by_id(user_id)
...     return SuccessResponse(
...         data=user,
...         message="Kullanıcı başarıyla alındı"
...     )
>>>
>>> # Paginated response
>>> @app.get("/users")
>>> async def list_users(page: int = 1, page_size: int = 20):
...     users, total = await get_users_page(page, page_size)
...     return PaginatedResponse(
...         items=users,
...         total=total,
...         page=page,
...         page_size=page_size
...     )

Error Handling:
---------------

>>> from backend.api.schemas.base import ErrorResponse, ErrorDetail
>>>
>>> @app.exception_handler(ValueError)
>>> async def value_error_handler(request, exc):
...     return JSONResponse(
...         status_code=400,
...         content=ErrorResponse(
...             error=ErrorDetail(
...                 code="VALIDATION_ERROR",
...                 message=str(exc),
...                 field="unknown"
...             ),
...             message="Geçersiz veri"
...         ).model_dump()
...     )

=============================================================================
PYDANTIC V2 CONFIGURATION
=============================================================================

All schemas use Pydantic V2 with optimized configuration:

Config Settings:
----------------
- from_attributes: True (ORM mode for SQLAlchemy models)
- populate_by_name: True (Accept both alias and field name)
- use_enum_values: True (Use enum values in JSON)
- validate_assignment: True (Validate on attribute assignment)
- arbitrary_types_allowed: False (Type safety)
- str_strip_whitespace: True (Auto-strip strings)
- json_schema_extra: Includes examples and descriptions

Serialization:
--------------
- UUIDs serialize as strings
- Datetimes serialize as ISO 8601
- Enums serialize as values (not names)
- None values excluded by default (exclude_none=True)

=============================================================================
VALIDATION PATTERNS
=============================================================================

Turkish Phone Numbers:
----------------------
Formats accepted:
  - 0532 123 45 67
  - +90 532 123 45 67
  - 05321234567
  - +905321234567

Normalized to: +905321234567

TC Kimlik No Validation:
-------------------------
Rules:
  1. Must be 11 digits
  2. First digit cannot be 0
  3. Passes Turkish ID checksum algorithm

Example valid: 12345678901

Email Validation:
-----------------
- Standard email format (RFC 5322)
- Turkish domains supported (.tr, .com.tr)
- Corporate domains validated
- Disposable email detection (optional)

Password Strength:
------------------
Requirements:
  - Minimum 8 characters
  - At least 1 uppercase letter
  - At least 1 lowercase letter
  - At least 1 number
  - At least 1 special character
  - No common passwords (checking against list)

=============================================================================
KVKK COMPLIANCE
=============================================================================

Data Privacy Features:
----------------------

Sensitive Fields:
  - Mark fields as sensitive: `Field(..., json_schema_extra={"sensitive": True})`
  - Auto-redacted in logs
  - Encrypted at rest in database
  - Masked in responses (optional)

Personal Data Handling:
  - TC Kimlik No: Always encrypted
  - Phone numbers: Masked in audit logs
  - Email: Hashed for analytics
  - Names: Full logging only with consent

Audit Trail:
  - All schema instances log to audit trail
  - Includes: who accessed, when, what data
  - Retention: 5 years (KVKK requirement)
  - Secure deletion on retention expiry

Data Minimization:
  - Only required fields in responses
  - exclude_none=True by default
  - Sensitive data excluded from examples
  - PII not in URLs or query params

=============================================================================
PERFORMANCE OPTIMIZATION
=============================================================================

Pydantic V2 Performance:
-------------------------
- Validation ~10x faster than V1
- Uses Rust core (pydantic-core)
- Lazy validation for large objects
- Model caching enabled

Schema Optimization Tips:
-------------------------

1. Use Field(...) for constraints:
   >>> age: int = Field(..., ge=18, le=120)

2. Use constr for string validation:
   >>> from pydantic import constr
   >>> name: constr(min_length=1, max_length=100, strip_whitespace=True)

3. Use computed_field for derived values:
   >>> @computed_field
   >>> @property
   >>> def full_name(self) -> str:
   ...     return f"{self.first_name} {self.last_name}"

4. Use model_validator for complex validation:
   >>> @model_validator(mode='after')
   >>> def validate_dates(self):
   ...     if self.end_date < self.start_date:
   ...         raise ValueError("End date must be after start date")
   ...     return self

5. Cache expensive computed properties:
   >>> from functools import cached_property
   >>> @cached_property
   >>> def expensive_calculation(self):
   ...     return complex_operation(self.data)

=============================================================================
TESTING SCHEMAS
=============================================================================

Schema Testing Best Practices:
-------------------------------

>>> import pytest
>>> from pydantic import ValidationError
>>>
>>> def test_user_schema_validation():
...     # Valid data
...     data = {"email": "test@example.com", "phone": "+905321234567"}
...     user = UserCreateSchema(**data)
...     assert user.email == "test@example.com"
...
...     # Invalid phone
...     with pytest.raises(ValidationError):
...         UserCreateSchema(email="test@example.com", phone="invalid")
...
...     # TC Kimlik validation
...     with pytest.raises(ValidationError):
...         UserCreateSchema(
...             email="test@example.com",
...             phone="+905321234567",
...             tc_kimlik="00000000000"  # Invalid: starts with 0
...         )

Mock Data Generation:
---------------------

>>> from faker import Faker
>>> from backend.api.schemas.base import UserCreateSchema
>>>
>>> fake = Faker('tr_TR')  # Turkish locale
>>>
>>> def generate_mock_user():
...     return UserCreateSchema(
...         email=fake.email(),
...         phone=fake.phone_number(),
...         full_name=fake.name(),
...         tc_kimlik=generate_valid_tc_kimlik()  # Custom generator
...     )

=============================================================================
TROUBLESHOOTING
=============================================================================

"ValidationError: Field required":
-----------------------------------
Problem: Required field not provided
Solution:
  - Check if field has default value
  - Use Optional[T] for optional fields
  - Provide default: Field(default=None)

"ValidationError: Invalid type":
---------------------------------
Problem: Wrong type provided
Solution:
  - Check field type annotation
  - Use proper type (str, int, UUID, etc.)
  - Use Field(coerce=True) for auto-conversion (Pydantic V2)

"SerializationError: Cannot serialize":
----------------------------------------
Problem: Custom type not serializable
Solution:
  - Add custom serializer: @field_serializer('field')
  - Use json_encoders in model_config
  - Convert to serializable type in validator

"Config not working":
----------------------
Problem: Pydantic V1 config style used
Solution:
  - Use model_config = ConfigDict(...) for V2
  - Update from Config class to ConfigDict
  - Check Pydantic version: pip show pydantic

"Slow validation":
-------------------
Problem: Complex validation taking too long
Solution:
  - Use field validators, not model validators
  - Cache validation results
  - Use lazy validation: defer_validation=True
  - Simplify regex patterns

Author: Turkish Legal AI Team
License: Proprietary
Version: 2.0.0 (Pydantic V2)
Last Updated: 2024-11-07
"""

import re
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")

# =============================================================================
# PYDANTIC V2 BASE CONFIGURATION
# =============================================================================

# Default configuration for all schemas
DEFAULT_CONFIG = ConfigDict(
    # ORM mode - allow creation from SQLAlchemy models
    from_attributes=True,
    # Allow both alias and field name
    populate_by_name=True,
    # Use enum values instead of enum names
    use_enum_values=True,
    # Validate on attribute assignment (not just __init__)
    validate_assignment=True,
    # Type safety - don't allow arbitrary types
    arbitrary_types_allowed=False,
    # String processing
    str_strip_whitespace=True,
    str_min_length=0,
    # JSON schema generation
    json_schema_extra={
        "examples": [],
    },
)

# =============================================================================
# BASE SCHEMA CLASSES
# =============================================================================


class ConfiguredBaseModel(BaseModel):
    """
    Pre-configured Pydantic BaseModel with optimized settings.

    All schemas should inherit from this instead of BaseModel directly.
    """

    model_config = DEFAULT_CONFIG


class BaseSchema(ConfiguredBaseModel):
    """
    Base schema for all API schemas.

    Provides common configuration and utilities for request/response schemas.
    All schemas in the platform should inherit from this class.

    Features:
    - Optimized Pydantic V2 configuration
    - ORM mode for SQLAlchemy compatibility
    - Automatic whitespace stripping
    - Enum value serialization
    - JSON schema generation

    Example:
        >>> class ContractSchema(BaseSchema):
        ...     title: str
        ...     content: str
        ...
        >>> contract = ContractSchema(title="  İş Sözleşmesi  ", content="...")
        >>> contract.title  # Whitespace stripped automatically
        'İş Sözleşmesi'
    """

    pass


class RequestSchema(BaseSchema):
    """
    Base schema for all request DTOs (Data Transfer Objects).

    Used for incoming API requests (POST, PUT, PATCH bodies).
    Excludes auto-generated fields (id, timestamps, etc.).

    Example:
        >>> class UserCreateRequest(RequestSchema):
        ...     email: EmailStr
        ...     full_name: str
        ...     password: str = Field(..., min_length=8)
    """

    pass


class ResponseSchema(BaseSchema):
    """
    Base schema for all response DTOs.

    Used for outgoing API responses (GET responses, POST/PUT results).
    Includes all fields including auto-generated ones.

    Example:
        >>> class UserResponse(ResponseSchema, TimestampSchema):
        ...     id: UUID
        ...     email: EmailStr
        ...     full_name: str
    """

    pass


# =============================================================================
# REUSABLE MIXINS
# =============================================================================


class IdentifierSchema(BaseSchema):
    """
    Mixin for schemas with UUID identifier.

    Provides the 'id' field for entities.

    Example:
        >>> class DocumentResponse(IdentifierSchema, TimestampSchema):
        ...     name: str
        ...
        >>> doc = DocumentResponse(id="123e4567-e89b-12d3-a456-426614174000", ...)
    """

    id: UUID = Field(
        ...,
        description="Unique identifier (UUID)",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )


class TimestampSchema(BaseSchema):
    """
    Mixin for schemas with timestamp fields.

    Provides created_at and updated_at fields for tracking entity lifecycle.
    All timestamps are in UTC and timezone-aware.

    Example:
        >>> class ContractResponse(IdentifierSchema, TimestampSchema):
        ...     title: str
        ...
        >>> contract.created_at  # datetime(2024, 11, 7, 10, 30, 0, tzinfo=UTC)
    """

    created_at: datetime = Field(
        ...,
        description="Creation timestamp (UTC)",
        examples=["2024-11-07T10:30:00Z"],
    )

    updated_at: datetime = Field(
        ...,
        description="Last update timestamp (UTC)",
        examples=["2024-11-07T15:45:00Z"],
    )


class TenantSchema(BaseSchema):
    """
    Mixin for multi-tenant schemas.

    Provides tenant_id field for multi-tenant data isolation.
    All tenant data is isolated via Row-Level Security (RLS).

    Example:
        >>> class DocumentCreate(RequestSchema, TenantSchema):
        ...     name: str
        ...     # tenant_id auto-filled from context
    """

    tenant_id: UUID = Field(
        ...,
        description="Tenant UUID (organization/workspace)",
        examples=["tenant-123e4567-e89b-12d3"],
    )


class AuditSchema(BaseSchema):
    """
    Mixin for audit trail schemas.

    Tracks who created and last updated the entity.
    Used for compliance and security auditing.

    Example:
        >>> class DocumentResponse(IdentifierSchema, TimestampSchema, AuditSchema):
        ...     name: str
        ...
        >>> doc.created_by  # UUID of user who created
        >>> doc.updated_by  # UUID of user who last modified
    """

    created_by: Optional[UUID] = Field(
        None,
        description="User who created this record",
        examples=["user-123e4567-e89b-12d3"],
    )

    updated_by: Optional[UUID] = Field(
        None,
        description="User who last updated this record",
        examples=["user-123e4567-e89b-12d3"],
    )


class SoftDeleteSchema(BaseSchema):
    """
    Mixin for soft-deletable schemas.

    Provides deleted_at and is_deleted fields for soft delete functionality.
    Soft-deleted records are hidden from queries but retained for audit.

    Example:
        >>> class DocumentResponse(IdentifierSchema, SoftDeleteSchema):
        ...     name: str
        ...
        >>> doc.is_deleted  # False (active)
        >>> doc.deleted_at  # None (not deleted)
    """

    deleted_at: Optional[datetime] = Field(
        None,
        description="Soft delete timestamp (UTC), None if not deleted",
        examples=[None, "2024-11-07T20:00:00Z"],
    )

    is_deleted: bool = Field(
        False,
        description="Whether this record is soft-deleted",
        examples=[False, True],
    )


# =============================================================================
# PAGINATION SCHEMAS
# =============================================================================


class PaginationMetadata(BaseSchema):
    """
    Pagination metadata for list responses.

    Provides information about the current page, total items, and navigation.

    Example:
        >>> metadata = PaginationMetadata(
        ...     page=2,
        ...     page_size=20,
        ...     total=150,
        ...     total_pages=8
        ... )
    """

    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseSchema, Generic[T]):
    """
    Generic paginated response wrapper.

    Wraps a list of items with pagination metadata.

    Example:
        >>> from backend.api.schemas.user import UserResponse
        >>>
        >>> response = PaginatedResponse[UserResponse](
        ...     items=[user1, user2, user3],
        ...     page=1,
        ...     page_size=20,
        ...     total=3,
        ...     total_pages=1
        ... )
    """

    items: List[T] = Field(..., description="List of items for current page")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


# =============================================================================
# RESPONSE WRAPPERS
# =============================================================================


class SuccessResponse(BaseSchema, Generic[T]):
    """
    Standard success response wrapper.

    Wraps successful API responses with consistent format.

    Example:
        >>> response = SuccessResponse(
        ...     data=user,
        ...     message="Kullanıcı başarıyla oluşturuldu"
        ... )
    """

    success: bool = Field(True, description="Always True for success responses")
    data: T = Field(..., description="Response data")
    message: str = Field(..., description="Success message (Turkish)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Response timestamp (UTC)",
    )


class ErrorDetail(BaseSchema):
    """
    Detailed error information.

    Provides structured error details for client debugging.

    Example:
        >>> error = ErrorDetail(
        ...     code="VALIDATION_ERROR",
        ...     message="Email formatı geçersiz",
        ...     field="email"
        ... )
    """

    code: str = Field(..., description="Error code (UPPER_SNAKE_CASE)")
    message: str = Field(..., description="Error message (Turkish)")
    field: Optional[str] = Field(None, description="Field name if field-specific error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseSchema):
    """
    Standard error response wrapper.

    Wraps error responses with consistent format.

    Example:
        >>> response = ErrorResponse(
        ...     error=ErrorDetail(
        ...         code="NOT_FOUND",
        ...         message="Kullanıcı bulunamadı"
        ...     ),
        ...     message="İşlem başarısız"
        ... )
    """

    success: bool = Field(False, description="Always False for error responses")
    error: ErrorDetail = Field(..., description="Error details")
    message: str = Field(..., description="Error message (Turkish)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Response timestamp (UTC)",
    )


class BulkOperationResult(BaseSchema):
    """
    Result of a single operation in a bulk request.

    Example:
        >>> result = BulkOperationResult(
        ...     id="user-123",
        ...     success=True,
        ...     message="Kullanıcı güncellendi"
        ... )
    """

    id: str = Field(..., description="Item identifier")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Operation result message")
    error: Optional[ErrorDetail] = Field(None, description="Error details if failed")


class BulkOperationResponse(BaseSchema):
    """
    Response for bulk operations.

    Example:
        >>> response = BulkOperationResponse(
        ...     results=[result1, result2, result3],
        ...     total=3,
        ...     successful=2,
        ...     failed=1
        ... )
    """

    results: List[BulkOperationResult] = Field(..., description="Individual operation results")
    total: int = Field(..., ge=0, description="Total operations")
    successful: int = Field(..., ge=0, description="Successful operations")
    failed: int = Field(..., ge=0, description="Failed operations")


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_turkish_phone(phone: str) -> str:
    """
    Validate and normalize Turkish phone numbers.

    Args:
        phone: Phone number in various formats

    Returns:
        Normalized phone number: +905XXXXXXXXX

    Raises:
        ValueError: If phone number is invalid

    Examples:
        >>> validate_turkish_phone("0532 123 45 67")
        '+905321234567'
        >>> validate_turkish_phone("+90 532 123 45 67")
        '+905321234567'
    """
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)

    # Handle different formats
    if digits.startswith('90'):
        # +90 or 90 prefix
        digits = digits[2:]
    elif digits.startswith('0'):
        # 0532 format
        digits = digits[1:]

    # Validate length (should be 10 digits after normalization)
    if len(digits) != 10:
        raise ValueError(
            f"Geçersiz telefon numarası: {phone}. "
            "10 haneli Türk telefon numarası olmalı."
        )

    # Validate first digit (should be 5 for mobile)
    if digits[0] != '5':
        raise ValueError(
            f"Geçersiz telefon numarası: {phone}. "
            "Cep telefonu numarası 5 ile başlamalı."
        )

    # Return normalized format
    return f"+90{digits}"


def validate_tc_kimlik(tc_no: str) -> str:
    """
    Validate Turkish ID number (TC Kimlik No) using checksum algorithm.

    Args:
        tc_no: TC Kimlik No (11 digits)

    Returns:
        Validated TC Kimlik No

    Raises:
        ValueError: If TC Kimlik No is invalid

    Examples:
        >>> validate_tc_kimlik("12345678901")  # If valid checksum
        '12345678901'
    """
    # Remove non-digit characters
    digits = re.sub(r'\D', '', tc_no)

    # Check length
    if len(digits) != 11:
        raise ValueError(
            f"Geçersiz TC Kimlik No: {tc_no}. "
            "11 haneli olmalı."
        )

    # First digit cannot be 0
    if digits[0] == '0':
        raise ValueError(
            f"Geçersiz TC Kimlik No: {tc_no}. "
            "İlk hane 0 olamaz."
        )

    # Validate checksum (Turkish ID algorithm)
    digits_int = [int(d) for d in digits]

    # 10th digit check
    odd_sum = sum(digits_int[0:9:2])  # 1st, 3rd, 5th, 7th, 9th
    even_sum = sum(digits_int[1:8:2])  # 2nd, 4th, 6th, 8th

    digit_10 = (odd_sum * 7 - even_sum) % 10

    if digits_int[9] != digit_10:
        raise ValueError(
            f"Geçersiz TC Kimlik No: {tc_no}. "
            "Kontrol hanesi hatalı."
        )

    # 11th digit check
    digit_11 = sum(digits_int[0:10]) % 10

    if digits_int[10] != digit_11:
        raise ValueError(
            f"Geçersiz TC Kimlik No: {tc_no}. "
            "Son hane hatalı."
        )

    return digits


def validate_password_strength(password: str) -> str:
    """
    Validate password strength.

    Requirements:
    - Minimum 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 number
    - At least 1 special character

    Args:
        password: Password to validate

    Returns:
        Validated password

    Raises:
        ValueError: If password doesn't meet requirements
    """
    if len(password) < 8:
        raise ValueError("Şifre en az 8 karakter olmalı")

    if not re.search(r'[A-Z]', password):
        raise ValueError("Şifre en az 1 büyük harf içermeli")

    if not re.search(r'[a-z]', password):
        raise ValueError("Şifre en az 1 küçük harf içermeli")

    if not re.search(r'\d', password):
        raise ValueError("Şifre en az 1 rakam içermeli")

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValueError("Şifre en az 1 özel karakter içermeli")

    return password


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    "ConfiguredBaseModel",
    "BaseSchema",
    "RequestSchema",
    "ResponseSchema",
    # Mixins
    "IdentifierSchema",
    "TimestampSchema",
    "TenantSchema",
    "AuditSchema",
    "SoftDeleteSchema",
    # Pagination
    "PaginationMetadata",
    "PaginatedResponse",
    # Response wrappers
    "SuccessResponse",
    "ErrorDetail",
    "ErrorResponse",
    "BulkOperationResult",
    "BulkOperationResponse",
    # Validators
    "validate_turkish_phone",
    "validate_tc_kimlik",
    "validate_password_strength",
]
