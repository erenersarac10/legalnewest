"""
Base model utilities and mixins for SQLAlchemy ORM models.

This module provides reusable mixins and utilities for all database models:
- BaseModelMixin: Primary key, timestamps, and common fields
- TenantMixin: Multi-tenant isolation with RLS (Row Level Security)
- AuditMixin: Track who created/updated records
- SoftDeleteMixin: Soft delete functionality
- TimestampMixin: Automatic timestamp management
- VersionMixin: Optimistic locking with version tracking

All mixins are designed to work together and follow best practices:
- Thread-safe context management
- Timezone-aware timestamps (UTC)
- Audit trail compatibility
- KVKK/GDPR compliance ready

Example:
    >>> class User(Base, BaseModelMixin, TenantMixin, AuditMixin):
    ...     email = Column(String, unique=True)
    ...     
    >>> user = User(email="lawyer@example.com", tenant_id=tenant_id)
    >>> user.validate_tenant_access(current_tenant_id)
"""

import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Session, declarative_base, relationship

from backend.core.constants import (
    MAX_STRING_LENGTH,
    SOFT_DELETE_FIELD_NAME,
    TENANT_ID_FIELD_NAME,
)
from backend.core.exceptions import (
    TenantContextError,
    TenantIsolationViolationError,
)
from backend.core.logging import get_logger

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# THREAD-SAFE TENANT CONTEXT
# =============================================================================

# ContextVar is thread-safe and async-safe (unlike global variables)
_tenant_context: ContextVar[str | None] = ContextVar("tenant_id", default=None)
_user_context: ContextVar[str | None] = ContextVar("user_id", default=None)


def set_tenant_context(tenant_id: str) -> None:
    """
    Set current tenant context (thread-safe).
    
    This context is used for Row-Level Security (RLS) enforcement
    and automatic tenant_id injection in queries.
    
    Args:
        tenant_id: UUID of the current tenant
        
    Example:
        >>> set_tenant_context(str(user.tenant_id))
        >>> # All queries now filtered by this tenant
    """
    if not tenant_id:
        raise ValueError("tenant_id cannot be empty")
    
    _tenant_context.set(tenant_id)
    logger.debug("Tenant context set", tenant_id=tenant_id)


def get_tenant_context() -> str:
    """
    Get current tenant context.
    
    Returns:
        str: Current tenant UUID
        
    Raises:
        TenantContextError: If no tenant context is set
        
    Example:
        >>> tenant_id = get_tenant_context()
    """
    tenant_id = _tenant_context.get()
    if not tenant_id:
        raise TenantContextError(
            "No tenant context set. Call set_tenant_context() first."
        )
    return tenant_id


def clear_tenant_context() -> None:
    """Clear tenant context (useful for tests)."""
    _tenant_context.set(None)
    logger.debug("Tenant context cleared")


def set_user_context(user_id: str) -> None:
    """
    Set current user context (for audit trail).
    
    Args:
        user_id: UUID of the current user
    """
    if not user_id:
        raise ValueError("user_id cannot be empty")
    
    _user_context.set(user_id)
    logger.debug("User context set", user_id=user_id)


def get_user_context() -> str | None:
    """
    Get current user context (optional).
    
    Returns:
        str | None: Current user UUID or None if not set
    """
    return _user_context.get()


def clear_user_context() -> None:
    """Clear user context."""
    _user_context.set(None)


# =============================================================================
# SQLALCHEMY BASE
# =============================================================================

Base = declarative_base()

# Type variable for model classes
ModelType = TypeVar("ModelType", bound=Base)

# =============================================================================
# BASE MODEL MIXIN
# =============================================================================


class BaseModelMixin:
    """
    Base mixin providing primary key and timestamps.
    
    All models should inherit from this mixin to ensure consistency.
    
    Attributes:
        id: UUID primary key
        created_at: Record creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
    
    Example:
        >>> class User(Base, BaseModelMixin):
        ...     __tablename__ = "users"
        ...     email = Column(String, unique=True)
    """
    
    # UUID primary key (more secure than auto-increment integers)
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier (UUID4)",
    )
    
    # Timestamps (timezone-aware UTC)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Record creation timestamp (UTC)",
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Last update timestamp (UTC)",
    )
    
    def __repr__(self) -> str:
        """String representation of model."""
        return f"<{self.__class__.__name__}(id={self.id})>"
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert model to dictionary (for JSON serialization).
        
        Returns:
            dict: Model fields as dictionary
            
        Example:
            >>> user = User(email="test@example.com")
            >>> user_dict = user.to_dict()
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            
            # Convert UUID to string
            if isinstance(value, uuid.UUID):
                value = str(value)
            
            # Convert datetime to ISO format
            elif isinstance(value, datetime):
                value = value.isoformat()
            
            result[column.name] = value
        
        return result


# =============================================================================
# TENANT ISOLATION MIXIN
# =============================================================================


class TenantMixin:
    """
    Multi-tenant isolation mixin with Row-Level Security (RLS).
    
    Adds tenant_id foreign key and automatic isolation checks.
    Ensures data cannot leak between tenants.
    
    Attributes:
        tenant_id: Foreign key to tenants table
        tenant: SQLAlchemy relationship to Tenant model
    
    Example:
        >>> class Document(Base, BaseModelMixin, TenantMixin):
        ...     __tablename__ = "documents"
        ...     title = Column(String)
        ...
        >>> doc = Document(title="Contract", tenant_id=tenant_id)
        >>> doc.validate_tenant_access(requesting_tenant_id)
    """
    
    @declared_attr
    def tenant_id(cls) -> Column:
        """Tenant foreign key with index for performance."""
        return Column(
            UUID(as_uuid=True),
            ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
            comment="Tenant isolation - foreign key to tenants table",
        )
    
    @declared_attr
    def tenant(cls):
        """Relationship to Tenant model."""
        return relationship("Tenant", back_populates=cls.__tablename__)
    
    def validate_tenant_access(self, requesting_tenant_id: str | uuid.UUID) -> None:
        """
        Validate that requesting tenant has access to this resource.
        
        Critical security check to prevent cross-tenant data access.
        Should be called in all API endpoints before returning data.
        
        Args:
            requesting_tenant_id: UUID of tenant making the request
            
        Raises:
            TenantIsolationViolationError: If tenant doesn't match
            
        Example:
            >>> document = await db.get(Document, doc_id)
            >>> document.validate_tenant_access(current_user.tenant_id)
        """
        # Convert to string for comparison
        resource_tenant = str(self.tenant_id)
        requesting_tenant = str(requesting_tenant_id)
        
        if resource_tenant != requesting_tenant:
            logger.warning(
                "Tenant isolation violation attempt",
                resource_tenant=resource_tenant,
                requesting_tenant=requesting_tenant,
                model=self.__class__.__name__,
                resource_id=str(self.id) if hasattr(self, "id") else None,
            )
            
            raise TenantIsolationViolationError(
                message="Access denied: Resource belongs to different tenant",
                details={
                    "resource_tenant": resource_tenant,
                    "requesting_tenant": requesting_tenant,
                    "model": self.__class__.__name__,
                },
            )
    
    @classmethod
    def __declare_last__(cls):
        """
        Create composite index on (tenant_id, id) for performance.
        
        This index dramatically improves query performance for tenant-scoped queries.
        """
        Index(
            f"ix_{cls.__tablename__}_tenant_id_id",
            cls.tenant_id,
            cls.id,
        )


# =============================================================================
# AUDIT TRAIL MIXIN
# =============================================================================


class AuditMixin:
    """
    Audit trail mixin tracking who created/updated records.
    
    Essential for KVKK/GDPR compliance and security auditing.
    
    Attributes:
        created_by_id: User who created the record
        updated_by_id: User who last updated the record
        created_by: Relationship to User model
        updated_by: Relationship to User model
    
    Example:
        >>> class Contract(Base, BaseModelMixin, TenantMixin, AuditMixin):
        ...     __tablename__ = "contracts"
        ...     title = Column(String)
        ...
        >>> contract = Contract(title="NDA", created_by_id=current_user.id)
    """
    
    @declared_attr
    def created_by_id(cls) -> Column:
        """User who created this record."""
        return Column(
            UUID(as_uuid=True),
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,  # Nullable to support system-created records
            comment="User who created this record",
        )
    
    @declared_attr
    def updated_by_id(cls) -> Column:
        """User who last updated this record."""
        return Column(
            UUID(as_uuid=True),
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            comment="User who last updated this record",
        )
    
    @declared_attr
    def created_by(cls):
        """Relationship to User who created the record."""
        return relationship(
            "User",
            foreign_keys=[cls.created_by_id],
            back_populates=f"{cls.__tablename__}_created",
        )
    
    @declared_attr
    def updated_by(cls):
        """Relationship to User who updated the record."""
        return relationship(
            "User",
            foreign_keys=[cls.updated_by_id],
            back_populates=f"{cls.__tablename__}_updated",
        )


# =============================================================================
# SOFT DELETE MIXIN
# =============================================================================


class SoftDeleteMixin:
    """
    Soft delete mixin for KVKK/GDPR compliance.
    
    Instead of permanently deleting records, mark them as deleted.
    This allows:
    - Data recovery if needed
    - Audit trail preservation
    - Compliance with data retention policies
    
    Attributes:
        deleted_at: Timestamp when record was soft deleted (NULL if active)
        is_deleted: Computed property indicating deletion status
    
    Example:
        >>> class Document(Base, BaseModelMixin, SoftDeleteMixin):
        ...     __tablename__ = "documents"
        ...     title = Column(String)
        ...
        >>> document.soft_delete()
        >>> assert document.is_deleted
        >>> document.restore()
        >>> assert not document.is_deleted
    """
    
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        index=True,  # Index for filtering active records
        comment="Soft delete timestamp (NULL if active)",
    )
    
    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted."""
        return self.deleted_at is not None
    
    def soft_delete(self, user_id: str | None = None) -> None:
        """
        Soft delete this record.
        
        Args:
            user_id: Optional user ID who performed the deletion
            
        Example:
            >>> document.soft_delete(user_id=current_user.id)
        """
        self.deleted_at = datetime.now(timezone.utc)
        
        # Update audit trail if available
        if hasattr(self, "updated_by_id") and user_id:
            self.updated_by_id = uuid.UUID(user_id)
        
        logger.info(
            "Record soft deleted",
            model=self.__class__.__name__,
            record_id=str(self.id) if hasattr(self, "id") else None,
            deleted_by=user_id,
        )
    
    def restore(self, user_id: str | None = None) -> None:
        """
        Restore a soft deleted record.
        
        Args:
            user_id: Optional user ID who performed the restore
            
        Example:
            >>> document.restore(user_id=admin_user.id)
        """
        self.deleted_at = None
        
        # Update audit trail if available
        if hasattr(self, "updated_by_id") and user_id:
            self.updated_by_id = uuid.UUID(user_id)
        
        logger.info(
            "Record restored",
            model=self.__class__.__name__,
            record_id=str(self.id) if hasattr(self, "id") else None,
            restored_by=user_id,
        )
    
    @classmethod
    def __declare_last__(cls):
        """Create index for filtering active records efficiently."""
        Index(
            f"ix_{cls.__tablename__}_not_deleted",
            cls.deleted_at,
            postgresql_where=text("deleted_at IS NULL"),
        )


# =============================================================================
# TIMESTAMP MIXIN (Simpler version without id)
# =============================================================================


class TimestampMixin:
    """
    Simple timestamp mixin without primary key.
    
    Use this for models that already have a custom primary key
    but need timestamp tracking.
    
    Attributes:
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Record creation timestamp (UTC)",
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Last update timestamp (UTC)",
    )


# =============================================================================
# VERSION MIXIN (Optimistic Locking)
# =============================================================================


class VersionMixin:
    """
    Optimistic locking mixin using version counter.
    
    Prevents concurrent update conflicts by tracking version number.
    SQLAlchemy automatically checks version on UPDATE.
    
    Attributes:
        version: Version counter (incremented on each update)
    
    Example:
        >>> class Contract(Base, BaseModelMixin, VersionMixin):
        ...     __tablename__ = "contracts"
        ...     title = Column(String)
        ...
        >>> # SQLAlchemy will raise StaleDataError if version mismatch
    """
    
    version = Column(
        Integer,
        nullable=False,
        default=1,
        comment="Version counter for optimistic locking",
    )
    
    __mapper_args__ = {"version_id_col": version}


# =============================================================================
# SQLALCHEMY EVENT LISTENERS
# =============================================================================


@event.listens_for(Session, "before_flush")
def set_rls_context_on_flush(session: Session, flush_context, instances) -> None:
    """
    Automatically set PostgreSQL RLS context before flush.
    
    This ensures all queries respect Row-Level Security policies.
    The tenant_id is injected into PostgreSQL session variables.
    """
    try:
        tenant_id = get_tenant_context()
        
        # Set PostgreSQL session variable for RLS
        session.execute(
            text(f"SET LOCAL app.{TENANT_ID_FIELD_NAME} = :tenant_id"),
            {"tenant_id": tenant_id},
        )
        
        logger.debug("RLS context set for flush", tenant_id=tenant_id)
        
    except TenantContextError:
        # No tenant context - allow for system operations
        logger.debug("No tenant context for flush (system operation)")


@event.listens_for(Session, "after_flush")
def auto_set_audit_fields(session: Session, flush_context) -> None:
    """
    Automatically set created_by and updated_by fields.
    
    Uses user context to populate audit trail fields.
    """
    user_id = get_user_context()
    if not user_id:
        return  # No user context
    
    user_uuid = uuid.UUID(user_id)
    
    # Set created_by for new records
    for obj in session.new:
        if hasattr(obj, "created_by_id") and obj.created_by_id is None:
            obj.created_by_id = user_uuid
            logger.debug(
                "Auto-set created_by",
                model=obj.__class__.__name__,
                user_id=user_id,
            )
    
    # Set updated_by for modified records
    for obj in session.dirty:
        if hasattr(obj, "updated_by_id"):
            obj.updated_by_id = user_uuid
            logger.debug(
                "Auto-set updated_by",
                model=obj.__class__.__name__,
                user_id=user_id,
            )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def generate_uuid() -> uuid.UUID:
    """
    Generate a new UUID4.
    
    Returns:
        uuid.UUID: Random UUID
        
    Example:
        >>> new_id = generate_uuid()
    """
    return uuid.uuid4()


def is_valid_uuid(value: str | uuid.UUID) -> bool:
    """
    Check if string is a valid UUID.
    
    Args:
        value: String or UUID to validate
        
    Returns:
        bool: True if valid UUID
        
    Example:
        >>> is_valid_uuid("123e4567-e89b-12d3-a456-426614174000")
        True
        >>> is_valid_uuid("invalid")
        False
    """
    if isinstance(value, uuid.UUID):
        return True
    
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base
    "Base",
    "ModelType",
    # Mixins
    "BaseModelMixin",
    "TenantMixin",
    "AuditMixin",
    "SoftDeleteMixin",
    "TimestampMixin",
    "VersionMixin",
    # Context management
    "set_tenant_context",
    "get_tenant_context",
    "clear_tenant_context",
    "set_user_context",
    "get_user_context",
    "clear_user_context",
    # Utilities
    "generate_uuid",
    "is_valid_uuid",
]