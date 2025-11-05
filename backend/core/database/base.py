"""
SQLAlchemy Base and model mixins for Turkish Legal AI.

This module provides:
- Declarative Base for all models
- Common model mixins (timestamps, soft delete, etc.)
- UUID primary keys
- Audit fields
- Tenant isolation support

All database models should inherit from Base and use appropriate mixins.

Example:
    >>> from backend.core.database.base import Base, TimestampMixin, UUIDMixin
    >>> 
    >>> class User(Base, UUIDMixin, TimestampMixin):
    ...     __tablename__ = "users"
    ...     email = Column(String, unique=True, nullable=False)
"""
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from backend.core.config.settings import settings


# =============================================================================
# DECLARATIVE BASE
# =============================================================================

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.
    
    Provides common functionality for all database models.
    """
    
    # Type annotation for id (will be overridden in mixins)
    id: Any
    
    def dict(self) -> dict[str, Any]:
        """
        Convert model to dictionary.
        
        Returns:
            dict: Model data as dictionary
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        """String representation of model."""
        columns = ', '.join(
            f"{col.name}={getattr(self, col.name)!r}"
            for col in self.__table__.columns
        )
        return f"{self.__class__.__name__}({columns})"


# =============================================================================
# MODEL MIXINS
# =============================================================================

class UUIDMixin:
    """
    Mixin for UUID primary key.
    
    Provides a UUID4 primary key column named 'id'.
    
    Example:
        >>> class User(Base, UUIDMixin):
        ...     __tablename__ = "users"
        ...     email = Column(String)
    """
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class TimestampMixin:
    """
    Mixin for created_at and updated_at timestamps.
    
    Automatically tracks record creation and update times.
    
    Example:
        >>> class Document(Base, TimestampMixin):
        ...     __tablename__ = "documents"
        ...     title = Column(String)
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class SoftDeleteMixin:
    """
    Mixin for soft delete functionality.
    
    Records are marked as deleted rather than physically removed.
    Useful for audit trails and data recovery.
    
    Example:
        >>> class User(Base, SoftDeleteMixin):
        ...     __tablename__ = "users"
        ...     email = Column(String)
        >>> 
        >>> # Soft delete
        >>> user.deleted_at = datetime.now(timezone.utc)
        >>> user.is_deleted = True
    """
    
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )
    
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    def soft_delete(self) -> None:
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self) -> None:
        """Restore soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class TenantMixin:
    """
    Mixin for multi-tenant support.
    
    Isolates data by tenant_id for SaaS applications.
    
    Example:
        >>> class Document(Base, TenantMixin):
        ...     __tablename__ = "documents"
        ...     title = Column(String)
    """
    
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )


class AuditMixin:
    """
    Mixin for audit trail fields.
    
    Tracks who created and updated records.
    
    Example:
        >>> class Document(Base, AuditMixin):
        ...     __tablename__ = "documents"
        ...     title = Column(String)
    """
    
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )


class VersionMixin:
    """
    Mixin for optimistic locking.
    
    Prevents concurrent update conflicts using version numbers.
    
    Example:
        >>> class Document(Base, VersionMixin):
        ...     __tablename__ = "documents"
        ...     title = Column(String)
    """
    
    version: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
    )


# =============================================================================
# COMBINED MIXINS
# =============================================================================

class BaseModelMixin(UUIDMixin, TimestampMixin):
    """
    Standard mixin combining UUID and timestamps.
    
    Most models should use this as a base.
    
    Example:
        >>> class User(Base, BaseModelMixin):
        ...     __tablename__ = "users"
        ...     email = Column(String)
    """
    pass


class FullAuditMixin(BaseModelMixin, AuditMixin, SoftDeleteMixin):
    """
    Complete audit trail mixin.
    
    Combines UUID, timestamps, audit fields, and soft delete.
    
    Example:
        >>> class ImportantDocument(Base, FullAuditMixin):
        ...     __tablename__ = "important_documents"
        ...     content = Column(Text)
    """
    pass


class TenantModelMixin(BaseModelMixin, TenantMixin):
    """
    Standard mixin for multi-tenant models.
    
    Combines UUID, timestamps, and tenant isolation.
    
    Example:
        >>> class TenantDocument(Base, TenantModelMixin):
        ...     __tablename__ = "tenant_documents"
        ...     title = Column(String)
    """
    pass


# =============================================================================
# TABLE NAME CONVENTION
# =============================================================================

def generate_table_name(cls_name: str) -> str:
    """
    Generate table name from class name.
    
    Converts CamelCase to snake_case and pluralizes.
    
    Args:
        cls_name: Class name
        
    Returns:
        str: Table name
        
    Example:
        >>> generate_table_name("User")
        'users'
        >>> generate_table_name("LegalDocument")
        'legal_documents'
    """
    import re
    
    # Convert CamelCase to snake_case
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls_name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    # Simple pluralization (can be improved)
    if name.endswith('y'):
        name = name[:-1] + 'ies'
    elif name.endswith('s'):
        name = name + 'es'
    else:
        name = name + 's'
    
    return name


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base
    "Base",
    # Mixins
    "UUIDMixin",
    "TimestampMixin",
    "SoftDeleteMixin",
    "TenantMixin",
    "AuditMixin",
    "VersionMixin",
    # Combined Mixins
    "BaseModelMixin",
    "FullAuditMixin",
    "TenantModelMixin",
    # Utilities
    "generate_table_name",
]
