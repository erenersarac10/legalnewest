"""
Contract Template model for standardized contract generation in Turkish Legal AI.

This module provides the ContractTemplate model for managing contract templates:
- Pre-defined contract templates (Turkish law compliant)
- Variable/placeholder management
- Multi-version template support
- Template categories and tags
- Custom template creation
- Template sharing (tenant/organization/public)
- Compliance validation
- Usage tracking

Template Types:
    - EMPLOYMENT: İş sözleşmeleri (Labor contracts)
    - SERVICE: Hizmet sözleşmeleri (Service agreements)
    - SALES: Satış sözleşmeleri (Sales contracts)
    - RENTAL: Kira sözleşmeleri (Rental agreements)
    - NDA: Gizlilik sözleşmeleri (Non-disclosure)
    - PARTNERSHIP: Ortaklık sözleşmeleri (Partnership)
    - LICENSE: Lisans sözleşmeleri (License agreements)
    - CONSULTING: Danışmanlık sözleşmeleri (Consulting)

Template Features:
    - Rich text content with variables {{variable_name}}
    - Required/optional variables
    - Default values
    - Validation rules
    - Conditional sections
    - Multi-language support (TR/EN)
    - Version control

Example:
    >>> # Create employment contract template
    >>> template = ContractTemplate.create_template(
    ...     tenant_id=tenant.id,
    ...     title="Belirli Süreli İş Sözleşmesi",
    ...     category=TemplateCategory.EMPLOYMENT,
    ...     content="İŞ SÖZLEŞMESİ\n\n{{employer_name}} ile {{employee_name}}...",
    ...     variables=[
    ...         {"name": "employer_name", "required": True, "type": "text"},
    ...         {"name": "employee_name", "required": True, "type": "text"},
    ...         {"name": "salary", "required": True, "type": "number"}
    ...     ],
    ...     created_by_id=user.id
    ... )
    >>> 
    >>> # Generate document from template
    >>> document = template.generate_document(
    ...     values={
    ...         "employer_name": "ABC Şirketi",
    ...         "employee_name": "Ahmet Yılmaz",
    ...         "salary": "15000"
    ...     }
    ... )
"""

import enum
import re
from datetime import datetime, timezone
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
    TemplateValidationError,
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


class TemplateCategory(str, enum.Enum):
    """
    Contract template category.
    
    Categories aligned with Turkish contract law:
    - EMPLOYMENT: İş sözleşmeleri
    - SERVICE: Hizmet sözleşmeleri
    - SALES: Satış sözleşmeleri
    - RENTAL: Kira sözleşmeleri
    - NDA: Gizlilik sözleşmeleri
    - PARTNERSHIP: Ortaklık sözleşmeleri
    - LICENSE: Lisans sözleşmeleri
    - CONSULTING: Danışmanlık sözleşmeleri
    - DISTRIBUTION: Distribütörlük sözleşmeleri
    - FRANCHISE: Franchising sözleşmeleri
    - LOAN: Kredi sözleşmeleri
    - OTHER: Diğer
    """
    
    EMPLOYMENT = "employment"
    SERVICE = "service"
    SALES = "sales"
    RENTAL = "rental"
    NDA = "nda"
    PARTNERSHIP = "partnership"
    LICENSE = "license"
    CONSULTING = "consulting"
    DISTRIBUTION = "distribution"
    FRANCHISE = "franchise"
    LOAN = "loan"
    OTHER = "other"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.EMPLOYMENT: "İş Sözleşmesi",
            self.SERVICE: "Hizmet Sözleşmesi",
            self.SALES: "Satış Sözleşmesi",
            self.RENTAL: "Kira Sözleşmesi",
            self.NDA: "Gizlilik Sözleşmesi",
            self.PARTNERSHIP: "Ortaklık Sözleşmesi",
            self.LICENSE: "Lisans Sözleşmesi",
            self.CONSULTING: "Danışmanlık Sözleşmesi",
            self.DISTRIBUTION: "Distribütörlük Sözleşmesi",
            self.FRANCHISE: "Franchising Sözleşmesi",
            self.LOAN: "Kredi Sözleşmesi",
            self.OTHER: "Diğer",
        }
        return names.get(self, self.value)


class TemplateStatus(str, enum.Enum):
    """Template lifecycle status."""
    
    DRAFT = "draft"              # Being created/edited
    ACTIVE = "active"            # Available for use
    DEPRECATED = "deprecated"    # Outdated, use newer version
    ARCHIVED = "archived"        # No longer available
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.DRAFT: "Taslak",
            self.ACTIVE: "Aktif",
            self.DEPRECATED: "Güncel Değil",
            self.ARCHIVED: "Arşivlenmiş",
        }
        return names.get(self, self.value)


class TemplateVisibility(str, enum.Enum):
    """Template visibility/sharing level."""
    
    PRIVATE = "private"          # Only creator can see
    TENANT = "tenant"            # All users in tenant
    ORGANIZATION = "organization"  # All users in organization
    PUBLIC = "public"            # All platform users
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.PRIVATE: "Özel",
            self.TENANT: "Şirket İçi",
            self.ORGANIZATION: "Organizasyon",
            self.PUBLIC: "Herkese Açık",
        }
        return names.get(self, self.value)


class VariableType(str, enum.Enum):
    """Variable data type for validation."""
    
    TEXT = "text"                # Free text
    NUMBER = "number"            # Numeric value
    DATE = "date"                # Date value
    EMAIL = "email"              # Email address
    PHONE = "phone"              # Phone number
    CURRENCY = "currency"        # Monetary value
    PERCENTAGE = "percentage"    # Percentage value
    SELECT = "select"            # Dropdown selection
    BOOLEAN = "boolean"          # Yes/No checkbox
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# CONTRACT TEMPLATE MODEL
# =============================================================================


class ContractTemplate(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Contract Template model for standardized document generation.
    
    Provides reusable contract templates with:
    - Variable substitution
    - Validation rules
    - Version control
    - Multi-language support
    - Compliance checking
    
    Template Structure:
        Content: Rich text with {{variable}} placeholders
        Variables: Definitions (name, type, required, validation)
        Metadata: Category, tags, language, legal references
    
    Template Usage:
    1. Create template with content and variables
    2. User selects template
    3. User fills in variable values
    4. System validates input
    5. Generate final document
    
    Attributes:
        title: Template name
        description: Template description
        
        category: Template category
        tags: Search tags (array)
        
        status: Template status
        visibility: Sharing level
        
        content: Template content with variables
        language: Content language (tr, en)
        
        variables: Variable definitions (JSON array)
        sections: Template sections (JSON array)
        
        version: Template version (1.0, 1.1, etc.)
        parent_template_id: Previous version
        parent_template: Parent relationship
        
        created_by_id: Template creator
        created_by: Creator relationship
        
        approved_by_id: Who approved template (if applicable)
        approved_at: Approval timestamp
        
        usage_count: How many times used
        last_used_at: Last usage timestamp
        
        legal_references: Referenced laws (array)
        compliance_notes: Compliance information
        
        metadata: Additional data (formatting, styling, etc.)
        
    Relationships:
        tenant: Parent tenant
        created_by: Template creator
        approved_by: Template approver
        parent_template: Previous version
    """
    
    __tablename__ = "contract_templates"
    
    # =========================================================================
    # TEMPLATE IDENTITY
    # =========================================================================
    
    title = Column(
        String(500),
        nullable=False,
        comment="Template title/name",
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Template description (purpose, usage notes)",
    )
    
    # =========================================================================
    # CLASSIFICATION
    # =========================================================================
    
    category = Column(
        Enum(TemplateCategory, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Template category",
    )
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Search tags for categorization",
    )
    
    # =========================================================================
    # STATUS & VISIBILITY
    # =========================================================================
    
    status = Column(
        Enum(TemplateStatus, native_enum=False, length=50),
        nullable=False,
        default=TemplateStatus.DRAFT,
        index=True,
        comment="Template lifecycle status",
    )
    
    visibility = Column(
        Enum(TemplateVisibility, native_enum=False, length=50),
        nullable=False,
        default=TemplateVisibility.PRIVATE,
        index=True,
        comment="Template sharing/visibility level",
    )
    
    # =========================================================================
    # CONTENT
    # =========================================================================
    
    content = Column(
        Text,
        nullable=False,
        comment="Template content with {{variable}} placeholders",
    )
    
    language = Column(
        String(5),
        nullable=False,
        default="tr",
        comment="Content language (tr, en)",
    )
    
    # =========================================================================
    # VARIABLES & STRUCTURE
    # =========================================================================
    
    variables = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Variable definitions with validation rules (array of objects)",
    )
    
    sections = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Template sections for organization (array)",
    )
    
    # =========================================================================
    # VERSIONING
    # =========================================================================
    
    version = Column(
        String(20),
        nullable=False,
        default="1.0",
        comment="Template version (semantic versioning)",
    )
    
    parent_template_id = Column(
        UUID(as_uuid=True),
        ForeignKey("contract_templates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Previous template version (for version history)",
    )
    
    parent_template = relationship(
        "ContractTemplate",
        remote_side="ContractTemplate.id",
        backref="child_templates",
        foreign_keys=[parent_template_id],
    )
    
    # =========================================================================
    # CREATOR & APPROVAL
    # =========================================================================
    
    created_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who created the template",
    )
    
    created_by = relationship(
        "User",
        foreign_keys=[created_by_id],
        back_populates="contract_templates_created",
    )
    
    approved_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="User who approved the template",
    )
    
    approved_by = relationship(
        "User",
        foreign_keys=[approved_by_id],
        back_populates="contract_templates_approved",
    )
    
    approved_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Template approval timestamp",
    )
    
    # =========================================================================
    # USAGE TRACKING
    # =========================================================================
    
    usage_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times template was used",
    )
    
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time template was used",
    )
    
    # =========================================================================
    # LEGAL COMPLIANCE
    # =========================================================================
    
    legal_references = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Referenced Turkish laws (TBK, TTK, etc.)",
    )
    
    compliance_notes = Column(
        Text,
        nullable=True,
        comment="Compliance information and warnings",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (formatting, styling, conditions)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for template catalog
        Index(
            "ix_contract_templates_catalog",
            "category",
            "status",
            "visibility",
        ),
        
        # Index for tenant templates
        Index(
            "ix_contract_templates_tenant",
            "tenant_id",
            "status",
        ),
        
        # Index for active templates
        Index(
            "ix_contract_templates_active",
            "status",
            "visibility",
            postgresql_where="status = 'active' AND deleted_at IS NULL",
        ),
        
        # Index for popular templates
        Index(
            "ix_contract_templates_popular",
            "usage_count",
            "status",
            postgresql_where="status = 'active'",
        ),
        
        # Index for creator's templates
        Index(
            "ix_contract_templates_creator",
            "created_by_id",
            "created_at",
        ),
        
        # Check: usage count non-negative
        CheckConstraint(
            "usage_count >= 0",
            name="ck_contract_templates_usage_count",
        ),
    )
    
    # =========================================================================
    # TEMPLATE CREATION
    # =========================================================================
    
    @classmethod
    def create_template(
        cls,
        tenant_id: UUIDType,
        title: str,
        category: TemplateCategory,
        content: str,
        variables: list[dict[str, Any]],
        created_by_id: UUIDType,
        description: str | None = None,
        tags: list[str] | None = None,
        language: str = "tr",
        visibility: TemplateVisibility = TemplateVisibility.TENANT,
        legal_references: list[str] | None = None,
    ) -> "ContractTemplate":
        """
        Create a new contract template.
        
        Args:
            tenant_id: Tenant UUID
            title: Template title
            category: Template category
            content: Template content with {{variables}}
            variables: Variable definitions
            created_by_id: Creator user UUID
            description: Template description
            tags: Search tags
            language: Content language
            visibility: Sharing level
            legal_references: Referenced laws
            
        Returns:
            ContractTemplate: New template instance
            
        Example:
            >>> template = ContractTemplate.create_template(
            ...     tenant_id=tenant.id,
            ...     title="Belirli Süreli İş Sözleşmesi",
            ...     category=TemplateCategory.EMPLOYMENT,
            ...     content="İŞ SÖZLEŞMESİ\n\nİşveren: {{employer_name}}...",
            ...     variables=[
            ...         {
            ...             "name": "employer_name",
            ...             "label": "İşveren Adı",
            ...             "type": "text",
            ...             "required": True,
            ...             "description": "İşverenin tam ünvanı"
            ...         },
            ...         {
            ...             "name": "salary",
            ...             "label": "Aylık Brüt Ücret",
            ...             "type": "currency",
            ...             "required": True,
            ...             "validation": {"min": 0}
            ...         }
            ...     ],
            ...     created_by_id=user.id,
            ...     tags=["iş", "belirli süreli", "tam zamanlı"],
            ...     legal_references=["İş Kanunu No. 4857", "TBK"]
            ... )
        """
        template = cls(
            tenant_id=tenant_id,
            title=title,
            description=description,
            category=category,
            content=content,
            variables=variables,
            created_by_id=created_by_id,
            tags=tags or [],
            language=language,
            visibility=visibility,
            status=TemplateStatus.DRAFT,
            legal_references=legal_references or [],
        )
        
        # Validate variables in content
        template._validate_variables()
        
        logger.info(
            "Contract template created",
            template_id=str(template.id),
            title=title,
            category=category.value,
        )
        
        return template
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def activate(self) -> None:
        """Activate template for use."""
        # Validate before activation
        self._validate_template()
        
        self.status = TemplateStatus.ACTIVE
        
        logger.info(
            "Contract template activated",
            template_id=str(self.id),
        )
    
    def deprecate(self, reason: str | None = None) -> None:
        """
        Deprecate template (mark as outdated).
        
        Args:
            reason: Deprecation reason
        """
        self.status = TemplateStatus.DEPRECATED
        
        if reason:
            if "deprecation" not in self.metadata:
                self.metadata["deprecation"] = {}
            self.metadata["deprecation"]["reason"] = reason
            self.metadata["deprecation"]["deprecated_at"] = datetime.now(timezone.utc).isoformat()
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "metadata")
        
        logger.info(
            "Contract template deprecated",
            template_id=str(self.id),
            reason=reason,
        )
    
    def archive(self) -> None:
        """Archive template."""
        self.status = TemplateStatus.ARCHIVED
        
        logger.info(
            "Contract template archived",
            template_id=str(self.id),
        )
    
    def approve(self, approver_id: UUIDType) -> None:
        """
        Approve template.
        
        Args:
            approver_id: Approver user UUID
        """
        self.approved_by_id = approver_id
        self.approved_at = datetime.now(timezone.utc)
        
        # Activate if in draft
        if self.status == TemplateStatus.DRAFT:
            self.activate()
        
        logger.info(
            "Contract template approved",
            template_id=str(self.id),
            approver_id=str(approver_id),
        )
    
    # =========================================================================
    # VARIABLE MANAGEMENT
    # =========================================================================
    
    def get_required_variables(self) -> list[dict[str, Any]]:
        """
        Get required variable definitions.
        
        Returns:
            list: Required variables
        """
        if not self.variables:
            return []
        
        return [
            var for var in self.variables
            if var.get("required", False)
        ]
    
    def extract_variables_from_content(self) -> list[str]:
        """
        Extract variable names from content.
        
        Returns:
            list: Variable names found in content
        """
        # Pattern: {{variable_name}}
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, self.content)
        return list(set(matches))  # Remove duplicates
    
    def _validate_variables(self) -> None:
        """
        Validate that all variables in content are defined.
        
        Raises:
            TemplateValidationError: If validation fails
        """
        content_vars = set(self.extract_variables_from_content())
        defined_vars = set(var["name"] for var in self.variables)
        
        # Check for undefined variables
        undefined = content_vars - defined_vars
        if undefined:
            raise TemplateValidationError(
                message=f"Undefined variables in content: {', '.join(undefined)}",
                template_id=str(self.id),
                undefined_variables=list(undefined),
            )
        
        # Check for unused variables (warning only)
        unused = defined_vars - content_vars
        if unused:
            logger.warning(
                "Unused variables in template",
                template_id=str(self.id),
                unused_variables=list(unused),
            )
    
    def validate_values(self, values: dict[str, Any]) -> dict[str, str]:
        """
        Validate variable values against definitions.
        
        Args:
            values: Variable values to validate
            
        Returns:
            dict: Validation errors (empty if valid)
            
        Example:
            >>> errors = template.validate_values({
            ...     "employer_name": "ABC Şirketi",
            ...     "salary": "-1000"  # Invalid: negative
            ... })
            >>> if errors:
            ...     print(errors)  # {"salary": "Value must be positive"}
        """
        errors = {}
        
        # Check required variables
        for var in self.get_required_variables():
            var_name = var["name"]
            if var_name not in values or not values[var_name]:
                errors[var_name] = f"Required field: {var.get('label', var_name)}"
        
        # Validate types and rules
        for var in self.variables:
            var_name = var["name"]
            if var_name not in values:
                continue
            
            value = values[var_name]
            var_type = var.get("type", "text")
            
            # Type-specific validation
            if var_type == "number":
                try:
                    num_value = float(value)
                    validation = var.get("validation", {})
                    if "min" in validation and num_value < validation["min"]:
                        errors[var_name] = f"Value must be at least {validation['min']}"
                    if "max" in validation and num_value > validation["max"]:
                        errors[var_name] = f"Value must be at most {validation['max']}"
                except (ValueError, TypeError):
                    errors[var_name] = "Must be a valid number"
            
            elif var_type == "email":
                if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", str(value)):
                    errors[var_name] = "Must be a valid email address"
        
        return errors
    
    # =========================================================================
    # DOCUMENT GENERATION
    # =========================================================================
    
    def generate_content(self, values: dict[str, Any]) -> str:
        """
        Generate document content from template and values.
        
        Args:
            values: Variable values
            
        Returns:
            str: Generated content
            
        Raises:
            TemplateValidationError: If validation fails
            
        Example:
            >>> content = template.generate_content({
            ...     "employer_name": "ABC Şirketi",
            ...     "employee_name": "Ahmet Yılmaz",
            ...     "salary": "15000"
            ... })
        """
        # Validate values
        errors = self.validate_values(values)
        if errors:
            raise TemplateValidationError(
                message="Invalid variable values",
                template_id=str(self.id),
                validation_errors=errors,
            )
        
        # Replace variables
        content = self.content
        for var_name, value in values.items():
            placeholder = f"{{{{{var_name}}}}}"
            content = content.replace(placeholder, str(value))
        
        return content
    
    # =========================================================================
    # USAGE TRACKING
    # =========================================================================
    
    def increment_usage(self) -> None:
        """Increment usage counter."""
        self.usage_count += 1
        self.last_used_at = datetime.now(timezone.utc)
        
        logger.debug(
            "Template usage incremented",
            template_id=str(self.id),
            usage_count=self.usage_count,
        )
    
    # =========================================================================
    # VERSIONING
    # =========================================================================
    
    def create_new_version(
        self,
        new_content: str | None = None,
        new_variables: list[dict[str, Any]] | None = None,
        version: str | None = None,
    ) -> "ContractTemplate":
        """
        Create a new version of this template.
        
        Args:
            new_content: Updated content
            new_variables: Updated variables
            version: Version identifier
            
        Returns:
            ContractTemplate: New version
            
        Example:
            >>> new_version = template.create_new_version(
            ...     new_content="Updated contract text...",
            ...     version="2.0"
            ... )
        """
        new_template = ContractTemplate(
            tenant_id=self.tenant_id,
            title=self.title,
            description=self.description,
            category=self.category,
            content=new_content or self.content,
            variables=new_variables or self.variables,
            created_by_id=self.created_by_id,
            tags=self.tags,
            language=self.language,
            visibility=self.visibility,
            status=TemplateStatus.DRAFT,
            version=version or self._increment_version(),
            parent_template_id=self.id,
            legal_references=self.legal_references,
        )
        
        # Deprecate old version
        self.deprecate(reason="New version created")
        
        logger.info(
            "New template version created",
            old_template_id=str(self.id),
            new_template_id=str(new_template.id),
            version=new_template.version,
        )
        
        return new_template
    
    def _increment_version(self) -> str:
        """
        Auto-increment version number.
        
        Returns:
            str: New version number
        """
        try:
            parts = self.version.split(".")
            major, minor = int(parts[0]), int(parts[1])
            return f"{major}.{minor + 1}"
        except (ValueError, IndexError):
            return "1.1"
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_template(self) -> None:
        """
        Validate template for activation.
        
        Raises:
            TemplateValidationError: If validation fails
        """
        # Check required fields
        if not self.title or not self.content:
            raise TemplateValidationError(
                message="Title and content are required",
                template_id=str(self.id),
            )
        
        # Validate variables
        self._validate_variables()
        
        # Check for required variables
        if not self.get_required_variables():
            logger.warning(
                "Template has no required variables",
                template_id=str(self.id),
            )
    
    @validates("title")
    def validate_title(self, key: str, title: str) -> str:
        """Validate title."""
        if not title or not title.strip():
            raise ValidationError(
                message="Title cannot be empty",
                field="title",
            )
        
        return title.strip()
    
    @validates("content")
    def validate_content(self, key: str, content: str) -> str:
        """Validate content."""
        if not content or not content.strip():
            raise ValidationError(
                message="Content cannot be empty",
                field="content",
            )
        
        return content.strip()
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ContractTemplate("
            f"id={self.id}, "
            f"title={self.title}, "
            f"category={self.category.value}"
            f")>"
        )
    
    def to_dict(self, include_content: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_content: Include full content (can be large)
            
        Returns:
            dict: Template data
        """
        data = super().to_dict()
        
        # Remove content if requested (for list views)
        if not include_content:
            data.pop("content", None)
        
        # Add display names
        data["category_display"] = self.category.display_name_tr
        data["status_display"] = self.status.display_name_tr
        data["visibility_display"] = self.visibility.display_name_tr
        
        # Add computed fields
        data["required_variable_count"] = len(self.get_required_variables())
        data["total_variable_count"] = len(self.variables) if self.variables else 0
        data["is_approved"] = self.approved_at is not None
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ContractTemplate",
    "TemplateCategory",
    "TemplateStatus",
    "TemplateVisibility",
    "VariableType",
]
