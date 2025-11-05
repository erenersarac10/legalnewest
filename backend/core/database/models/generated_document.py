"""
Generated Document model for AI-generated legal documents in Turkish Legal AI.

This module provides the GeneratedDocument model for tracking AI-generated documents:
- Template-based document generation
- AI-drafted documents (contracts, letters, forms)
- Generation history and versions
- Quality tracking and review
- User feedback on generated content
- Export to multiple formats
- Legal compliance validation

Generation Types:
    - TEMPLATE: Generated from template with variable substitution
    - AI_DRAFT: AI-drafted from scratch (GPT/Claude)
    - AI_ASSISTED: User-drafted with AI assistance
    - REVISION: AI-revised version of existing document
    - TRANSLATION: AI-translated document
    - SUMMARY: AI-generated summary

Document Types:
    - CONTRACT: Contracts (employment, service, sales, etc.)
    - LETTER: Official letters (demand, notice, etc.)
    - FORM: Legal forms (petition, application, etc.)
    - AGREEMENT: Agreements and protocols
    - POLICY: Policies and procedures
    - NOTICE: Legal notices

Example:
    >>> # Generate contract from template
    >>> doc = GeneratedDocument.create_from_template(
    ...     template_id=template.id,
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     title="İş Sözleşmesi - Ahmet Yılmaz",
    ...     variables={
    ...         "employer_name": "ABC Şirketi",
    ...         "employee_name": "Ahmet Yılmaz",
    ...         "salary": "15000"
    ...     }
    ... )
    >>> 
    >>> # AI-drafted document
    >>> doc = GeneratedDocument.create_ai_draft(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     document_type=GeneratedDocumentType.CONTRACT,
    ...     prompt="Freelance yazılımcı için hizmet sözleşmesi hazırla",
    ...     model="claude-sonnet-4.5"
    ... )
"""

import enum
from datetime import datetime, timezone
from typing import Any
from uuid import UUID as UUIDType

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
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

from backend.core.exceptions import ValidationError
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


class GenerationType(str, enum.Enum):
    """
    Document generation method.
    
    Types:
    - TEMPLATE: Generated from template with variables
    - AI_DRAFT: AI-drafted from scratch
    - AI_ASSISTED: User-drafted with AI suggestions
    - REVISION: AI-revised existing document
    - TRANSLATION: AI-translated document
    - SUMMARY: AI-generated summary
    - EXTRACTION: AI-extracted from source
    """
    
    TEMPLATE = "template"
    AI_DRAFT = "ai_draft"
    AI_ASSISTED = "ai_assisted"
    REVISION = "revision"
    TRANSLATION = "translation"
    SUMMARY = "summary"
    EXTRACTION = "extraction"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.TEMPLATE: "Şablondan Oluşturuldu",
            self.AI_DRAFT: "Yapay Zeka Taslağı",
            self.AI_ASSISTED: "Yapay Zeka Destekli",
            self.REVISION: "Yapay Zeka Revizyonu",
            self.TRANSLATION: "Çeviri",
            self.SUMMARY: "Özet",
            self.EXTRACTION: "Çıkarım",
        }
        return names.get(self, self.value)


class GeneratedDocumentType(str, enum.Enum):
    """Generated document type."""
    
    CONTRACT = "contract"          # Sözleşme
    LETTER = "letter"              # Mektup
    FORM = "form"                  # Form
    AGREEMENT = "agreement"        # Anlaşma
    POLICY = "policy"              # Politika
    NOTICE = "notice"              # Bildirim
    PETITION = "petition"          # Dilekçe
    RESPONSE = "response"          # Cevap/Yanıt
    REPORT = "report"              # Rapor
    OTHER = "other"                # Diğer
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.CONTRACT: "Sözleşme",
            self.LETTER: "Mektup",
            self.FORM: "Form",
            self.AGREEMENT: "Anlaşma",
            self.POLICY: "Politika",
            self.NOTICE: "Bildirim",
            self.PETITION: "Dilekçe",
            self.RESPONSE: "Cevap",
            self.REPORT: "Rapor",
            self.OTHER: "Diğer",
        }
        return names.get(self, self.value)


class GenerationStatus(str, enum.Enum):
    """Document generation status."""
    
    PENDING = "pending"              # Queued for generation
    GENERATING = "generating"        # Currently generating
    GENERATED = "generated"          # Successfully generated
    REVIEWED = "reviewed"            # Reviewed by user
    APPROVED = "approved"            # Approved for use
    EXPORTED = "exported"            # Exported to file
    FAILED = "failed"                # Generation failed
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.PENDING: "Bekliyor",
            self.GENERATING: "Oluşturuluyor",
            self.GENERATED: "Oluşturuldu",
            self.REVIEWED: "İncelendi",
            self.APPROVED: "Onaylandı",
            self.EXPORTED: "Dışa Aktarıldı",
            self.FAILED: "Başarısız",
        }
        return names.get(self, self.value)


class QualityLevel(str, enum.Enum):
    """Generated content quality assessment."""
    
    EXCELLENT = "excellent"    # 90-100% quality
    GOOD = "good"              # 70-89% quality
    ACCEPTABLE = "acceptable"  # 50-69% quality
    POOR = "poor"              # Below 50% quality
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# GENERATED DOCUMENT MODEL
# =============================================================================


class GeneratedDocument(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Generated Document model for AI-generated legal documents.
    
    Tracks documents generated by:
    - Template substitution
    - AI drafting (GPT/Claude)
    - AI-assisted writing
    - Document revision
    - Translation
    
    Generation Workflow:
    1. User initiates generation (template or prompt)
    2. System generates content (AI or template)
    3. User reviews generated content
    4. Optional: User edits and regenerates
    5. User approves and exports
    
    Quality Control:
        - Automatic quality scoring
        - User feedback collection
        - Compliance checking
        - Human review tracking
    
    Attributes:
        title: Document title
        
        generation_type: How document was generated
        document_type: Type of document
        status: Generation status
        
        user_id: User who requested generation
        user: User relationship
        
        template_id: Source template (if template-based)
        template: Template relationship
        
        source_document_id: Source document (if revision/translation)
        source_document: Source relationship
        
        content: Generated content (text)
        content_html: Generated content (HTML)
        
        prompt: AI prompt used (if AI-generated)
        variables: Template variables (JSON)
        
        model_used: AI model identifier
        model_version: Model version
        
        generation_params: Generation parameters (JSON)
        
        started_at: Generation start timestamp
        completed_at: Generation completion timestamp
        generation_time_seconds: Duration
        
        token_count: Total tokens used
        estimated_cost: Generation cost
        
        quality_score: Quality assessment (0-1)
        quality_level: Quality level enum
        
        user_rating: User satisfaction rating (1-5)
        user_feedback: User feedback text
        
        is_approved: User approved for use
        approved_at: Approval timestamp
        
        export_format: Export format (docx, pdf, etc.)
        exported_at: Export timestamp
        
        revisions: Revision history (JSON array)
        
        metadata: Additional context
        
        error_message: Error details if failed
        
    Relationships:
        tenant: Parent tenant
        user: User who generated document
        template: Source template
        source_document: Source document (if applicable)
    """
    
    __tablename__ = "generated_documents"
    
    # =========================================================================
    # DOCUMENT IDENTITY
    # =========================================================================
    
    title = Column(
        String(500),
        nullable=False,
        comment="Generated document title",
    )
    
    # =========================================================================
    # GENERATION TYPE
    # =========================================================================
    
    generation_type = Column(
        Enum(GenerationType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="How document was generated",
    )
    
    document_type = Column(
        Enum(GeneratedDocumentType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of generated document",
    )
    
    status = Column(
        Enum(GenerationStatus, native_enum=False, length=50),
        nullable=False,
        default=GenerationStatus.PENDING,
        index=True,
        comment="Generation status",
    )
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who requested generation",
    )
    
    user = relationship(
        "User",
        back_populates="generated_documents",
    )
    
    # =========================================================================
    # SOURCE TEMPLATE
    # =========================================================================
    
    template_id = Column(
        UUID(as_uuid=True),
        ForeignKey("contract_templates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source template (if template-based generation)",
    )
    
    template = relationship(
        "ContractTemplate",
        back_populates="generated_documents",
    )
    
    # =========================================================================
    # SOURCE DOCUMENT (for revisions/translations)
    # =========================================================================
    
    source_document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source document (if revision/translation/summary)",
    )
    
    source_document = relationship(
        "Document",
        back_populates="generated_documents",
    )
    
    # =========================================================================
    # CONTENT
    # =========================================================================
    
    content = Column(
        Text,
        nullable=True,
        comment="Generated content (plain text or markdown)",
    )
    
    content_html = Column(
        Text,
        nullable=True,
        comment="Generated content (HTML format)",
    )
    
    # =========================================================================
    # GENERATION INPUTS
    # =========================================================================
    
    prompt = Column(
        Text,
        nullable=True,
        comment="AI prompt used for generation (if AI-generated)",
    )
    
    variables = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Template variables used (if template-based)",
    )
    
    # =========================================================================
    # AI MODEL INFORMATION
    # =========================================================================
    
    model_used = Column(
        String(100),
        nullable=True,
        comment="AI model identifier (claude-sonnet-4.5, gpt-4, etc.)",
    )
    
    model_version = Column(
        String(50),
        nullable=True,
        comment="Model version",
    )
    
    generation_params = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Generation parameters (temperature, max_tokens, etc.)",
    )
    
    # =========================================================================
    # TIMING
    # =========================================================================
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When generation started",
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When generation completed",
    )
    
    generation_time_seconds = Column(
        Integer,
        nullable=True,
        comment="Generation duration in seconds",
    )
    
    # =========================================================================
    # COST TRACKING
    # =========================================================================
    
    token_count = Column(
        Integer,
        nullable=True,
        comment="Total tokens used (prompt + completion)",
    )
    
    estimated_cost = Column(
        Float,
        nullable=True,
        comment="Estimated generation cost (USD)",
    )
    
    # =========================================================================
    # QUALITY ASSESSMENT
    # =========================================================================
    
    quality_score = Column(
        Float,
        nullable=True,
        comment="Quality score (0.0-1.0, automated assessment)",
    )
    
    quality_level = Column(
        Enum(QualityLevel, native_enum=False, length=50),
        nullable=True,
        comment="Quality level (excellent, good, acceptable, poor)",
    )
    
    # =========================================================================
    # USER FEEDBACK
    # =========================================================================
    
    user_rating = Column(
        Integer,
        nullable=True,
        comment="User satisfaction rating (1-5 stars)",
    )
    
    user_feedback = Column(
        Text,
        nullable=True,
        comment="User feedback text",
    )
    
    # =========================================================================
    # APPROVAL
    # =========================================================================
    
    is_approved = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="User approved document for use",
    )
    
    approved_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When document was approved",
    )
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    export_format = Column(
        String(20),
        nullable=True,
        comment="Export format (docx, pdf, txt, html)",
    )
    
    exported_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When document was exported",
    )
    
    # =========================================================================
    # REVISIONS
    # =========================================================================
    
    revisions = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Revision history (array of revision objects)",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (generation settings, features used, etc.)",
    )
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if generation failed",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user's documents
        Index(
            "ix_generated_documents_user",
            "user_id",
            "created_at",
        ),
        
        # Index for template usage
        Index(
            "ix_generated_documents_template",
            "template_id",
            "created_at",
            postgresql_where="template_id IS NOT NULL",
        ),
        
        # Index for generation type
        Index(
            "ix_generated_documents_type",
            "tenant_id",
            "generation_type",
            "status",
        ),
        
        # Index for approved documents
        Index(
            "ix_generated_documents_approved",
            "user_id",
            "is_approved",
            postgresql_where="is_approved = true",
        ),
        
        # Index for quality analysis
        Index(
            "ix_generated_documents_quality",
            "quality_level",
            "user_rating",
            postgresql_where="status = 'generated' OR status = 'approved'",
        ),
        
        # Check: user rating range
        CheckConstraint(
            "user_rating IS NULL OR (user_rating >= 1 AND user_rating <= 5)",
            name="ck_generated_documents_rating",
        ),
        
        # Check: quality score range
        CheckConstraint(
            "quality_score IS NULL OR (quality_score >= 0.0 AND quality_score <= 1.0)",
            name="ck_generated_documents_quality_score",
        ),
        
        # Check: token count non-negative
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="ck_generated_documents_tokens",
        ),
    )
    
    # =========================================================================
    # DOCUMENT CREATION
    # =========================================================================
    
    @classmethod
    def create_from_template(
        cls,
        template_id: UUIDType,
        user_id: UUIDType,
        tenant_id: UUIDType,
        title: str,
        variables: dict[str, Any],
        document_type: GeneratedDocumentType | None = None,
    ) -> "GeneratedDocument":
        """
        Create document from template.
        
        Args:
            template_id: Template UUID
            user_id: User UUID
            tenant_id: Tenant UUID
            title: Document title
            variables: Template variable values
            document_type: Document type (optional)
            
        Returns:
            GeneratedDocument: New instance
            
        Example:
            >>> doc = GeneratedDocument.create_from_template(
            ...     template_id=template.id,
            ...     user_id=user.id,
            ...     tenant_id=tenant.id,
            ...     title="İş Sözleşmesi - Ahmet Yılmaz",
            ...     variables={
            ...         "employer_name": "ABC Şirketi",
            ...         "employee_name": "Ahmet Yılmaz",
            ...         "start_date": "01.01.2025",
            ...         "salary": "15000"
            ...     },
            ...     document_type=GeneratedDocumentType.CONTRACT
            ... )
        """
        doc = cls(
            template_id=template_id,
            user_id=user_id,
            tenant_id=tenant_id,
            title=title,
            generation_type=GenerationType.TEMPLATE,
            document_type=document_type or GeneratedDocumentType.OTHER,
            status=GenerationStatus.PENDING,
            variables=variables,
        )
        
        logger.info(
            "Generated document created from template",
            document_id=str(doc.id),
            template_id=str(template_id),
            user_id=str(user_id),
        )
        
        return doc
    
    @classmethod
    def create_ai_draft(
        cls,
        user_id: UUIDType,
        tenant_id: UUIDType,
        document_type: GeneratedDocumentType,
        prompt: str,
        title: str | None = None,
        model: str = "claude-sonnet-4.5",
        generation_params: dict[str, Any] | None = None,
    ) -> "GeneratedDocument":
        """
        Create AI-drafted document.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant UUID
            document_type: Document type
            prompt: Generation prompt
            title: Document title
            model: AI model to use
            generation_params: Generation parameters
            
        Returns:
            GeneratedDocument: New instance
            
        Example:
            >>> doc = GeneratedDocument.create_ai_draft(
            ...     user_id=user.id,
            ...     tenant_id=tenant.id,
            ...     document_type=GeneratedDocumentType.CONTRACT,
            ...     prompt="Freelance yazılım geliştirici için hizmet sözleşmesi hazırla",
            ...     title="Freelance Hizmet Sözleşmesi",
            ...     model="claude-sonnet-4.5",
            ...     generation_params={"temperature": 0.7, "max_tokens": 4000}
            ... )
        """
        doc = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            title=title or f"Yeni {document_type.display_name_tr}",
            generation_type=GenerationType.AI_DRAFT,
            document_type=document_type,
            status=GenerationStatus.PENDING,
            prompt=prompt,
            model_used=model,
            generation_params=generation_params or {},
        )
        
        logger.info(
            "AI draft document created",
            document_id=str(doc.id),
            user_id=str(user_id),
            document_type=document_type.value,
        )
        
        return doc
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def start_generation(self) -> None:
        """Mark generation as started."""
        self.status = GenerationStatus.GENERATING
        self.started_at = datetime.now(timezone.utc)
        
        logger.info(
            "Document generation started",
            document_id=str(self.id),
        )
    
    def complete_generation(
        self,
        content: str,
        content_html: str | None = None,
        token_count: int | None = None,
        quality_score: float | None = None,
    ) -> None:
        """
        Complete generation successfully.
        
        Args:
            content: Generated content (text)
            content_html: Generated content (HTML)
            token_count: Tokens used
            quality_score: Quality score
        """
        self.status = GenerationStatus.GENERATED
        self.completed_at = datetime.now(timezone.utc)
        self.content = content
        self.content_html = content_html
        self.token_count = token_count
        self.quality_score = quality_score
        
        # Calculate generation time
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.generation_time_seconds = int(delta.total_seconds())
        
        # Set quality level
        if quality_score:
            self.quality_level = self._calculate_quality_level(quality_score)
        
        logger.info(
            "Document generation completed",
            document_id=str(self.id),
            token_count=token_count,
            quality_score=quality_score,
            generation_time_seconds=self.generation_time_seconds,
        )
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark generation as failed.
        
        Args:
            error_message: Error description
        """
        self.status = GenerationStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        
        logger.error(
            "Document generation failed",
            document_id=str(self.id),
            error=error_message,
        )
    
    @staticmethod
    def _calculate_quality_level(quality_score: float) -> QualityLevel:
        """
        Calculate quality level from score.
        
        Args:
            quality_score: Quality score (0.0-1.0)
            
        Returns:
            QualityLevel: Quality level enum
        """
        if quality_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif quality_score >= 0.7:
            return QualityLevel.GOOD
        elif quality_score >= 0.5:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    # =========================================================================
    # USER ACTIONS
    # =========================================================================
    
    def review(self) -> None:
        """Mark document as reviewed by user."""
        self.status = GenerationStatus.REVIEWED
        
        logger.info(
            "Document reviewed",
            document_id=str(self.id),
        )
    
    def approve(self) -> None:
        """Approve document for use."""
        self.is_approved = True
        self.approved_at = datetime.now(timezone.utc)
        self.status = GenerationStatus.APPROVED
        
        logger.info(
            "Document approved",
            document_id=str(self.id),
        )
    
    def add_user_feedback(self, rating: int, feedback: str | None = None) -> None:
        """
        Add user feedback.
        
        Args:
            rating: User rating (1-5)
            feedback: Feedback text
            
        Example:
            >>> doc.add_user_feedback(
            ...     rating=5,
            ...     feedback="Mükemmel bir sözleşme oluşturdu!"
            ... )
        """
        self.user_rating = rating
        self.user_feedback = feedback
        
        logger.info(
            "User feedback added to generated document",
            document_id=str(self.id),
            rating=rating,
        )
    
    def export(self, export_format: str) -> None:
        """
        Mark document as exported.
        
        Args:
            export_format: Export format (docx, pdf, txt, html)
        """
        self.export_format = export_format
        self.exported_at = datetime.now(timezone.utc)
        self.status = GenerationStatus.EXPORTED
        
        logger.info(
            "Document exported",
            document_id=str(self.id),
            format=export_format,
        )
    
    # =========================================================================
    # REVISION TRACKING
    # =========================================================================
    
    def add_revision(
        self,
        revision_content: str,
        revision_reason: str | None = None,
        revised_by: str | None = None,
    ) -> None:
        """
        Add a revision to history.
        
        Args:
            revision_content: New content
            revision_reason: Why revised
            revised_by: Who revised (user or AI)
        """
        revision = {
            "content": revision_content,
            "reason": revision_reason,
            "revised_by": revised_by,
            "revised_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if not isinstance(self.revisions, list):
            self.revisions = []
        
        self.revisions.append(revision)
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "revisions")
        
        logger.info(
            "Revision added to document",
            document_id=str(self.id),
            revision_count=len(self.revisions),
        )
    
    def get_revision_count(self) -> int:
        """Get number of revisions."""
        return len(self.revisions) if self.revisions else 0
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("title")
    def validate_title(self, key: str, title: str) -> str:
        """Validate title."""
        if not title or not title.strip():
            raise ValidationError(
                message="Title cannot be empty",
                field="title",
            )
        
        return title.strip()
    
    @validates("user_rating")
    def validate_user_rating(self, key: str, rating: int | None) -> int | None:
        """Validate user rating."""
        if rating is not None and not 1 <= rating <= 5:
            raise ValidationError(
                message="Rating must be between 1 and 5",
                field="user_rating",
            )
        
        return rating
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<GeneratedDocument("
            f"id={self.id}, "
            f"title={self.title}, "
            f"type={self.generation_type.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self, include_content: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_content: Include full content (can be large)
            
        Returns:
            dict: Document data
        """
        data = super().to_dict()
        
        # Remove large content by default
        if not include_content:
            data.pop("content", None)
            data.pop("content_html", None)
        
        # Add display names
        data["generation_type_display"] = self.generation_type.display_name_tr
        data["document_type_display"] = self.document_type.display_name_tr
        data["status_display"] = self.status.display_name_tr
        
        if self.quality_level:
            data["quality_level_display"] = self.quality_level.value.upper()
        
        # Add computed fields
        data["revision_count"] = self.get_revision_count()
        data["has_user_feedback"] = self.user_rating is not None
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "GeneratedDocument",
    "GenerationType",
    "GeneratedDocumentType",
    "GenerationStatus",
    "QualityLevel",
]
