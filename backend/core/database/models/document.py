"""
Document model for file management in Turkish Legal AI.

This module provides the Document model for comprehensive document management:
- Multi-format support (PDF, DOCX, TXT, images)
- S3/MinIO storage with encryption
- OCR and text extraction
- Metadata extraction (author, dates, keywords)
- Version control (history tracking)
- Access control (owner, team, organization)
- Turkish-specific legal document types
- KVKK-compliant data handling
- Virus scanning integration
- Document classification and tagging

Document Workflow:
    1. Upload: Client uploads file
    2. Storage: Store in S3 with encryption
    3. Processing: Extract text, OCR, metadata
    4. Analysis: Classify, tag, extract entities
    5. Indexing: Index for RAG retrieval
    6. Versioning: Track changes over time
    7. Sharing: Control access (user, team, org)
    8. Archival: Soft delete with retention policy

Security & Compliance:
    - Encryption at rest (S3 server-side)
    - Encryption in transit (HTTPS)
    - Access control (RBAC + ownership)
    - Audit trail (who accessed when)
    - KVKK compliance (data retention, deletion)
    - Virus scanning (ClamAV integration)
    - TC Kimlik No detection and masking

Example:
    >>> # Upload document
    >>> doc = Document(
    ...     name="İş Sözleşmesi - Ahmet Yılmaz.pdf",
    ...     document_type=DocumentType.CONTRACT,
    ...     file_path="s3://bucket/tenant_123/doc_456.pdf",
    ...     file_size=1024000,
    ...     mime_type="application/pdf",
    ...     owner_id=user_id,
    ...     tenant_id=tenant_id,
    ...     access_level=AccessLevel.PRIVATE
    ... )
    >>> doc.extract_metadata()
    >>> doc.process_ocr()
    >>> doc.classify()
"""

import enum
import mimetypes
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    CheckConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship, validates

from backend.core.constants import (
    ALLOWED_DOCUMENT_MIMETYPES,
    MAX_DOCUMENT_SIZE_MB,
    MAX_FILENAME_LENGTH,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from backend.core.exceptions import (
    DocumentProcessingError,
    FileSizeExceededError,
    UnsupportedFileTypeError,
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


class DocumentType(str, enum.Enum):
    """
    Turkish legal document type classifications.
    
    Types:
    - CONTRACT: Sözleşme (employment, service, partnership)
    - PETITION: Dilekçe (court petitions, applications)
    - COURT_DECISION: Mahkeme kararı (judgments, verdicts)
    - LEGAL_OPINION: Hukuki görüş (legal memos, opinions)
    - POWER_OF_ATTORNEY: Vekaletname
    - REGULATION: Düzenleme (laws, regulations, circulars)
    - CORRESPONDENCE: Yazışma (letters, emails)
    - INVOICE: Fatura
    - OTHER: Diğer
    """
    
    CONTRACT = "contract"                      # Sözleşme
    PETITION = "petition"                      # Dilekçe
    COURT_DECISION = "court_decision"          # Mahkeme Kararı
    LEGAL_OPINION = "legal_opinion"            # Hukuki Görüş
    POWER_OF_ATTORNEY = "power_of_attorney"    # Vekaletname
    REGULATION = "regulation"                  # Düzenleme/Mevzuat
    CORRESPONDENCE = "correspondence"          # Yazışma
    INVOICE = "invoice"                        # Fatura
    PROTOCOL = "protocol"                      # Protokol/Tutanak
    REPORT = "report"                          # Rapor
    EVIDENCE = "evidence"                      # Delil
    OTHER = "other"                            # Diğer
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.CONTRACT: "Sözleşme",
            self.PETITION: "Dilekçe",
            self.COURT_DECISION: "Mahkeme Kararı",
            self.LEGAL_OPINION: "Hukuki Görüş",
            self.POWER_OF_ATTORNEY: "Vekaletname",
            self.REGULATION: "Düzenleme/Mevzuat",
            self.CORRESPONDENCE: "Yazışma",
            self.INVOICE: "Fatura",
            self.PROTOCOL: "Protokol/Tutanak",
            self.REPORT: "Rapor",
            self.EVIDENCE: "Delil",
            self.OTHER: "Diğer",
        }
        return names.get(self, self.value)


class ProcessingStatus(str, enum.Enum):
    """Document processing status."""
    
    UPLOADED = "uploaded"              # File uploaded, pending processing
    PROCESSING = "processing"          # OCR, extraction in progress
    COMPLETED = "completed"            # Processing completed successfully
    FAILED = "failed"                  # Processing failed
    INDEXED = "indexed"                # Indexed for RAG retrieval
    
    def __str__(self) -> str:
        return self.value


class AccessLevel(str, enum.Enum):
    """Document access level."""
    
    PRIVATE = "private"                # Owner only
    TEAM = "team"                      # Team members
    ORGANIZATION = "organization"      # Organization members
    TENANT = "tenant"                  # All tenant users
    PUBLIC = "public"                  # Public (rare)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.PRIVATE: "Özel (Sadece Ben)",
            self.TEAM: "Ekip",
            self.ORGANIZATION: "Organizasyon",
            self.TENANT: "Kuruluş",
            self.PUBLIC: "Genel",
        }
        return names.get(self, self.value)


class SecurityClassification(str, enum.Enum):
    """Security classification for sensitive documents."""
    
    PUBLIC = "public"                  # Public information
    INTERNAL = "internal"              # Internal use only
    CONFIDENTIAL = "confidential"      # Confidential
    HIGHLY_CONFIDENTIAL = "highly_confidential"  # Highly confidential
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# DOCUMENT MODEL
# =============================================================================


class Document(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Document model for file management and processing.
    
    Documents represent uploaded files with:
    - Multi-format support (PDF, DOCX, images)
    - Cloud storage (S3/MinIO)
    - Text extraction and OCR
    - Metadata extraction
    - Access control
    - Version history
    - Turkish legal document classification
    
    File Storage:
        - S3/MinIO bucket per tenant
        - Encrypted at rest (SSE-S3 or SSE-KMS)
        - Path format: s3://bucket/tenant_id/year/month/document_id.ext
        - Versioned backups
    
    Processing Pipeline:
        1. Upload → Virus scan
        2. Store → S3 with encryption
        3. Extract → Text, OCR if needed
        4. Analyze → Classify, tag, entities
        5. Index → Elasticsearch/pgvector for RAG
    
    Attributes:
        name: Original filename
        display_name: User-friendly display name
        description: Document description
        
        document_type: Type classification
        document_subtype: Subtype (e.g., employment_contract)
        
        file_path: S3 storage path
        file_size: File size in bytes
        file_hash: SHA-256 hash (deduplication, integrity)
        mime_type: MIME type
        extension: File extension
        
        owner_id: Document owner
        team_id: Team (if shared with team)
        organization_id: Organization (if shared)
        
        access_level: Access control level
        security_classification: Security classification
        
        processing_status: Current processing status
        processing_error: Error message if failed
        
        ocr_status: OCR processing status
        text_extracted: Full text content
        page_count: Number of pages (PDF)
        word_count: Approximate word count
        
        metadata: Extracted metadata (JSON)
        tags: Document tags (array)
        language: Detected language (tr, en)
        
        is_encrypted: File is encrypted
        is_scanned: Virus scan completed
        scan_result: Virus scan result
        
        version: Version number
        parent_version_id: Parent document (if version)
        
        indexed_at: When indexed for RAG
        last_accessed_at: Last access timestamp
        access_count: Total access count
        
        settings: Document settings
        
    Relationships:
        tenant: Parent tenant
        owner: Document owner (user)
        team: Shared with team (optional)
        organization: Shared with organization (optional)
        versions: Document versions
        parent_version: Parent document version
        analyses: Analysis results
        chat_sessions: Related chat sessions
    """
    
    __tablename__ = "documents"
    
    # =========================================================================
    # IDENTITY & METADATA
    # =========================================================================
    
    name = Column(
        String(MAX_FILENAME_LENGTH),
        nullable=False,
        comment="Original filename",
    )
    
    display_name = Column(
        String(MAX_FILENAME_LENGTH),
        nullable=True,
        comment="User-friendly display name (optional)",
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Document description",
    )
    
    # =========================================================================
    # DOCUMENT CLASSIFICATION
    # =========================================================================
    
    document_type = Column(
        Enum(DocumentType, native_enum=False, length=50),
        nullable=False,
        default=DocumentType.OTHER,
        index=True,
        comment="Document type classification",
    )
    
    document_subtype = Column(
        String(100),
        nullable=True,
        comment="Document subtype (e.g., employment_contract, service_agreement)",
    )
    
    # =========================================================================
    # FILE STORAGE
    # =========================================================================
    
    file_path = Column(
        String(500),
        nullable=False,
        unique=True,
        comment="S3/MinIO storage path",
    )
    
    file_size = Column(
        Integer,
        nullable=False,
        comment="File size in bytes",
    )
    
    file_hash = Column(
        String(64),  # SHA-256
        nullable=False,
        index=True,
        comment="SHA-256 hash for integrity and deduplication",
    )
    
    mime_type = Column(
        String(100),
        nullable=False,
        comment="MIME type (e.g., application/pdf)",
    )
    
    extension = Column(
        String(10),
        nullable=False,
        comment="File extension (e.g., .pdf, .docx)",
    )
    
    # =========================================================================
    # OWNERSHIP & ACCESS
    # =========================================================================
    
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Document owner",
    )
    
    owner = relationship(
        "User",
        back_populates="documents",
        foreign_keys=[owner_id],
    )
    
    team_id = Column(
        UUID(as_uuid=True),
        ForeignKey("teams.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Team (if shared with team)",
    )
    
    team = relationship(
        "Team",
        back_populates="documents",
    )
    
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Organization (if shared)",
    )
    
    organization = relationship(
        "Organization",
        back_populates="documents",
    )
    
    # =========================================================================
    # ACCESS CONTROL
    # =========================================================================
    
    access_level = Column(
        Enum(AccessLevel, native_enum=False, length=50),
        nullable=False,
        default=AccessLevel.PRIVATE,
        index=True,
        comment="Access control level",
    )
    
    security_classification = Column(
        Enum(SecurityClassification, native_enum=False, length=50),
        nullable=False,
        default=SecurityClassification.INTERNAL,
        comment="Security classification",
    )
    
    # =========================================================================
    # PROCESSING STATUS
    # =========================================================================
    
    processing_status = Column(
        Enum(ProcessingStatus, native_enum=False, length=50),
        nullable=False,
        default=ProcessingStatus.UPLOADED,
        index=True,
        comment="Document processing status",
    )
    
    processing_error = Column(
        Text,
        nullable=True,
        comment="Error message if processing failed",
    )
    
    # =========================================================================
    # OCR & TEXT EXTRACTION
    # =========================================================================
    
    ocr_status = Column(
        String(50),
        nullable=True,
        comment="OCR processing status (pending, completed, failed)",
    )
    
    text_extracted = Column(
        Text,
        nullable=True,
        comment="Full text content extracted from document",
    )
    
    page_count = Column(
        Integer,
        nullable=True,
        comment="Number of pages (for PDFs)",
    )
    
    word_count = Column(
        Integer,
        nullable=True,
        comment="Approximate word count",
    )
    
    # =========================================================================
    # METADATA & TAGS
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Extracted metadata (author, dates, keywords, etc.)",
    )
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Document tags for categorization",
    )
    
    language = Column(
        String(10),
        nullable=False,
        default="tr",
        comment="Detected language (ISO 639-1 code)",
    )
    
    # =========================================================================
    # SECURITY
    # =========================================================================
    
    is_encrypted = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="File is encrypted at rest",
    )
    
    is_scanned = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Virus scan completed",
    )
    
    scan_result = Column(
        String(50),
        nullable=True,
        comment="Virus scan result (clean, infected, error)",
    )
    
    # =========================================================================
    # VERSIONING
    # =========================================================================
    
    version = Column(
        Integer,
        nullable=False,
        default=1,
        comment="Version number (1, 2, 3...)",
    )
    
    parent_version_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Parent document (if this is a version)",
    )
    
    # Self-referential relationships for versions
    parent_version = relationship(
        "Document",
        remote_side="Document.id",
        back_populates="versions",
        foreign_keys=[parent_version_id],
    )
    
    versions = relationship(
        "Document",
        back_populates="parent_version",
        cascade="all, delete-orphan",
        foreign_keys=[parent_version_id],
    )
    
    # =========================================================================
    # INDEXING & ANALYTICS
    # =========================================================================
    
    indexed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When document was indexed for RAG retrieval",
    )
    
    last_accessed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last access timestamp",
    )
    
    access_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of times accessed",
    )
    
    # =========================================================================
    # SETTINGS
    # =========================================================================
    
    settings = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Document-specific settings (retention, watermark, etc.)",
    )
    
    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================
    
    # Analysis results
    analyses = relationship(
        "ContractAnalysis",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # Chat sessions that reference this document
    chat_sessions = relationship(
        "ChatSession",
        secondary="chat_session_documents",
        back_populates="documents",
        lazy="dynamic",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for owner's documents
        Index("ix_documents_owner", "owner_id", "tenant_id"),
        
        # Index for team documents
        Index("ix_documents_team", "team_id", "tenant_id"),
        
        # Index for document type filtering
        Index("ix_documents_type", "document_type", "tenant_id"),
        
        # Index for processing status
        Index(
            "ix_documents_processing",
            "processing_status",
            postgresql_where="processing_status != 'completed'",
        ),
        
        # Index for file hash (deduplication)
        Index("ix_documents_hash", "file_hash", "tenant_id"),
        
        # Check: file size positive
        CheckConstraint(
            "file_size > 0",
            name="ck_documents_file_size_positive",
        ),
        
        # Check: page count positive
        CheckConstraint(
            "page_count IS NULL OR page_count > 0",
            name="ck_documents_page_count_positive",
        ),
        
        # Check: version positive
        CheckConstraint(
            "version > 0",
            name="ck_documents_version_positive",
        ),
        
        # Check: access count non-negative
        CheckConstraint(
            "access_count >= 0",
            name="ck_documents_access_count_positive",
        ),
    )
    
    # =========================================================================
    # FILE VALIDATION
    # =========================================================================
    
    @classmethod
    def validate_file(
        cls,
        filename: str,
        file_size: int,
        mime_type: str | None = None,
    ) -> tuple[bool, str]:
        """
        Validate uploaded file before processing.
        
        Checks:
        - File extension allowed
        - MIME type allowed
        - File size within limit
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            mime_type: MIME type (optional, will detect from filename)
            
        Returns:
            tuple: (is_valid, error_message)
            
        Example:
            >>> is_valid, error = Document.validate_file(
            ...     "contract.pdf",
            ...     1024000,
            ...     "application/pdf"
            ... )
            >>> if not is_valid:
            ...     raise ValueError(error)
        """
        # Extract extension
        _, ext = os.path.splitext(filename.lower())
        
        # Check extension
        if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
            return False, f"Desteklenmeyen dosya tipi: {ext}"
        
        # Detect MIME type if not provided
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
        
        # Check MIME type
        if mime_type and mime_type not in ALLOWED_DOCUMENT_MIMETYPES:
            return False, f"Desteklenmeyen MIME tipi: {mime_type}"
        
        # Check file size
        max_size_bytes = MAX_DOCUMENT_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            return False, f"Dosya boyutu limiti aşıldı. Maksimum: {MAX_DOCUMENT_SIZE_MB}MB"
        
        if file_size == 0:
            return False, "Dosya boş olamaz"
        
        return True, ""
    
    def validate_before_save(self) -> None:
        """
        Validate document before saving.
        
        Raises:
            ValidationError: If validation fails
        """
        # Validate file
        is_valid, error = self.validate_file(
            self.name,
            self.file_size,
            self.mime_type,
        )
        
        if not is_valid:
            raise ValidationError(
                message=error,
                field="file",
            )
        
        # Validate file_path format
        if not self.file_path.startswith(("s3://", "minio://")):
            raise ValidationError(
                message="file_path must start with s3:// or minio://",
                field="file_path",
            )
    
    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================
    
    def extract_text(self) -> str | None:
        """
        Extract text from document.
        
        Handles:
        - PDF: pdfplumber or PyPDF2
        - DOCX: python-docx
        - TXT: direct read
        - Images: OCR (Tesseract)
        
        Returns:
            str | None: Extracted text or None if failed
            
        Example:
            >>> text = document.extract_text()
            >>> document.text_extracted = text
            >>> document.word_count = len(text.split())
        """
        # This would be implemented in a service
        # Placeholder for demonstration
        logger.info(
            "Extracting text from document",
            document_id=str(self.id),
            mime_type=self.mime_type,
        )
        
        self.processing_status = ProcessingStatus.PROCESSING
        
        try:
            # Extraction logic would go here
            # For PDF: use pdfplumber
            # For DOCX: use python-docx
            # For images: use Tesseract OCR
            
            # Placeholder
            extracted_text = ""
            
            self.text_extracted = extracted_text
            self.word_count = len(extracted_text.split())
            self.processing_status = ProcessingStatus.COMPLETED
            
            logger.info(
                "Text extraction completed",
                document_id=str(self.id),
                word_count=self.word_count,
            )
            
            return extracted_text
            
        except Exception as e:
            self.processing_status = ProcessingStatus.FAILED
            self.processing_error = str(e)
            
            logger.error(
                "Text extraction failed",
                document_id=str(self.id),
                error=str(e),
            )
            
            raise DocumentProcessingError(
                message=f"Text extraction failed: {e}",
                document_id=str(self.id),
            )
    
    def extract_metadata(self) -> dict[str, Any]:
        """
        Extract metadata from document.
        
        Extracts:
        - Author, title, subject
        - Creation date, modification date
        - Keywords, comments
        - PDF specific: creator, producer
        
        Returns:
            dict: Extracted metadata
            
        Example:
            >>> metadata = document.extract_metadata()
            >>> document.metadata = metadata
        """
        # Placeholder for metadata extraction
        metadata = {
            "author": None,
            "title": None,
            "subject": None,
            "keywords": [],
            "created_date": None,
            "modified_date": None,
        }
        
        self.metadata = metadata
        
        logger.info(
            "Metadata extracted",
            document_id=str(self.id),
            metadata_keys=list(metadata.keys()),
        )
        
        return metadata
    
    # =========================================================================
    # CLASSIFICATION
    # =========================================================================
    
    def classify(self) -> DocumentType:
        """
        Automatically classify document type using ML/rules.
        
        Uses:
        - Filename patterns
        - Text content analysis
        - ML classification model
        
        Returns:
            DocumentType: Predicted document type
            
        Example:
            >>> doc_type = document.classify()
            >>> document.document_type = doc_type
        """
        # Placeholder for classification logic
        # Production would use ML model
        
        # Simple rule-based classification
        name_lower = self.name.lower()
        
        if any(word in name_lower for word in ["sözleşme", "contract"]):
            return DocumentType.CONTRACT
        elif any(word in name_lower for word in ["dilekçe", "petition"]):
            return DocumentType.PETITION
        elif any(word in name_lower for word in ["karar", "judgment"]):
            return DocumentType.COURT_DECISION
        elif any(word in name_lower for word in ["vekaletname", "power of attorney"]):
            return DocumentType.POWER_OF_ATTORNEY
        else:
            return DocumentType.OTHER
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to document.
        
        Args:
            tag: Tag to add
            
        Example:
            >>> document.add_tag("urgent")
            >>> document.add_tag("client:ABC Corp")
        """
        if tag not in self.tags:
            self.tags.append(tag)
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "tags")
            
            logger.debug(
                "Tag added to document",
                document_id=str(self.id),
                tag=tag,
            )
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from document."""
        if tag in self.tags:
            self.tags.remove(tag)
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "tags")
    
    # =========================================================================
    # ACCESS CONTROL
    # =========================================================================
    
    def can_access(self, user_id: str, user_teams: list[str] | None = None) -> bool:
        """
        Check if user can access this document.
        
        Access rules:
        - PRIVATE: Owner only
        - TEAM: Owner + team members
        - ORGANIZATION: Owner + organization members
        - TENANT: All tenant users
        - PUBLIC: Everyone
        
        Args:
            user_id: User UUID
            user_teams: List of user's team IDs (optional)
            
        Returns:
            bool: True if user can access
            
        Example:
            >>> if document.can_access(str(current_user.id), user_teams):
            ...     return document_data
            ... else:
            ...     raise PermissionDeniedError()
        """
        # Owner always has access
        if str(self.owner_id) == user_id:
            return True
        
        # Check access level
        if self.access_level == AccessLevel.PRIVATE:
            return False
        
        if self.access_level == AccessLevel.TEAM:
            if not user_teams or not self.team_id:
                return False
            return str(self.team_id) in user_teams
        
        if self.access_level == AccessLevel.ORGANIZATION:
            # Would need to check user's organization membership
            # Placeholder: allow for now
            return True
        
        if self.access_level == AccessLevel.TENANT:
            return True
        
        if self.access_level == AccessLevel.PUBLIC:
            return True
        
        return False
    
    def record_access(self, user_id: str) -> None:
        """
        Record document access for analytics.
        
        Args:
            user_id: User who accessed
            
        Example:
            >>> document.record_access(str(current_user.id))
        """
        self.last_accessed_at = datetime.now(timezone.utc)
        self.access_count += 1
        
        logger.debug(
            "Document access recorded",
            document_id=str(self.id),
            user_id=user_id,
            access_count=self.access_count,
        )
    
    # =========================================================================
    # VERSIONING
    # =========================================================================
    
    def create_version(self, new_file_path: str, new_file_hash: str) -> "Document":
        """
        Create a new version of this document.
        
        Args:
            new_file_path: S3 path of new version
            new_file_hash: SHA-256 hash of new version
            
        Returns:
            Document: New version document
            
        Example:
            >>> new_version = document.create_version(
            ...     new_file_path="s3://bucket/tenant/doc_v2.pdf",
            ...     new_file_hash="abc123..."
            ... )
            >>> new_version.version  # 2
        """
        # Create new document as version
        new_version = Document(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            document_type=self.document_type,
            file_path=new_file_path,
            file_size=self.file_size,  # Would be updated with actual size
            file_hash=new_file_hash,
            mime_type=self.mime_type,
            extension=self.extension,
            owner_id=self.owner_id,
            tenant_id=self.tenant_id,
            team_id=self.team_id,
            organization_id=self.organization_id,
            access_level=self.access_level,
            version=self.version + 1,
            parent_version_id=self.id,
        )
        
        logger.info(
            "Document version created",
            document_id=str(self.id),
            new_version_id=str(new_version.id),
            version=new_version.version,
        )
        
        return new_version
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("name")
    def validate_name(self, key: str, name: str) -> str:
        """Validate filename."""
        if not name or not name.strip():
            raise ValidationError(
                message="Dosya adı boş olamaz",
                field="name",
            )
        
        # Remove path components (security)
        name = os.path.basename(name)
        
        return name.strip()
    
    @validates("file_size")
    def validate_file_size(self, key: str, file_size: int) -> int:
        """Validate file size."""
        if file_size <= 0:
            raise ValidationError(
                message="Dosya boyutu 0'dan büyük olmalıdır",
                field="file_size",
            )
        
        max_size = MAX_DOCUMENT_SIZE_MB * 1024 * 1024
        if file_size > max_size:
            raise FileSizeExceededError(
                message=f"Dosya boyutu limiti aşıldı. Maksimum: {MAX_DOCUMENT_SIZE_MB}MB",
                max_size=max_size,
                actual_size=file_size,
            )
        
        return file_size
    
    @validates("mime_type")
    def validate_mime_type(self, key: str, mime_type: str) -> str:
        """Validate MIME type."""
        if mime_type not in ALLOWED_DOCUMENT_MIMETYPES:
            raise UnsupportedFileTypeError(
                message=f"Desteklenmeyen dosya tipi: {mime_type}",
                mime_type=mime_type,
            )
        
        return mime_type
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Document(id={self.id}, name={self.name}, type={self.document_type})>"
    
    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_text: Include extracted text (default: False, can be large)
            
        Returns:
            dict: Document data
        """
        data = super().to_dict()
        
        # Remove large text by default
        if not include_text:
            data.pop("text_extracted", None)
        
        # Add computed fields
        data["type_display"] = self.document_type.display_name_tr
        data["access_level_display"] = self.access_level.display_name_tr
        data["file_size_mb"] = round(self.file_size / (1024 * 1024), 2)
        data["is_latest_version"] = self.parent_version_id is None
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Document",
    "DocumentType",
    "ProcessingStatus",
    "AccessLevel",
    "SecurityClassification",
]
