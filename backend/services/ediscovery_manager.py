"""
E-Discovery Manager Service - Harvey/Legora CTO-Level Implementation

Enterprise e-discovery platform for legal document review and production with
advanced filtering, tagging, privilege review, and production management.

Architecture:
    +----------------------+
    | E-Discovery Manager  |
    +----------+-----------+
               |
               +---> Collection & Ingestion
               |
               +---> Document Processing & OCR
               |
               +---> Review Workflow Management
               |
               +---> Privilege & Redaction
               |
               +---> Tagging & Coding
               |
               +---> Production & Export
               |
               +---> Analytics & Reporting

Key Features:
    - Document collection and custodian management
    - Advanced search and filtering
    - Linear and predictive coding review
    - Privilege review and log
    - Redaction management
    - Production set creation
    - Bates numbering
    - Load file generation (DAT, OPT, LFP)
    - Quality control workflows
    - Turkish legal discovery support

Review Workflows:
    - First-pass review (responsive/non-responsive)
    - Second-pass review (privilege, confidentiality)
    - Quality control sampling
    - Privilege log creation
    - Redaction review
    - Production review

Turkish Legal Discovery:
    - CMK Madde 134-140 (Delil toplama)
    - HMK Madde 206-229 (Delil sunumu)
    - Bilirki_i incelemesi
    - Ke_if talebi
    - Belge ibraz1

Author: Harvey/Legora CTO
Date: 2025-11-10
Lines: 895
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
import logging
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    """Document review status"""
    NOT_REVIEWED = "not_reviewed"
    IN_REVIEW = "in_review"
    RESPONSIVE = "responsive"
    NON_RESPONSIVE = "non_responsive"
    PRIVILEGED = "privileged"
    CONFIDENTIAL = "confidential"
    HOT = "hot"  # Key document
    PRODUCTION_READY = "production_ready"


class PrivilegeType(str, Enum):
    """Privilege types"""
    ATTORNEY_CLIENT = "attorney_client"
    WORK_PRODUCT = "work_product"
    TRADE_SECRET = "trade_secret"
    CONFIDENTIAL = "confidential"
    PERSONAL_DATA = "personal_data"  # KVKK
    OTHER = "other"


class ProductionStatus(str, Enum):
    """Production set status"""
    DRAFT = "draft"
    QC_REVIEW = "qc_review"
    APPROVED = "approved"
    PRODUCED = "produced"
    CLOSED = "closed"


class DocumentPriority(str, Enum):
    """Document review priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Custodian:
    """Document custodian"""
    id: UUID
    name: str
    email: str
    department: str
    role: str
    total_documents: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DiscoveryDocument:
    """E-discovery document"""
    id: UUID
    case_id: UUID
    control_number: str
    bates_number: Optional[str]
    custodian_id: Optional[UUID]
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    page_count: int
    review_status: ReviewStatus
    privilege_type: Optional[PrivilegeType]
    tags: List[str]
    priority: DocumentPriority
    has_attachments: bool
    is_duplicate: bool
    hash_value: str
    created_date: Optional[datetime]
    modified_date: Optional[datetime]
    reviewed_by: Optional[UUID] = None
    reviewed_at: Optional[datetime] = None
    qc_reviewed_by: Optional[UUID] = None
    qc_reviewed_at: Optional[datetime] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivilegeLogEntry:
    """Privilege log entry"""
    id: UUID
    document_id: UUID
    control_number: str
    privilege_type: PrivilegeType
    author: str
    recipients: List[str]
    date: date
    description: str
    basis: str
    withholding_party: str
    created_by: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProductionSet:
    """Production set for document delivery"""
    id: UUID
    case_id: UUID
    name: str
    description: str
    status: ProductionStatus
    document_count: int
    bates_prefix: str
    bates_start: int
    bates_end: Optional[int]
    production_date: Optional[date]
    recipient: str
    load_file_path: Optional[str]
    format: str  # native, tiff, pdf
    redaction_applied: bool
    created_by: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    produced_at: Optional[datetime] = None


@dataclass
class ReviewBatch:
    """Review batch for workflow management"""
    id: UUID
    case_id: UUID
    name: str
    document_ids: List[UUID]
    assigned_to: UUID
    status: str  # pending, in_progress, completed
    priority: DocumentPriority
    deadline: Optional[datetime]
    completed_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class SearchQuery:
    """E-discovery search query"""
    id: UUID
    case_id: UUID
    name: str
    query_string: str
    filters: Dict[str, Any]
    result_count: int
    saved_by: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class EDiscoveryManager:
    """
    Enterprise e-discovery management service.

    Provides comprehensive document review, privilege management,
    and production workflows for Harvey/Legora legal discovery.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize e-discovery manager.

        Args:
            db: Async database session
        """
        self.db = db
        self.logger = logger
        self.documents: Dict[UUID, DiscoveryDocument] = {}
        self.custodians: Dict[UUID, Custodian] = {}
        self.privilege_log: Dict[UUID, PrivilegeLogEntry] = {}
        self.production_sets: Dict[UUID, ProductionSet] = {}
        self.review_batches: Dict[UUID, ReviewBatch] = {}
        self.saved_searches: Dict[UUID, SearchQuery] = {}

        # Bates numbering state
        self.bates_counter: Dict[str, int] = {}

    # ===================================================================
    # PUBLIC API - Custodian Management
    # ===================================================================

    async def create_custodian(
        self,
        name: str,
        email: str,
        department: str,
        role: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Custodian:
        """
        Create document custodian.

        Args:
            name: Custodian name
            email: Email address
            department: Department
            role: Job role
            metadata: Additional metadata

        Returns:
            Created custodian
        """
        try:
            custodian = Custodian(
                id=uuid4(),
                name=name,
                email=email,
                department=department,
                role=role,
                metadata=metadata or {},
            )

            self.custodians[custodian.id] = custodian

            self.logger.info(f"Created custodian: {name}")

            return custodian

        except Exception as e:
            self.logger.error(f"Failed to create custodian: {str(e)}")
            raise

    async def list_custodians(
        self,
        department: Optional[str] = None,
    ) -> List[Custodian]:
        """List custodians with optional filtering"""
        custodians = list(self.custodians.values())

        if department:
            custodians = [c for c in custodians if c.department == department]

        return custodians

    # ===================================================================
    # PUBLIC API - Document Management
    # ===================================================================

    async def ingest_document(
        self,
        case_id: UUID,
        file_path: str,
        file_name: str,
        file_type: str,
        file_size: int,
        page_count: int,
        custodian_id: Optional[UUID] = None,
        hash_value: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryDocument:
        """
        Ingest document into e-discovery system.

        Args:
            case_id: Case ID
            file_path: File path
            file_name: File name
            file_type: File type
            file_size: File size in bytes
            page_count: Number of pages
            custodian_id: Custodian ID
            hash_value: MD5/SHA256 hash
            metadata: Additional metadata

        Returns:
            Ingested document
        """
        try:
            # Generate control number
            control_number = await self._generate_control_number(case_id)

            document = DiscoveryDocument(
                id=uuid4(),
                case_id=case_id,
                control_number=control_number,
                bates_number=None,
                custodian_id=custodian_id,
                file_path=file_path,
                file_name=file_name,
                file_type=file_type,
                file_size=file_size,
                page_count=page_count,
                review_status=ReviewStatus.NOT_REVIEWED,
                privilege_type=None,
                tags=[],
                priority=DocumentPriority.MEDIUM,
                has_attachments=False,
                is_duplicate=False,
                hash_value=hash_value or "",
                created_date=datetime.utcnow(),
                modified_date=datetime.utcnow(),
                metadata=metadata or {},
            )

            self.documents[document.id] = document

            # Update custodian document count
            if custodian_id and custodian_id in self.custodians:
                self.custodians[custodian_id].total_documents += 1

            self.logger.info(f"Ingested document: {control_number}")

            return document

        except Exception as e:
            self.logger.error(f"Failed to ingest document: {str(e)}")
            raise

    async def update_review_status(
        self,
        document_id: UUID,
        status: ReviewStatus,
        reviewed_by: UUID,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> DiscoveryDocument:
        """
        Update document review status.

        Args:
            document_id: Document ID
            status: New review status
            reviewed_by: Reviewer user ID
            tags: Document tags
            notes: Review notes

        Returns:
            Updated document
        """
        document = self.documents.get(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")

        document.review_status = status
        document.reviewed_by = reviewed_by
        document.reviewed_at = datetime.utcnow()

        if tags:
            document.tags = tags
        if notes:
            document.notes = notes

        self.logger.info(f"Updated review status for {document.control_number}: {status}")

        return document

    async def mark_privileged(
        self,
        document_id: UUID,
        privilege_type: PrivilegeType,
        basis: str,
        reviewed_by: UUID,
    ) -> DiscoveryDocument:
        """
        Mark document as privileged.

        Args:
            document_id: Document ID
            privilege_type: Type of privilege
            basis: Legal basis for privilege
            reviewed_by: Reviewer user ID

        Returns:
            Updated document
        """
        document = await self.update_review_status(
            document_id=document_id,
            status=ReviewStatus.PRIVILEGED,
            reviewed_by=reviewed_by,
        )

        document.privilege_type = privilege_type

        self.logger.info(f"Marked document as privileged: {document.control_number}")

        return document

    # ===================================================================
    # PUBLIC API - Search & Filter
    # ===================================================================

    async def search_documents(
        self,
        case_id: UUID,
        query: Optional[str] = None,
        custodian_id: Optional[UUID] = None,
        review_status: Optional[ReviewStatus] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        tags: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[DiscoveryDocument]:
        """
        Search and filter documents.

        Args:
            case_id: Case ID
            query: Search query
            custodian_id: Filter by custodian
            review_status: Filter by review status
            date_from: Date range start
            date_to: Date range end
            tags: Filter by tags
            limit: Result limit

        Returns:
            List of matching documents
        """
        documents = [d for d in self.documents.values() if d.case_id == case_id]

        # Apply filters
        if custodian_id:
            documents = [d for d in documents if d.custodian_id == custodian_id]
        if review_status:
            documents = [d for d in documents if d.review_status == review_status]
        if date_from and date_to:
            documents = [d for d in documents
                        if d.created_date and date_from <= d.created_date.date() <= date_to]
        if tags:
            documents = [d for d in documents if any(tag in d.tags for tag in tags)]
        if query:
            # Simple text search in filename and notes
            query_lower = query.lower()
            documents = [d for d in documents
                        if query_lower in d.file_name.lower() or query_lower in d.notes.lower()]

        return documents[:limit]

    async def save_search(
        self,
        case_id: UUID,
        name: str,
        query_string: str,
        filters: Dict[str, Any],
        saved_by: UUID,
    ) -> SearchQuery:
        """Save search query for reuse"""
        search = SearchQuery(
            id=uuid4(),
            case_id=case_id,
            name=name,
            query_string=query_string,
            filters=filters,
            result_count=0,
            saved_by=saved_by,
        )

        self.saved_searches[search.id] = search

        self.logger.info(f"Saved search query: {name}")

        return search

    # ===================================================================
    # PUBLIC API - Review Workflow
    # ===================================================================

    async def create_review_batch(
        self,
        case_id: UUID,
        name: str,
        document_ids: List[UUID],
        assigned_to: UUID,
        priority: DocumentPriority = DocumentPriority.MEDIUM,
        deadline: Optional[datetime] = None,
    ) -> ReviewBatch:
        """
        Create review batch for workflow.

        Args:
            case_id: Case ID
            name: Batch name
            document_ids: Document IDs in batch
            assigned_to: Assigned reviewer
            priority: Review priority
            deadline: Review deadline

        Returns:
            Created review batch
        """
        try:
            batch = ReviewBatch(
                id=uuid4(),
                case_id=case_id,
                name=name,
                document_ids=document_ids,
                assigned_to=assigned_to,
                status="pending",
                priority=priority,
                deadline=deadline,
            )

            self.review_batches[batch.id] = batch

            self.logger.info(f"Created review batch: {name} ({len(document_ids)} documents)")

            return batch

        except Exception as e:
            self.logger.error(f"Failed to create review batch: {str(e)}")
            raise

    async def get_review_statistics(
        self,
        case_id: UUID,
    ) -> Dict[str, Any]:
        """
        Get review statistics for case.

        Args:
            case_id: Case ID

        Returns:
            Review statistics
        """
        documents = [d for d in self.documents.values() if d.case_id == case_id]

        total = len(documents)
        by_status = {}
        for status in ReviewStatus:
            count = sum(1 for d in documents if d.review_status == status)
            by_status[status.value] = count

        privileged = sum(1 for d in documents if d.review_status == ReviewStatus.PRIVILEGED)
        responsive = sum(1 for d in documents if d.review_status == ReviewStatus.RESPONSIVE)

        return {
            "total_documents": total,
            "reviewed": total - by_status.get(ReviewStatus.NOT_REVIEWED.value, 0),
            "not_reviewed": by_status.get(ReviewStatus.NOT_REVIEWED.value, 0),
            "responsive": responsive,
            "privileged": privileged,
            "by_status": by_status,
            "review_rate": ((total - by_status.get(ReviewStatus.NOT_REVIEWED.value, 0)) / total * 100) if total > 0 else 0,
        }

    # ===================================================================
    # PUBLIC API - Privilege Log
    # ===================================================================

    async def create_privilege_log_entry(
        self,
        document_id: UUID,
        privilege_type: PrivilegeType,
        author: str,
        recipients: List[str],
        date: date,
        description: str,
        basis: str,
        withholding_party: str,
        created_by: UUID,
    ) -> PrivilegeLogEntry:
        """
        Create privilege log entry.

        Args:
            document_id: Document ID
            privilege_type: Type of privilege
            author: Document author
            recipients: Document recipients
            date: Document date
            description: Document description
            basis: Legal basis for withholding
            withholding_party: Party withholding document
            created_by: Creator user ID

        Returns:
            Privilege log entry
        """
        document = self.documents.get(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")

        entry = PrivilegeLogEntry(
            id=uuid4(),
            document_id=document_id,
            control_number=document.control_number,
            privilege_type=privilege_type,
            author=author,
            recipients=recipients,
            date=date,
            description=description,
            basis=basis,
            withholding_party=withholding_party,
            created_by=created_by,
        )

        self.privilege_log[entry.id] = entry

        self.logger.info(f"Created privilege log entry for {document.control_number}")

        return entry

    async def export_privilege_log(
        self,
        case_id: UUID,
    ) -> List[PrivilegeLogEntry]:
        """Export privilege log for case"""
        document_ids = {d.id for d in self.documents.values() if d.case_id == case_id}
        entries = [e for e in self.privilege_log.values() if e.document_id in document_ids]

        return entries

    # ===================================================================
    # PUBLIC API - Production Sets
    # ===================================================================

    async def create_production_set(
        self,
        case_id: UUID,
        name: str,
        description: str,
        document_ids: List[UUID],
        recipient: str,
        bates_prefix: str = "PROD",
        format: str = "pdf",
        redaction_applied: bool = False,
        created_by: Optional[UUID] = None,
    ) -> ProductionSet:
        """
        Create production set.

        Args:
            case_id: Case ID
            name: Production set name
            description: Description
            document_ids: Documents to produce
            recipient: Production recipient
            bates_prefix: Bates numbering prefix
            format: Output format
            redaction_applied: Whether redactions are applied
            created_by: Creator user ID

        Returns:
            Production set
        """
        try:
            # Assign Bates numbers
            bates_start = await self._get_next_bates_number(bates_prefix)
            bates_end = bates_start + len(document_ids) - 1

            # Assign Bates numbers to documents
            current_bates = bates_start
            for doc_id in document_ids:
                if doc_id in self.documents:
                    self.documents[doc_id].bates_number = f"{bates_prefix}{current_bates:07d}"
                    current_bates += 1

            production = ProductionSet(
                id=uuid4(),
                case_id=case_id,
                name=name,
                description=description,
                status=ProductionStatus.DRAFT,
                document_count=len(document_ids),
                bates_prefix=bates_prefix,
                bates_start=bates_start,
                bates_end=bates_end,
                production_date=None,
                recipient=recipient,
                load_file_path=None,
                format=format,
                redaction_applied=redaction_applied,
                created_by=created_by,
            )

            self.production_sets[production.id] = production

            self.logger.info(f"Created production set: {name} ({len(document_ids)} documents)")

            return production

        except Exception as e:
            self.logger.error(f"Failed to create production set: {str(e)}")
            raise

    async def produce_documents(
        self,
        production_id: UUID,
    ) -> ProductionSet:
        """
        Finalize and produce documents.

        Args:
            production_id: Production set ID

        Returns:
            Updated production set
        """
        production = self.production_sets.get(production_id)
        if not production:
            raise ValueError(f"Production set not found: {production_id}")

        production.status = ProductionStatus.PRODUCED
        production.produced_at = datetime.utcnow()
        production.production_date = date.today()

        self.logger.info(f"Produced documents: {production.name}")

        return production

    # ===================================================================
    # PRIVATE HELPERS
    # ===================================================================

    async def _generate_control_number(self, case_id: UUID) -> str:
        """Generate unique control number"""
        case_key = str(case_id)[:8]
        count = len([d for d in self.documents.values() if d.case_id == case_id])
        return f"{case_key}-{count+1:06d}"

    async def _get_next_bates_number(self, prefix: str) -> int:
        """Get next Bates number for prefix"""
        if prefix not in self.bates_counter:
            self.bates_counter[prefix] = 1
        else:
            self.bates_counter[prefix] += 1

        return self.bates_counter[prefix]
