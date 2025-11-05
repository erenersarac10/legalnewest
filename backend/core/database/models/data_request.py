"""
Data Request model for KVKK data subject rights management in Turkish Legal AI.

This module provides the DataRequest model for handling KVKK data requests:
- Right to access (Article 11)
- Right to correction (Article 11)
- Right to deletion (Article 7, 17)
- Right to objection (Article 11)
- Data portability
- Request tracking and fulfillment
- Legal compliance (response within 30 days)
- Audit trail

KVKK Data Subject Rights (Article 11):
    a) Right to know if personal data is processed
    b) Right to information about processing
    c) Right to know processing purpose
    d) Right to know third parties receiving data
    e) Right to correction if incomplete/inaccurate
    f) Right to deletion/destruction (Article 7 conditions)
    g) Right to notification of correction/deletion
    h) Right to objection to automated processing
    i) Right to compensation for damages

Request Types:
    - ACCESS: Request copy of personal data
    - CORRECTION: Request correction of inaccurate data
    - DELETION: Request deletion of personal data
    - OBJECTION: Object to data processing
    - PORTABILITY: Request data in portable format
    - INFORMATION: Request information about processing
    - STOP_PROCESSING: Request to stop processing
    - ANONYMIZATION: Request data anonymization

Example:
    >>> # User requests data access
    >>> request = DataRequest.create_request(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     request_type=RequestType.ACCESS,
    ...     description="Kişisel verilerimin bir kopyasını talep ediyorum",
    ...     channel=RequestChannel.WEB
    ... )
    >>> 
    >>> # Assign to DPO
    >>> request.assign(assigned_to_id=dpo.id)
    >>> 
    >>> # Fulfill request
    >>> request.fulfill(
    ...     response="Verileriniz ekteki PDF'te yer almaktadır",
    ...     fulfilled_by_id=dpo.id,
    ...     attachments=[document_id]
    ... )
"""

import enum
from datetime import datetime, timedelta, timezone
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
# CONSTANTS
# =============================================================================

# KVKK requires response within 30 days
KVKK_RESPONSE_DEADLINE_DAYS = 30


# =============================================================================
# ENUMS
# =============================================================================


class RequestType(str, enum.Enum):
    """
    Data request type based on KVKK Article 11.
    
    Types:
    - ACCESS: Right to access personal data
    - CORRECTION: Right to correct inaccurate data
    - DELETION: Right to delete/destroy data
    - OBJECTION: Right to object to processing
    - PORTABILITY: Right to receive data in portable format
    - INFORMATION: Right to information about processing
    - STOP_PROCESSING: Right to stop processing
    - ANONYMIZATION: Right to anonymize data
    - THIRD_PARTY_INFO: Right to know third party recipients
    """
    
    ACCESS = "access"                      # Erişim talebi (KVKK 11/b)
    CORRECTION = "correction"              # Düzeltme talebi (KVKK 11/e)
    DELETION = "deletion"                  # Silme talebi (KVKK 11/f, 7/1)
    OBJECTION = "objection"                # İtiraz talebi (KVKK 11/h)
    PORTABILITY = "portability"            # Taşınabilirlik talebi
    INFORMATION = "information"            # Bilgi talebi (KVKK 11/a-d)
    STOP_PROCESSING = "stop_processing"    # İşlemeyi durdurma talebi
    ANONYMIZATION = "anonymization"        # Anonimleştirme talebi
    THIRD_PARTY_INFO = "third_party_info"  # Üçüncü tarafları öğrenme (KVKK 11/d)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.ACCESS: "Veri Erişim Talebi",
            self.CORRECTION: "Veri Düzeltme Talebi",
            self.DELETION: "Veri Silme Talebi",
            self.OBJECTION: "İtiraz Talebi",
            self.PORTABILITY: "Veri Taşınabilirliği Talebi",
            self.INFORMATION: "Bilgi Edinme Talebi",
            self.STOP_PROCESSING: "İşlemeyi Durdurma Talebi",
            self.ANONYMIZATION: "Anonimleştirme Talebi",
            self.THIRD_PARTY_INFO: "Üçüncü Taraf Bilgisi Talebi",
        }
        return names.get(self, self.value)
    
    @property
    def kvkk_article(self) -> str:
        """KVKK article reference."""
        articles = {
            self.ACCESS: "KVKK Madde 11/b",
            self.CORRECTION: "KVKK Madde 11/e",
            self.DELETION: "KVKK Madde 11/f, 7/1",
            self.OBJECTION: "KVKK Madde 11/h",
            self.INFORMATION: "KVKK Madde 11/a-d",
            self.THIRD_PARTY_INFO: "KVKK Madde 11/d",
        }
        return articles.get(self, "KVKK Madde 11")


class RequestStatus(str, enum.Enum):
    """Data request processing status."""
    
    SUBMITTED = "submitted"          # Newly submitted
    ACKNOWLEDGED = "acknowledged"    # Acknowledged by system
    ASSIGNED = "assigned"            # Assigned to DPO/handler
    IN_PROGRESS = "in_progress"      # Being processed
    FULFILLED = "fulfilled"          # Completed successfully
    REJECTED = "rejected"            # Rejected with reason
    CANCELLED = "cancelled"          # Cancelled by user
    EXPIRED = "expired"              # Response deadline passed
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.SUBMITTED: "Gönderildi",
            self.ACKNOWLEDGED: "Alındı",
            self.ASSIGNED: "Atandı",
            self.IN_PROGRESS: "İşleniyor",
            self.FULFILLED: "Tamamlandı",
            self.REJECTED: "Reddedildi",
            self.CANCELLED: "İptal Edildi",
            self.EXPIRED: "Süresi Doldu",
        }
        return names.get(self, self.value)


class RequestChannel(str, enum.Enum):
    """Channel through which request was submitted."""
    
    WEB = "web"              # Web application
    MOBILE = "mobile"        # Mobile app
    EMAIL = "email"          # Email
    MAIL = "mail"            # Postal mail
    IN_PERSON = "in_person"  # In person at office
    PHONE = "phone"          # Phone call
    FAX = "fax"              # Fax
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.WEB: "Web Sitesi",
            self.MOBILE: "Mobil Uygulama",
            self.EMAIL: "E-posta",
            self.MAIL: "Posta",
            self.IN_PERSON: "Şahsen Başvuru",
            self.PHONE: "Telefon",
            self.FAX: "Faks",
        }
        return names.get(self, self.value)


class RequestPriority(str, enum.Enum):
    """Request priority level."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# DATA REQUEST MODEL
# =============================================================================


class DataRequest(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Data Request model for KVKK data subject rights.
    
    Manages user requests for:
    - Data access
    - Data correction
    - Data deletion
    - Objections
    - Data portability
    
    KVKK Compliance:
        - Article 11: Data subject rights
        - 30-day response deadline
        - Free of charge (first request)
        - Audit trail required
        - Rejection must be justified
    
    Request Lifecycle:
    1. User submits request
    2. System acknowledges (auto)
    3. Assigned to DPO/handler
    4. DPO processes request
    5. Request fulfilled or rejected
    6. User notified
    
    Attributes:
        user_id: User making the request
        user: User relationship
        
        request_type: Type of request
        request_number: Unique request number
        
        status: Current status
        priority: Priority level
        
        description: User's request description
        detailed_request: Detailed request information
        
        channel: How request was submitted
        
        submitted_at: Submission timestamp
        deadline: Response deadline (30 days)
        
        assigned_to_id: DPO/handler assigned
        assigned_to: Assigned user relationship
        assigned_at: Assignment timestamp
        
        in_progress_at: Processing start timestamp
        completed_at: Completion timestamp
        
        response: Response/explanation
        response_details: Detailed response (JSON)
        
        fulfilled_by_id: Who fulfilled the request
        fulfilled_by: Fulfiller relationship
        
        rejection_reason: Why request was rejected
        rejection_legal_basis: Legal basis for rejection
        
        attachments: Related document IDs (array)
        
        is_free: Request is free of charge (first request)
        fee_amount: Fee charged (if applicable)
        
        metadata: Additional context
        
        verification_required: Identity verification needed
        verification_completed_at: Verification timestamp
        
    Relationships:
        tenant: Parent tenant
        user: User making request
        assigned_to: DPO/handler
        fulfilled_by: Who completed request
    """
    
    __tablename__ = "data_requests"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User making the data request",
    )
    
    user = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="data_requests",
    )
    
    # =========================================================================
    # REQUEST IDENTIFICATION
    # =========================================================================
    
    request_type = Column(
        Enum(RequestType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of data request (KVKK Article 11)",
    )
    
    request_number = Column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique request identifier (e.g., KVK-2025-00001)",
    )
    
    # =========================================================================
    # STATUS & PRIORITY
    # =========================================================================
    
    status = Column(
        Enum(RequestStatus, native_enum=False, length=50),
        nullable=False,
        default=RequestStatus.SUBMITTED,
        index=True,
        comment="Request processing status",
    )
    
    priority = Column(
        Enum(RequestPriority, native_enum=False, length=50),
        nullable=False,
        default=RequestPriority.NORMAL,
        comment="Request priority level",
    )
    
    # =========================================================================
    # REQUEST CONTENT
    # =========================================================================
    
    description = Column(
        Text,
        nullable=False,
        comment="User's request description",
    )
    
    detailed_request = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed request information (specific data categories, etc.)",
    )
    
    # =========================================================================
    # CHANNEL
    # =========================================================================
    
    channel = Column(
        Enum(RequestChannel, native_enum=False, length=50),
        nullable=False,
        comment="Channel through which request was submitted",
    )
    
    # =========================================================================
    # TIMELINE
    # =========================================================================
    
    submitted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="When request was submitted",
    )
    
    deadline = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Response deadline (KVKK: 30 days from submission)",
    )
    
    assigned_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When request was assigned to handler",
    )
    
    in_progress_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing started",
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When request was completed (fulfilled/rejected)",
    )
    
    # =========================================================================
    # ASSIGNMENT
    # =========================================================================
    
    assigned_to_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="DPO or handler assigned to process request",
    )
    
    assigned_to = relationship(
        "User",
        foreign_keys=[assigned_to_id],
        back_populates="data_requests_assigned",
    )
    
    # =========================================================================
    # RESPONSE
    # =========================================================================
    
    response = Column(
        Text,
        nullable=True,
        comment="Response text to user",
    )
    
    response_details = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed response information (actions taken, data provided)",
    )
    
    fulfilled_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="User who fulfilled the request",
    )
    
    fulfilled_by = relationship(
        "User",
        foreign_keys=[fulfilled_by_id],
        back_populates="data_requests_fulfilled",
    )
    
    # =========================================================================
    # REJECTION
    # =========================================================================
    
    rejection_reason = Column(
        Text,
        nullable=True,
        comment="Reason for rejection (must be legally justified)",
    )
    
    rejection_legal_basis = Column(
        String(500),
        nullable=True,
        comment="Legal basis for rejection (KVKK article/clause)",
    )
    
    # =========================================================================
    # ATTACHMENTS
    # =========================================================================
    
    attachments = Column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list,
        comment="Related document IDs (identity verification, response docs)",
    )
    
    # =========================================================================
    # FEES
    # =========================================================================
    
    is_free = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Request is free of charge (KVKK: first request free)",
    )
    
    fee_amount = Column(
        Integer,
        nullable=True,
        comment="Fee charged in kuruş (if applicable for repeated requests)",
    )
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    verification_required = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Identity verification required for sensitive requests",
    )
    
    verification_completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When identity verification was completed",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (IP, user agent, communication history)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user's requests
        Index(
            "ix_data_requests_user",
            "user_id",
            "submitted_at",
        ),
        
        # Index for assigned requests
        Index(
            "ix_data_requests_assigned",
            "assigned_to_id",
            "status",
            postgresql_where="assigned_to_id IS NOT NULL",
        ),
        
        # Index for pending requests
        Index(
            "ix_data_requests_pending",
            "status",
            "deadline",
            postgresql_where="status IN ('submitted', 'acknowledged', 'assigned', 'in_progress')",
        ),
        
        # Index for overdue requests
        Index(
            "ix_data_requests_overdue",
            "deadline",
            "status",
            postgresql_where="status NOT IN ('fulfilled', 'rejected', 'cancelled')",
        ),
        
        # Index for request type statistics
        Index(
            "ix_data_requests_type",
            "tenant_id",
            "request_type",
            "status",
        ),
        
        # Check: fee amount non-negative
        CheckConstraint(
            "fee_amount IS NULL OR fee_amount >= 0",
            name="ck_data_requests_fee",
        ),
    )
    
    # =========================================================================
    # REQUEST CREATION
    # =========================================================================
    
    @classmethod
    def create_request(
        cls,
        user_id: UUIDType,
        tenant_id: UUIDType,
        request_type: RequestType,
        description: str,
        channel: RequestChannel,
        detailed_request: dict[str, Any] | None = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        verification_required: bool = False,
    ) -> "DataRequest":
        """
        Create a new data request.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant UUID
            request_type: Type of request
            description: Request description
            channel: Submission channel
            detailed_request: Additional details
            priority: Priority level
            verification_required: Require identity verification
            
        Returns:
            DataRequest: New request instance
            
        Example:
            >>> request = DataRequest.create_request(
            ...     user_id=user.id,
            ...     tenant_id=tenant.id,
            ...     request_type=RequestType.ACCESS,
            ...     description="Kişisel verilerimin bir kopyasını talep ediyorum",
            ...     channel=RequestChannel.WEB,
            ...     detailed_request={
            ...         "categories": ["kimlik", "iletişim", "işlem"]
            ...     }
            ... )
        """
        # Generate request number
        request_number = cls._generate_request_number()
        
        # Calculate deadline (30 days from now)
        submitted_at = datetime.now(timezone.utc)
        deadline = submitted_at + timedelta(days=KVKK_RESPONSE_DEADLINE_DAYS)
        
        request = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            request_type=request_type,
            request_number=request_number,
            description=description,
            channel=channel,
            priority=priority,
            status=RequestStatus.SUBMITTED,
            submitted_at=submitted_at,
            deadline=deadline,
            detailed_request=detailed_request or {},
            verification_required=verification_required,
            is_free=True,  # First request always free per KVKK
        )
        
        logger.info(
            "Data request created",
            request_id=str(request.id),
            request_number=request_number,
            user_id=str(user_id),
            request_type=request_type.value,
        )
        
        return request
    
    @staticmethod
    def _generate_request_number() -> str:
        """
        Generate unique request number.
        
        Format: KVK-YYYY-NNNNN
        
        Returns:
            str: Request number
        """
        import random
        
        year = datetime.now(timezone.utc).year
        # In production, this would query DB for next sequence number
        sequence = random.randint(1, 99999)
        return f"KVK-{year}-{sequence:05d}"
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def acknowledge(self) -> None:
        """Acknowledge receipt of request (auto)."""
        self.status = RequestStatus.ACKNOWLEDGED
        
        logger.info(
            "Data request acknowledged",
            request_id=str(self.id),
            request_number=self.request_number,
        )
    
    def assign(self, assigned_to_id: UUIDType) -> None:
        """
        Assign request to DPO/handler.
        
        Args:
            assigned_to_id: Handler user UUID
        """
        self.assigned_to_id = assigned_to_id
        self.assigned_at = datetime.now(timezone.utc)
        self.status = RequestStatus.ASSIGNED
        
        logger.info(
            "Data request assigned",
            request_id=str(self.id),
            assigned_to_id=str(assigned_to_id),
        )
    
    def start_processing(self) -> None:
        """Start processing request."""
        self.status = RequestStatus.IN_PROGRESS
        self.in_progress_at = datetime.now(timezone.utc)
        
        logger.info(
            "Data request processing started",
            request_id=str(self.id),
        )
    
    def fulfill(
        self,
        response: str,
        fulfilled_by_id: UUIDType,
        response_details: dict[str, Any] | None = None,
        attachments: list[UUIDType] | None = None,
    ) -> None:
        """
        Fulfill request.
        
        Args:
            response: Response text
            fulfilled_by_id: Who fulfilled
            response_details: Detailed response
            attachments: Response documents
            
        Example:
            >>> request.fulfill(
            ...     response="Kişisel verileriniz ekte yer almaktadır",
            ...     fulfilled_by_id=dpo.id,
            ...     response_details={
            ...         "data_categories": ["kimlik", "iletişim"],
            ...         "documents_count": 5
            ...     },
            ...     attachments=[doc_id]
            ... )
        """
        self.status = RequestStatus.FULFILLED
        self.completed_at = datetime.now(timezone.utc)
        self.response = response
        self.response_details = response_details or {}
        self.fulfilled_by_id = fulfilled_by_id
        
        if attachments:
            self.attachments = attachments
        
        logger.info(
            "Data request fulfilled",
            request_id=str(self.id),
            request_number=self.request_number,
            fulfilled_by_id=str(fulfilled_by_id),
            days_taken=self.days_to_complete(),
        )
    
    def reject(
        self,
        rejection_reason: str,
        rejection_legal_basis: str,
        fulfilled_by_id: UUIDType,
    ) -> None:
        """
        Reject request with legal justification.
        
        Args:
            rejection_reason: Human-readable reason
            rejection_legal_basis: Legal basis (KVKK article)
            fulfilled_by_id: Who rejected
            
        Example:
            >>> request.reject(
            ...     rejection_reason="Talep edilen veri mevcut değil",
            ...     rejection_legal_basis="KVKK Madde 7/1-a",
            ...     fulfilled_by_id=dpo.id
            ... )
        """
        self.status = RequestStatus.REJECTED
        self.completed_at = datetime.now(timezone.utc)
        self.rejection_reason = rejection_reason
        self.rejection_legal_basis = rejection_legal_basis
        self.fulfilled_by_id = fulfilled_by_id
        
        logger.warning(
            "Data request rejected",
            request_id=str(self.id),
            request_number=self.request_number,
            reason=rejection_reason,
        )
    
    def cancel(self, reason: str | None = None) -> None:
        """
        Cancel request (user-initiated).
        
        Args:
            reason: Cancellation reason
        """
        self.status = RequestStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        
        if reason:
            self.metadata["cancellation_reason"] = reason
            self.metadata["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "metadata")
        
        logger.info(
            "Data request cancelled",
            request_id=str(self.id),
            reason=reason,
        )
    
    def mark_expired(self) -> None:
        """Mark request as expired (deadline passed)."""
        self.status = RequestStatus.EXPIRED
        
        logger.error(
            "Data request expired",
            request_id=str(self.id),
            request_number=self.request_number,
            deadline=self.deadline.isoformat(),
        )
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def complete_verification(self) -> None:
        """Mark identity verification as completed."""
        self.verification_completed_at = datetime.now(timezone.utc)
        
        logger.info(
            "Identity verification completed",
            request_id=str(self.id),
        )
    
    # =========================================================================
    # TIMELINE HELPERS
    # =========================================================================
    
    def days_to_deadline(self) -> int:
        """
        Get days remaining until deadline.
        
        Returns:
            int: Days remaining (negative if overdue)
        """
        delta = self.deadline - datetime.now(timezone.utc)
        return delta.days
    
    def is_overdue(self) -> bool:
        """Check if request is overdue."""
        if self.status in [RequestStatus.FULFILLED, RequestStatus.REJECTED, RequestStatus.CANCELLED]:
            return False
        
        return datetime.now(timezone.utc) > self.deadline
    
    def days_to_complete(self) -> int | None:
        """
        Get days taken to complete request.
        
        Returns:
            int | None: Days taken or None if not completed
        """
        if not self.completed_at:
            return None
        
        delta = self.completed_at - self.submitted_at
        return delta.days
    
    def was_completed_on_time(self) -> bool | None:
        """
        Check if request was completed within deadline.
        
        Returns:
            bool | None: True if on time, False if late, None if not completed
        """
        if not self.completed_at:
            return None
        
        return self.completed_at <= self.deadline
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("description")
    def validate_description(self, key: str, description: str) -> str:
        """Validate description."""
        if not description or not description.strip():
            raise ValidationError(
                message="Description cannot be empty",
                field="description",
            )
        
        return description.strip()
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<DataRequest("
            f"id={self.id}, "
            f"number={self.request_number}, "
            f"type={self.request_type.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        
        # Add display names
        data["request_type_display"] = self.request_type.display_name_tr
        data["request_type_kvkk_article"] = self.request_type.kvkk_article
        data["status_display"] = self.status.display_name_tr
        data["channel_display"] = self.channel.display_name_tr
        data["priority_display"] = self.priority.value.upper()
        
        # Add computed fields
        data["days_to_deadline"] = self.days_to_deadline()
        data["is_overdue"] = self.is_overdue()
        data["days_to_complete"] = self.days_to_complete()
        
        if self.completed_at:
            data["was_completed_on_time"] = self.was_completed_on_time()
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "DataRequest",
    "RequestType",
    "RequestStatus",
    "RequestChannel",
    "RequestPriority",
    "KVKK_RESPONSE_DEADLINE_DAYS",
]