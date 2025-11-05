"""
Feedback model for user feedback and satisfaction tracking in Turkish Legal AI.

This module provides the Feedback model for collecting user feedback:
- Product feedback (feature requests, bugs, improvements)
- User satisfaction (NPS, CSAT)
- Document quality ratings
- AI response quality ratings
- Support ticket feedback
- Onboarding experience feedback
- Multi-dimensional ratings
- Sentiment analysis ready

Feedback Types:
    - GENERAL: General product feedback
    - FEATURE_REQUEST: Feature suggestion
    - BUG_REPORT: Bug or issue report
    - DOCUMENT_QUALITY: Document processing quality
    - AI_RESPONSE: AI chat response quality
    - SUPPORT: Support interaction quality
    - ONBOARDING: Onboarding experience
    - NPS: Net Promoter Score

Feedback Categories:
    - FUNCTIONALITY: Product functionality
    - USABILITY: Ease of use
    - PERFORMANCE: Speed and performance
    - ACCURACY: AI accuracy
    - DESIGN: UI/UX design
    - SUPPORT: Customer support
    - DOCUMENTATION: Help documentation

Example:
    >>> # User rates AI response
    >>> feedback = Feedback.create_feedback(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     feedback_type=FeedbackType.AI_RESPONSE,
    ...     rating=5,
    ...     comment="Çok yardımcı oldu, teşekkürler!",
    ...     related_entity_type="chat_message",
    ...     related_entity_id=message_id
    ... )
    >>> 
    >>> # Submit NPS survey
    >>> nps = Feedback.create_feedback(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     feedback_type=FeedbackType.NPS,
    ...     rating=9,
    ...     comment="Mükemmel ürün, çok memnunum"
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
    SoftDeleteMixin,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class FeedbackType(str, enum.Enum):
    """
    Feedback type classification.
    
    Types:
    - GENERAL: General product feedback
    - FEATURE_REQUEST: Feature suggestion or request
    - BUG_REPORT: Bug or technical issue report
    - DOCUMENT_QUALITY: Document processing quality feedback
    - AI_RESPONSE: AI chat response quality feedback
    - SUPPORT: Customer support interaction feedback
    - ONBOARDING: Onboarding/registration experience
    - NPS: Net Promoter Score survey
    - CSAT: Customer Satisfaction survey
    """
    
    GENERAL = "general"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    DOCUMENT_QUALITY = "document_quality"
    AI_RESPONSE = "ai_response"
    SUPPORT = "support"
    ONBOARDING = "onboarding"
    NPS = "nps"
    CSAT = "csat"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.GENERAL: "Genel Geri Bildirim",
            self.FEATURE_REQUEST: "Özellik İsteği",
            self.BUG_REPORT: "Hata Bildirimi",
            self.DOCUMENT_QUALITY: "Belge Kalitesi",
            self.AI_RESPONSE: "Yapay Zeka Yanıtı",
            self.SUPPORT: "Destek Hizmeti",
            self.ONBOARDING: "İlk Kullanım Deneyimi",
            self.NPS: "Net Promoter Score",
            self.CSAT: "Müşteri Memnuniyeti",
        }
        return names.get(self, self.value)


class FeedbackCategory(str, enum.Enum):
    """Feedback category for grouping."""
    
    FUNCTIONALITY = "functionality"      # Feature functionality
    USABILITY = "usability"              # Ease of use
    PERFORMANCE = "performance"          # Speed, reliability
    ACCURACY = "accuracy"                # AI accuracy, correctness
    DESIGN = "design"                    # UI/UX design
    SUPPORT = "support"                  # Customer support quality
    DOCUMENTATION = "documentation"      # Help docs, guides
    PRICING = "pricing"                  # Pricing and billing
    INTEGRATION = "integration"          # Third-party integrations
    SECURITY = "security"                # Security and privacy
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.FUNCTIONALITY: "İşlevsellik",
            self.USABILITY: "Kullanım Kolaylığı",
            self.PERFORMANCE: "Performans",
            self.ACCURACY: "Doğruluk",
            self.DESIGN: "Tasarım",
            self.SUPPORT: "Destek",
            self.DOCUMENTATION: "Dokümantasyon",
            self.PRICING: "Fiyatlandırma",
            self.INTEGRATION: "Entegrasyon",
            self.SECURITY: "Güvenlik",
        }
        return names.get(self, self.value)


class FeedbackStatus(str, enum.Enum):
    """Feedback processing status."""
    
    NEW = "new"                  # Newly submitted
    REVIEWED = "reviewed"        # Reviewed by team
    IN_PROGRESS = "in_progress"  # Being worked on
    RESOLVED = "resolved"        # Issue resolved
    CLOSED = "closed"            # Closed (no action)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.NEW: "Yeni",
            self.REVIEWED: "İncelendi",
            self.IN_PROGRESS: "İşlemde",
            self.RESOLVED: "Çözüldü",
            self.CLOSED: "Kapatıldı",
        }
        return names.get(self, self.value)


class Sentiment(str, enum.Enum):
    """Feedback sentiment (for analysis)."""
    
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# FEEDBACK MODEL
# =============================================================================


class Feedback(Base, BaseModelMixin, TenantMixin, SoftDeleteMixin):
    """
    Feedback model for user feedback collection and tracking.
    
    Collects and manages:
    - User satisfaction ratings
    - Feature requests
    - Bug reports
    - Product feedback
    - NPS/CSAT surveys
    
    Rating Systems:
        - 1-5 stars: General satisfaction
        - 0-10: NPS (Net Promoter Score)
        - 1-5: CSAT (Customer Satisfaction)
        - Multi-dimensional: Multiple rating dimensions
    
    NPS Calculation:
        - 0-6: Detractors
        - 7-8: Passives
        - 9-10: Promoters
        - NPS = % Promoters - % Detractors
    
    Attributes:
        user_id: User providing feedback
        user: User relationship
        
        feedback_type: Type of feedback
        category: Feedback category
        
        rating: Numeric rating (scale depends on type)
        comment: User's feedback text
        
        related_entity_type: Entity being rated (document, message, etc.)
        related_entity_id: Entity UUID
        
        dimensions: Multi-dimensional ratings (JSON)
        
        status: Processing status
        sentiment: Detected sentiment
        
        tags: Categorization tags (array)
        
        is_anonymous: Anonymous feedback flag
        
        response: Team response to feedback
        responded_by_id: Who responded
        responded_at: Response timestamp
        
        metadata: Additional context (browser, OS, session)
        
        upvotes: Number of upvotes (if public)
        downvotes: Number of downvotes (if public)
        
    Relationships:
        tenant: Parent tenant
        user: User providing feedback
        responded_by: Team member who responded
    """
    
    __tablename__ = "feedback"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,  # Nullable for anonymous feedback
        index=True,
        comment="User providing feedback (NULL if anonymous)",
    )
    
    user = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="feedback",
    )
    
    # =========================================================================
    # FEEDBACK CLASSIFICATION
    # =========================================================================
    
    feedback_type = Column(
        Enum(FeedbackType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of feedback",
    )
    
    category = Column(
        Enum(FeedbackCategory, native_enum=False, length=50),
        nullable=True,
        index=True,
        comment="Feedback category",
    )
    
    # =========================================================================
    # RATING & COMMENT
    # =========================================================================
    
    rating = Column(
        Integer,
        nullable=True,
        comment="Numeric rating (1-5 for general, 0-10 for NPS)",
    )
    
    comment = Column(
        Text,
        nullable=True,
        comment="User's feedback text/comment",
    )
    
    # =========================================================================
    # RELATED ENTITY (what is being rated)
    # =========================================================================
    
    related_entity_type = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Entity type (document, chat_message, support_ticket, etc.)",
    )
    
    related_entity_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Entity UUID being rated",
    )
    
    # =========================================================================
    # MULTI-DIMENSIONAL RATINGS
    # =========================================================================
    
    dimensions = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Multi-dimensional ratings (accuracy, speed, helpfulness, etc.)",
    )
    
    # =========================================================================
    # STATUS & SENTIMENT
    # =========================================================================
    
    status = Column(
        Enum(FeedbackStatus, native_enum=False, length=50),
        nullable=False,
        default=FeedbackStatus.NEW,
        index=True,
        comment="Processing status",
    )
    
    sentiment = Column(
        Enum(Sentiment, native_enum=False, length=50),
        nullable=True,
        index=True,
        comment="Detected sentiment (from comment analysis)",
    )
    
    # =========================================================================
    # TAGS
    # =========================================================================
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Categorization tags for filtering and analysis",
    )
    
    # =========================================================================
    # ANONYMITY
    # =========================================================================
    
    is_anonymous = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Anonymous feedback (user_id hidden from analysis)",
    )
    
    # =========================================================================
    # RESPONSE
    # =========================================================================
    
    response = Column(
        Text,
        nullable=True,
        comment="Team response to feedback",
    )
    
    responded_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Team member who responded",
    )
    
    responded_by = relationship(
        "User",
        foreign_keys=[responded_by_id],
        back_populates="feedback_responses",
    )
    
    responded_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When team responded",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (browser, OS, page, session_id)",
    )
    
    # =========================================================================
    # SOCIAL (if feedback is public)
    # =========================================================================
    
    upvotes = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of upvotes (for public feedback/feature requests)",
    )
    
    downvotes = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of downvotes",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user's feedback
        Index(
            "ix_feedback_user",
            "user_id",
            "created_at",
        ),
        
        # Index for entity feedback
        Index(
            "ix_feedback_entity",
            "related_entity_type",
            "related_entity_id",
        ),
        
        # Index for type and rating (analytics)
        Index(
            "ix_feedback_type_rating",
            "feedback_type",
            "rating",
            postgresql_where="rating IS NOT NULL",
        ),
        
        # Index for NPS calculation
        Index(
            "ix_feedback_nps",
            "tenant_id",
            "feedback_type",
            "rating",
            "created_at",
            postgresql_where="feedback_type = 'nps' AND rating IS NOT NULL",
        ),
        
        # Index for new feedback
        Index(
            "ix_feedback_new",
            "status",
            "created_at",
            postgresql_where="status = 'new'",
        ),
        
        # Check: rating range depends on type
        CheckConstraint(
            "rating IS NULL OR "
            "(feedback_type = 'nps' AND rating >= 0 AND rating <= 10) OR "
            "(feedback_type != 'nps' AND rating >= 1 AND rating <= 5)",
            name="ck_feedback_rating_range",
        ),
        
        # Check: upvotes/downvotes non-negative
        CheckConstraint(
            "upvotes >= 0 AND downvotes >= 0",
            name="ck_feedback_votes",
        ),
    )
    
    # =========================================================================
    # FEEDBACK CREATION
    # =========================================================================
    
    @classmethod
    def create_feedback(
        cls,
        user_id: UUIDType | None,
        tenant_id: UUIDType,
        feedback_type: FeedbackType,
        rating: int | None = None,
        comment: str | None = None,
        category: FeedbackCategory | None = None,
        related_entity_type: str | None = None,
        related_entity_id: UUIDType | None = None,
        dimensions: dict[str, int] | None = None,
        tags: list[str] | None = None,
        is_anonymous: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "Feedback":
        """
        Create a new feedback entry.
        
        Args:
            user_id: User UUID (None if anonymous)
            tenant_id: Tenant UUID
            feedback_type: Type of feedback
            rating: Numeric rating
            comment: Feedback text
            category: Feedback category
            related_entity_type: Entity being rated
            related_entity_id: Entity UUID
            dimensions: Multi-dimensional ratings
            tags: Categorization tags
            is_anonymous: Anonymous flag
            metadata: Additional context
            
        Returns:
            Feedback: New feedback instance
            
        Example:
            >>> feedback = Feedback.create_feedback(
            ...     user_id=user.id,
            ...     tenant_id=tenant.id,
            ...     feedback_type=FeedbackType.AI_RESPONSE,
            ...     rating=5,
            ...     comment="Çok yardımcı bir yanıt, teşekkürler!",
            ...     category=FeedbackCategory.ACCURACY,
            ...     related_entity_type="chat_message",
            ...     related_entity_id=message_id,
            ...     dimensions={
            ...         "accuracy": 5,
            ...         "helpfulness": 5,
            ...         "clarity": 4
            ...     }
            ... )
        """
        feedback = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            category=category,
            related_entity_type=related_entity_type,
            related_entity_id=related_entity_id,
            dimensions=dimensions or {},
            tags=tags or [],
            is_anonymous=is_anonymous,
            metadata=metadata or {},
            status=FeedbackStatus.NEW,
        )
        
        # Auto-detect sentiment if comment provided
        if comment:
            feedback.sentiment = feedback._detect_sentiment(comment)
        
        logger.info(
            "Feedback created",
            feedback_id=str(feedback.id),
            feedback_type=feedback_type.value,
            rating=rating,
            is_anonymous=is_anonymous,
        )
        
        return feedback
    
    @staticmethod
    def _detect_sentiment(comment: str) -> Sentiment:
        """
        Detect sentiment from comment.
        
        This is a placeholder. Production would use:
        - NLP sentiment analysis
        - Turkish language model
        - Multi-class classification
        
        Args:
            comment: Comment text
            
        Returns:
            Sentiment: Detected sentiment
        """
        # Placeholder: Simple keyword-based detection
        comment_lower = comment.lower()
        
        positive_words = ["mükemmel", "harika", "çok iyi", "teşekkür", "güzel", "başarılı"]
        negative_words = ["kötü", "berbat", "çalışmıyor", "hata", "sorun", "yetersiz"]
        
        positive_count = sum(1 for word in positive_words if word in comment_lower)
        negative_count = sum(1 for word in negative_words if word in comment_lower)
        
        if positive_count > negative_count:
            return Sentiment.POSITIVE
        elif negative_count > positive_count:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.NEUTRAL
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def mark_reviewed(self) -> None:
        """Mark feedback as reviewed."""
        self.status = FeedbackStatus.REVIEWED
        
        logger.debug(
            "Feedback marked as reviewed",
            feedback_id=str(self.id),
        )
    
    def start_processing(self) -> None:
        """Mark feedback as being worked on."""
        self.status = FeedbackStatus.IN_PROGRESS
        
        logger.info(
            "Feedback processing started",
            feedback_id=str(self.id),
        )
    
    def resolve(self, resolution_notes: str | None = None) -> None:
        """
        Mark feedback as resolved.
        
        Args:
            resolution_notes: Resolution details
        """
        self.status = FeedbackStatus.RESOLVED
        
        if resolution_notes:
            self.metadata["resolution_notes"] = resolution_notes
            self.metadata["resolved_at"] = datetime.now(timezone.utc).isoformat()
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "metadata")
        
        logger.info(
            "Feedback resolved",
            feedback_id=str(self.id),
        )
    
    def close(self, reason: str | None = None) -> None:
        """
        Close feedback without action.
        
        Args:
            reason: Closure reason
        """
        self.status = FeedbackStatus.CLOSED
        
        if reason:
            self.metadata["closure_reason"] = reason
            self.metadata["closed_at"] = datetime.now(timezone.utc).isoformat()
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "metadata")
        
        logger.info(
            "Feedback closed",
            feedback_id=str(self.id),
            reason=reason,
        )
    
    # =========================================================================
    # RESPONSE MANAGEMENT
    # =========================================================================
    
    def add_response(self, response: str, responded_by_id: UUIDType) -> None:
        """
        Add team response to feedback.
        
        Args:
            response: Response text
            responded_by_id: Team member UUID
            
        Example:
            >>> feedback.add_response(
            ...     response="Geri bildiriminiz için teşekkürler! Bu özelliği geliştirme planımıza ekledik.",
            ...     responded_by_id=admin.id
            ... )
        """
        self.response = response
        self.responded_by_id = responded_by_id
        self.responded_at = datetime.now(timezone.utc)
        
        logger.info(
            "Response added to feedback",
            feedback_id=str(self.id),
            responded_by_id=str(responded_by_id),
        )
    
    # =========================================================================
    # VOTING (for public feedback)
    # =========================================================================
    
    def upvote(self) -> None:
        """Increment upvote count."""
        self.upvotes += 1
        
        logger.debug(
            "Feedback upvoted",
            feedback_id=str(self.id),
            upvotes=self.upvotes,
        )
    
    def downvote(self) -> None:
        """Increment downvote count."""
        self.downvotes += 1
        
        logger.debug(
            "Feedback downvoted",
            feedback_id=str(self.id),
            downvotes=self.downvotes,
        )
    
    def get_vote_score(self) -> int:
        """
        Get net vote score.
        
        Returns:
            int: Upvotes - downvotes
        """
        return self.upvotes - self.downvotes
    
    # =========================================================================
    # NPS HELPERS
    # =========================================================================
    
    def get_nps_category(self) -> str | None:
        """
        Get NPS category for this feedback.
        
        Returns:
            str | None: "promoter", "passive", "detractor", or None
        """
        if self.feedback_type != FeedbackType.NPS or self.rating is None:
            return None
        
        if self.rating >= 9:
            return "promoter"
        elif self.rating >= 7:
            return "passive"
        else:
            return "detractor"
    
    @classmethod
    def calculate_nps(cls, feedback_list: list["Feedback"]) -> float | None:
        """
        Calculate NPS from feedback list.
        
        NPS = % Promoters - % Detractors
        
        Args:
            feedback_list: List of NPS feedback
            
        Returns:
            float | None: NPS score (-100 to 100) or None if no data
        """
        nps_feedback = [
            f for f in feedback_list
            if f.feedback_type == FeedbackType.NPS and f.rating is not None
        ]
        
        if not nps_feedback:
            return None
        
        total = len(nps_feedback)
        promoters = sum(1 for f in nps_feedback if f.rating >= 9)
        detractors = sum(1 for f in nps_feedback if f.rating <= 6)
        
        nps = ((promoters - detractors) / total) * 100
        return round(nps, 2)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("rating")
    def validate_rating(self, key: str, rating: int | None) -> int | None:
        """Validate rating based on feedback type."""
        if rating is None:
            return rating
        
        # NPS: 0-10
        if self.feedback_type == FeedbackType.NPS:
            if not 0 <= rating <= 10:
                raise ValidationError(
                    message="NPS rating must be between 0 and 10",
                    field="rating",
                )
        # Others: 1-5
        else:
            if not 1 <= rating <= 5:
                raise ValidationError(
                    message="Rating must be between 1 and 5",
                    field="rating",
                )
        
        return rating
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Feedback("
            f"id={self.id}, "
            f"type={self.feedback_type.value}, "
            f"rating={self.rating}"
            f")>"
        )
    
    def to_dict(self, include_user_info: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_user_info: Include user information (respect anonymity)
            
        Returns:
            dict: Feedback data
        """
        data = super().to_dict()
        
        # Respect anonymity
        if self.is_anonymous and not include_user_info:
            data.pop("user_id", None)
        
        # Add display names
        data["feedback_type_display"] = self.feedback_type.display_name_tr
        
        if self.category:
            data["category_display"] = self.category.display_name_tr
        
        data["status_display"] = self.status.display_name_tr
        
        # Add computed fields
        data["vote_score"] = self.get_vote_score()
        data["has_response"] = self.response is not None
        
        if self.feedback_type == FeedbackType.NPS:
            data["nps_category"] = self.get_nps_category()
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Feedback",
    "FeedbackType",
    "FeedbackCategory",
    "FeedbackStatus",
    "Sentiment",
]