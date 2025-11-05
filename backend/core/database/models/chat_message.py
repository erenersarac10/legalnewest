"""
Chat Message model for individual conversation messages in Turkish Legal AI.

This module provides the ChatMessage model for storing AI conversation messages:
- User and assistant messages
- Message threading and branching
- Token tracking per message
- Citation and reference tracking
- Message regeneration support
- Edit history
- KVKK-compliant storage
- Turkish language optimization

Message Architecture:
    ChatSession → ChatMessage → Citations/References
    
    Each message contains:
    - Role (user, assistant, system)
    - Content (text, structured data)
    - Token counts
    - Citations (document references)
    - Metadata (model, temperature, timing)
    - Parent message (for threading)

Message Types:
    - User: Human input
    - Assistant: AI response
    - System: System notifications
    - Function: Function call results

Features:
    - Message regeneration (retry with different params)
    - Message editing (user edits their message)
    - Message branching (multiple AI responses)
    - Citation tracking (which documents were used)
    - Token usage per message
    - Response time tracking

Example:
    >>> # Create user message
    >>> user_msg = ChatMessage(
    ...     session_id=session.id,
    ...     role=MessageRole.USER,
    ...     content="Bu sözleşmedeki fesih koşulları nelerdir?",
    ...     tenant_id=tenant_id
    ... )
    >>> 
    >>> # Create assistant response
    >>> assistant_msg = ChatMessage(
    ...     session_id=session.id,
    ...     role=MessageRole.ASSISTANT,
    ...     content="Sözleşmede belirtilen fesih koşulları...",
    ...     parent_message_id=user_msg.id,
    ...     prompt_tokens=500,
    ...     completion_tokens=300,
    ...     tenant_id=tenant_id
    ... )
    >>> 
    >>> # Add citations
    >>> assistant_msg.add_citation(document_id, page_numbers=[5, 6])
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


class MessageRole(str, enum.Enum):
    """
    Message role in conversation.
    
    Roles:
    - SYSTEM: System instructions/notifications
    - USER: Human user message
    - ASSISTANT: AI assistant response
    - FUNCTION: Function call result
    """
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.SYSTEM: "Sistem",
            self.USER: "Kullanıcı",
            self.ASSISTANT: "Asistan",
            self.FUNCTION: "Fonksiyon",
        }
        return names.get(self, self.value)


class MessageStatus(str, enum.Enum):
    """Message processing status."""
    
    PENDING = "pending"              # Waiting for processing
    PROCESSING = "processing"        # AI generating response
    COMPLETED = "completed"          # Successfully completed
    FAILED = "failed"                # Processing failed
    CANCELLED = "cancelled"          # User cancelled
    
    def __str__(self) -> str:
        return self.value


class ContentType(str, enum.Enum):
    """Message content type."""
    
    TEXT = "text"                    # Plain text
    MARKDOWN = "markdown"            # Markdown formatted
    JSON = "json"                    # Structured JSON
    CODE = "code"                    # Code block
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# CHAT MESSAGE MODEL
# =============================================================================


class ChatMessage(Base, BaseModelMixin, TenantMixin, SoftDeleteMixin):
    """
    Chat Message model for conversation messages.
    
    Messages represent individual turns in a conversation:
    - User questions/inputs
    - AI assistant responses
    - System messages
    - Function call results
    
    Message Threading:
        - Parent-child relationships
        - Multiple responses per user message
        - Message regeneration support
        - Branching conversations
    
    Citation Tracking:
        - Which documents were referenced
        - Page numbers cited
        - Relevance scores
        - Direct quotes
    
    Token Tracking:
        - Input tokens (prompt)
        - Output tokens (completion)
        - Per-message cost calculation
    
    Attributes:
        session_id: Parent chat session
        session: Session relationship
        
        role: Message role (user, assistant, system)
        content: Message content
        content_type: Content format
        
        parent_message_id: Parent message (threading)
        parent_message: Parent relationship
        child_messages: Child messages (branches)
        
        sequence_number: Order in conversation
        
        prompt_tokens: Input tokens used
        completion_tokens: Output tokens generated
        total_tokens: Total tokens
        
        model: AI model used
        temperature: Model temperature
        
        processing_status: Processing status
        error_message: Error details if failed
        
        response_time_ms: Response generation time
        
        citations: Document citations (JSON)
        metadata: Additional metadata
        
        is_edited: Message was edited
        is_regenerated: Response was regenerated
        
        feedback_rating: User feedback (1-5)
        feedback_comment: User feedback text
        
    Relationships:
        tenant: Parent tenant
        session: Parent chat session
        parent_message: Parent message
        child_messages: Child messages
    """
    
    __tablename__ = "chat_messages"
    
    # =========================================================================
    # SESSION RELATIONSHIP
    # =========================================================================
    
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent chat session",
    )
    
    session = relationship(
        "ChatSession",
        back_populates="messages",
    )
    
    # =========================================================================
    # MESSAGE CONTENT
    # =========================================================================
    
    role = Column(
        Enum(MessageRole, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Message role (user, assistant, system, function)",
    )
    
    content = Column(
        Text,
        nullable=False,
        comment="Message content (text, markdown, json)",
    )
    
    content_type = Column(
        Enum(ContentType, native_enum=False, length=50),
        nullable=False,
        default=ContentType.TEXT,
        comment="Content format type",
    )
    
    # =========================================================================
    # MESSAGE THREADING
    # =========================================================================
    
    parent_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Parent message (for threading/branching)",
    )
    
    # Self-referential relationship
    parent_message = relationship(
        "ChatMessage",
        remote_side="ChatMessage.id",
        back_populates="child_messages",
        foreign_keys=[parent_message_id],
    )
    
    child_messages = relationship(
        "ChatMessage",
        back_populates="parent_message",
        foreign_keys=[parent_message_id],
    )
    
    sequence_number = Column(
        Integer,
        nullable=False,
        comment="Message order in conversation (1, 2, 3...)",
    )
    
    # =========================================================================
    # TOKEN USAGE
    # =========================================================================
    
    prompt_tokens = Column(
        Integer,
        nullable=True,
        comment="Input tokens used (for assistant messages)",
    )
    
    completion_tokens = Column(
        Integer,
        nullable=True,
        comment="Output tokens generated (for assistant messages)",
    )
    
    total_tokens = Column(
        Integer,
        nullable=True,
        comment="Total tokens (prompt + completion)",
    )
    
    # =========================================================================
    # AI CONFIGURATION
    # =========================================================================
    
    model = Column(
        String(100),
        nullable=True,
        comment="AI model used for this message",
    )
    
    temperature = Column(
        Float,
        nullable=True,
        comment="Model temperature used",
    )
    
    # =========================================================================
    # PROCESSING STATUS
    # =========================================================================
    
    processing_status = Column(
        Enum(MessageStatus, native_enum=False, length=50),
        nullable=False,
        default=MessageStatus.COMPLETED,
        index=True,
        comment="Message processing status",
    )
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if processing failed",
    )
    
    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================
    
    response_time_ms = Column(
        Integer,
        nullable=True,
        comment="Response generation time in milliseconds",
    )
    
    # =========================================================================
    # CITATIONS & REFERENCES
    # =========================================================================
    
    citations = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Document citations with page numbers and quotes",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (function calls, tool usage, etc.)",
    )
    
    # =========================================================================
    # FLAGS
    # =========================================================================
    
    is_edited = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Message was edited by user",
    )
    
    is_regenerated = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Response was regenerated",
    )
    
    # =========================================================================
    # USER FEEDBACK
    # =========================================================================
    
    feedback_rating = Column(
        Integer,
        nullable=True,
        comment="User feedback rating (1-5)",
    )
    
    feedback_comment = Column(
        Text,
        nullable=True,
        comment="User feedback comment",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for session messages (ordered)
        Index(
            "ix_chat_messages_session_seq",
            "session_id",
            "sequence_number",
        ),
        
        # Index for user messages
        Index(
            "ix_chat_messages_user",
            "session_id",
            "role",
            postgresql_where="role = 'user'",
        ),
        
        # Index for assistant messages
        Index(
            "ix_chat_messages_assistant",
            "session_id",
            "role",
            postgresql_where="role = 'assistant'",
        ),
        
        # Index for processing status
        Index(
            "ix_chat_messages_processing",
            "processing_status",
            postgresql_where="processing_status IN ('pending', 'processing')",
        ),
        
        # Check: sequence number positive
        CheckConstraint(
            "sequence_number > 0",
            name="ck_chat_messages_sequence_positive",
        ),
        
        # Check: token counts non-negative
        CheckConstraint(
            "(prompt_tokens IS NULL OR prompt_tokens >= 0) AND "
            "(completion_tokens IS NULL OR completion_tokens >= 0) AND "
            "(total_tokens IS NULL OR total_tokens >= 0)",
            name="ck_chat_messages_token_counts",
        ),
        
        # Check: feedback rating range
        CheckConstraint(
            "feedback_rating IS NULL OR (feedback_rating >= 1 AND feedback_rating <= 5)",
            name="ck_chat_messages_feedback_range",
        ),
        
        # Check: response time non-negative
        CheckConstraint(
            "response_time_ms IS NULL OR response_time_ms >= 0",
            name="ck_chat_messages_response_time",
        ),
    )
    
    # =========================================================================
    # MESSAGE CREATION
    # =========================================================================
    
    @classmethod
    def create_user_message(
        cls,
        session_id: UUIDType,
        content: str,
        tenant_id: UUIDType,
        sequence_number: int,
        parent_message_id: UUIDType | None = None,
    ) -> "ChatMessage":
        """
        Create a user message.
        
        Args:
            session_id: Chat session UUID
            content: Message content
            tenant_id: Tenant UUID
            sequence_number: Message sequence
            parent_message_id: Parent message UUID (optional)
            
        Returns:
            ChatMessage: New user message
            
        Example:
            >>> msg = ChatMessage.create_user_message(
            ...     session_id=session.id,
            ...     content="Bu sözleşme geçerli mi?",
            ...     tenant_id=tenant.id,
            ...     sequence_number=1
            ... )
        """
        message = cls(
            session_id=session_id,
            tenant_id=tenant_id,
            role=MessageRole.USER,
            content=content,
            sequence_number=sequence_number,
            parent_message_id=parent_message_id,
            processing_status=MessageStatus.COMPLETED,
        )
        
        logger.info(
            "User message created",
            message_id=str(message.id),
            session_id=str(session_id),
            sequence=sequence_number,
        )
        
        return message
    
    @classmethod
    def create_assistant_message(
        cls,
        session_id: UUIDType,
        content: str,
        tenant_id: UUIDType,
        sequence_number: int,
        parent_message_id: UUIDType | None = None,
        model: str | None = None,
        temperature: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        response_time_ms: int | None = None,
    ) -> "ChatMessage":
        """
        Create an assistant message.
        
        Args:
            session_id: Chat session UUID
            content: AI response content
            tenant_id: Tenant UUID
            sequence_number: Message sequence
            parent_message_id: Parent message UUID (usually user question)
            model: AI model used
            temperature: Temperature used
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            response_time_ms: Generation time
            
        Returns:
            ChatMessage: New assistant message
            
        Example:
            >>> msg = ChatMessage.create_assistant_message(
            ...     session_id=session.id,
            ...     content="Evet, bu sözleşme TBK'ya uygun...",
            ...     tenant_id=tenant.id,
            ...     sequence_number=2,
            ...     parent_message_id=user_msg.id,
            ...     model="claude-sonnet-4.5",
            ...     prompt_tokens=500,
            ...     completion_tokens=300,
            ...     response_time_ms=2500
            ... )
        """
        # Calculate total tokens
        total = None
        if prompt_tokens is not None and completion_tokens is not None:
            total = prompt_tokens + completion_tokens
        
        message = cls(
            session_id=session_id,
            tenant_id=tenant_id,
            role=MessageRole.ASSISTANT,
            content=content,
            sequence_number=sequence_number,
            parent_message_id=parent_message_id,
            model=model,
            temperature=temperature,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            response_time_ms=response_time_ms,
            processing_status=MessageStatus.COMPLETED,
        )
        
        logger.info(
            "Assistant message created",
            message_id=str(message.id),
            session_id=str(session_id),
            sequence=sequence_number,
            tokens=total,
            response_time_ms=response_time_ms,
        )
        
        return message
    
    # =========================================================================
    # CITATION MANAGEMENT
    # =========================================================================
    
    def add_citation(
        self,
        document_id: UUIDType,
        page_numbers: list[int] | None = None,
        quote: str | None = None,
        relevance_score: float | None = None,
    ) -> None:
        """
        Add a document citation to message.
        
        Args:
            document_id: Document UUID
            page_numbers: Cited page numbers
            quote: Direct quote from document
            relevance_score: Relevance score (0.0-1.0)
            
        Example:
            >>> msg.add_citation(
            ...     document_id=doc.id,
            ...     page_numbers=[5, 6],
            ...     quote="Madde 5.2: Fesih koşulları...",
            ...     relevance_score=0.95
            ... )
        """
        citation = {
            "document_id": str(document_id),
            "page_numbers": page_numbers or [],
            "quote": quote,
            "relevance_score": relevance_score,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Initialize citations if needed
        if not isinstance(self.citations, list):
            self.citations = []
        
        self.citations.append(citation)
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "citations")
        
        logger.debug(
            "Citation added to message",
            message_id=str(self.id),
            document_id=str(document_id),
            pages=page_numbers,
        )
    
    def get_cited_documents(self) -> list[str]:
        """
        Get list of cited document IDs.
        
        Returns:
            list: Document UUIDs
            
        Example:
            >>> doc_ids = msg.get_cited_documents()
            >>> # ['123e4567-...', '234e5678-...']
        """
        if not self.citations:
            return []
        
        return [
            citation["document_id"]
            for citation in self.citations
            if "document_id" in citation
        ]
    
    def has_citations(self) -> bool:
        """Check if message has any citations."""
        return bool(self.citations) and len(self.citations) > 0
    
    # =========================================================================
    # MESSAGE EDITING
    # =========================================================================
    
    def edit_content(self, new_content: str) -> None:
        """
        Edit message content (user messages only).
        
        Args:
            new_content: New message content
            
        Raises:
            ValidationError: If not a user message
            
        Example:
            >>> user_msg.edit_content("Güncellenmiş soru: Bu sözleşme...")
        """
        if self.role != MessageRole.USER:
            raise ValidationError(
                message="Only user messages can be edited",
                field="content",
            )
        
        # Store original content in metadata
        if "edit_history" not in self.metadata:
            self.metadata["edit_history"] = []
        
        self.metadata["edit_history"].append({
            "original_content": self.content,
            "edited_at": datetime.now(timezone.utc).isoformat(),
        })
        
        self.content = new_content
        self.is_edited = True
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "metadata")
        
        logger.info(
            "Message content edited",
            message_id=str(self.id),
        )
    
    # =========================================================================
    # MESSAGE REGENERATION
    # =========================================================================
    
    def mark_as_regenerated(self) -> None:
        """Mark message as regenerated response."""
        self.is_regenerated = True
        
        if "regeneration_count" not in self.metadata:
            self.metadata["regeneration_count"] = 0
        
        self.metadata["regeneration_count"] += 1
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "metadata")
        
        logger.info(
            "Message marked as regenerated",
            message_id=str(self.id),
            count=self.metadata["regeneration_count"],
        )
    
    # =========================================================================
    # PROCESSING STATUS
    # =========================================================================
    
    def mark_processing(self) -> None:
        """Mark message as processing."""
        self.processing_status = MessageStatus.PROCESSING
        
        logger.debug(
            "Message processing started",
            message_id=str(self.id),
        )
    
    def mark_completed(self) -> None:
        """Mark message processing as completed."""
        self.processing_status = MessageStatus.COMPLETED
        
        logger.debug(
            "Message processing completed",
            message_id=str(self.id),
        )
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark message processing as failed.
        
        Args:
            error_message: Error description
        """
        self.processing_status = MessageStatus.FAILED
        self.error_message = error_message
        
        logger.error(
            "Message processing failed",
            message_id=str(self.id),
            error=error_message,
        )
    
    def mark_cancelled(self) -> None:
        """Mark message processing as cancelled."""
        self.processing_status = MessageStatus.CANCELLED
        
        logger.info(
            "Message processing cancelled",
            message_id=str(self.id),
        )
    
    # =========================================================================
    # USER FEEDBACK
    # =========================================================================
    
    def add_feedback(
        self,
        rating: int,
        comment: str | None = None,
    ) -> None:
        """
        Add user feedback to message.
        
        Args:
            rating: Rating 1-5 (1=poor, 5=excellent)
            comment: Optional feedback text
            
        Example:
            >>> assistant_msg.add_feedback(
            ...     rating=5,
            ...     comment="Çok yardımcı oldu, teşekkürler!"
            ... )
        """
        if not 1 <= rating <= 5:
            raise ValidationError(
                message="Rating must be between 1 and 5",
                field="feedback_rating",
            )
        
        self.feedback_rating = rating
        self.feedback_comment = comment
        
        # Store feedback timestamp
        if "feedback" not in self.metadata:
            self.metadata["feedback"] = {}
        
        self.metadata["feedback"]["submitted_at"] = datetime.now(timezone.utc).isoformat()
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "metadata")
        
        logger.info(
            "Feedback added to message",
            message_id=str(self.id),
            rating=rating,
        )
    
    # =========================================================================
    # BRANCHING
    # =========================================================================
    
    def get_branches(self) -> list["ChatMessage"]:
        """
        Get alternative responses (branches) for this message.
        
        Returns:
            list: Child messages (different AI responses)
            
        Example:
            >>> # Get alternative AI responses
            >>> branches = user_msg.get_branches()
            >>> for branch in branches:
            ...     print(f"Response {branch.sequence_number}: {branch.content[:50]}")
        """
        return list(self.child_messages) if self.child_messages else []
    
    def has_branches(self) -> bool:
        """Check if message has branches (multiple responses)."""
        return len(self.get_branches()) > 0
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("content")
    def validate_content(self, key: str, content: str) -> str:
        """Validate message content."""
        if not content or not content.strip():
            raise ValidationError(
                message="Message content cannot be empty",
                field="content",
            )
        
        return content.strip()
    
    @validates("sequence_number")
    def validate_sequence_number(self, key: str, sequence_number: int) -> int:
        """Validate sequence number."""
        if sequence_number <= 0:
            raise ValidationError(
                message="Sequence number must be positive",
                field="sequence_number",
            )
        
        return sequence_number
    
    @validates("temperature")
    def validate_temperature(self, key: str, temperature: float | None) -> float | None:
        """Validate temperature range."""
        if temperature is not None and not 0.0 <= temperature <= 1.0:
            raise ValidationError(
                message="Temperature must be between 0.0 and 1.0",
                field="temperature",
            )
        
        return temperature
    
    @validates("feedback_rating")
    def validate_feedback_rating(self, key: str, rating: int | None) -> int | None:
        """Validate feedback rating."""
        if rating is not None and not 1 <= rating <= 5:
            raise ValidationError(
                message="Feedback rating must be between 1 and 5",
                field="feedback_rating",
            )
        
        return rating
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_content_preview(self, max_length: int = 100) -> str:
        """
        Get content preview (truncated).
        
        Args:
            max_length: Maximum preview length
            
        Returns:
            str: Truncated content
        """
        if len(self.content) <= max_length:
            return self.content
        
        return self.content[:max_length] + "..."
    
    def get_token_cost(self) -> float:
        """
        Calculate estimated cost for this message.
        
        Returns:
            float: Cost in USD
        """
        if not self.prompt_tokens or not self.completion_tokens:
            return 0.0
        
        # Example pricing for Claude Sonnet 4.5
        input_price_per_million = 3.0
        output_price_per_million = 15.0
        
        input_cost = (self.prompt_tokens / 1_000_000) * input_price_per_million
        output_cost = (self.completion_tokens / 1_000_000) * output_price_per_million
        
        return round(input_cost + output_cost, 6)
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ChatMessage("
            f"id={self.id}, "
            f"role={self.role}, "
            f"seq={self.sequence_number}"
            f")>"
        )
    
    def to_dict(self, include_citations: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_citations: Include citation details
            
        Returns:
            dict: Message data
        """
        data = super().to_dict()
        
        # Add computed fields
        data["role_display"] = self.role.display_name_tr
        data["content_preview"] = self.get_content_preview()
        data["has_citations"] = self.has_citations()
        data["has_branches"] = self.has_branches()
        data["estimated_cost"] = self.get_token_cost()
        
        # Include citations if requested
        if include_citations:
            data["citation_count"] = len(self.citations) if self.citations else 0
        else:
            data.pop("citations", None)
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ChatMessage",
    "MessageRole",
    "MessageStatus",
    "ContentType",
]
