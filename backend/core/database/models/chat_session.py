"""
Chat Session model for conversational AI interactions in Turkish Legal AI.

This module provides the ChatSession model for managing AI chat conversations:
- Multi-turn conversation tracking
- Context window management
- RAG integration (document references)
- Token usage tracking
- Conversation branching
- Export and sharing
- KVKK-compliant data handling
- Turkish language optimization

Chat Architecture:
    Session → Messages → Context → RAG Documents
    
    Each session contains:
    - Multiple messages (user ↔ assistant)
    - System prompt and settings
    - Referenced documents (RAG context)
    - Token usage metrics
    - Conversation metadata

Use Cases:
    - Legal Q&A with document context
    - Contract analysis conversations
    - Multi-document research sessions
    - Legal opinion drafting
    - Client consultation notes

Security & Compliance:
    - User-level access control
    - Team sharing support
    - Conversation encryption (optional)
    - Audit trail (KVKK)
    - Data retention policies
    - Export for compliance

Example:
    >>> # Create chat session
    >>> session = ChatSession(
    ...     title="İş Sözleşmesi İncelemesi",
    ...     user_id=user_id,
    ...     tenant_id=tenant_id,
    ...     system_prompt="Sen Türk hukuku uzmanı bir yapay zeka asistanısın.",
    ...     model="claude-sonnet-4.5"
    ... )
    >>> 
    >>> # Add document context
    >>> session.add_document(document_id)
    >>> 
    >>> # Track tokens
    >>> session.add_token_usage(
    ...     prompt_tokens=500,
    ...     completion_tokens=300
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
    Numeric,
    String,
    Text,
    CheckConstraint,
    Index,
    Table,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func, text

from backend.core.constants import (
    DEFAULT_MODEL,
    MAX_CHAT_TITLE_LENGTH,
    MAX_CONTEXT_WINDOW_TOKENS,
    MAX_MESSAGES_PER_SESSION,
)
from backend.core.exceptions import (
    ChatLimitExceededError,
    TokenLimitExceededError,
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


class ChatStatus(str, enum.Enum):
    """Chat session status."""
    
    ACTIVE = "active"              # Currently active
    PAUSED = "paused"              # User paused conversation
    COMPLETED = "completed"        # Conversation finished
    ARCHIVED = "archived"          # Archived by user
    ERROR = "error"                # Error occurred
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.ACTIVE: "Aktif",
            self.PAUSED: "Duraklatıldı",
            self.COMPLETED: "Tamamlandı",
            self.ARCHIVED: "Arşivlendi",
            self.ERROR: "Hata",
        }
        return names.get(self, self.value)


class ChatMode(str, enum.Enum):
    """Chat interaction mode."""
    
    GENERAL = "general"            # General Q&A
    DOCUMENT = "document"          # Document-focused (RAG)
    ANALYSIS = "analysis"          # Analysis mode
    DRAFTING = "drafting"          # Document drafting
    RESEARCH = "research"          # Legal research
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.GENERAL: "Genel Sohbet",
            self.DOCUMENT: "Belge İncelemesi",
            self.ANALYSIS: "Analiz",
            self.DRAFTING: "Belge Hazırlama",
            self.RESEARCH: "Araştırma",
        }
        return names.get(self, self.value)


# =============================================================================
# ASSOCIATION TABLE (Many-to-Many: ChatSession ↔ Document)
# =============================================================================

chat_session_documents = Table(
    "chat_session_documents",
    Base.metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        comment="Unique identifier",
    ),
    Column(
        "chat_session_id",
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Chat session ID",
    ),
    Column(
        "document_id",
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Document ID",
    ),
    Column(
        "added_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When document was added to session",
    ),
    Column(
        "relevance_score",
        Numeric(3, 2),
        nullable=True,
        comment="Document relevance score (0.00-1.00)",
    ),
    
    # Index for document's chat sessions
    Index("ix_chat_session_documents_document", "document_id"),
)


# =============================================================================
# CHAT SESSION MODEL
# =============================================================================


class ChatSession(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Chat Session model for AI conversation management.
    
    Chat sessions represent conversations between users and AI:
    - Multi-turn conversations
    - Document context (RAG)
    - Token usage tracking
    - Conversation history
    - Team sharing support
    
    Context Management:
        - System prompt (role definition)
        - Document context (RAG retrieval)
        - Conversation history (sliding window)
        - Token limit enforcement
    
    Token Tracking:
        - Per-message token counts
        - Cumulative session usage
        - Cost calculation
        - Quota enforcement
    
    Attributes:
        title: Conversation title
        description: Session description
        
        user_id: Session owner
        team_id: Shared with team (optional)
        
        status: Session status
        mode: Interaction mode
        
        system_prompt: AI system instructions
        model: AI model used (claude-sonnet-4.5)
        temperature: Model temperature (0.0-1.0)
        max_tokens: Max completion tokens
        
        message_count: Number of messages
        total_prompt_tokens: Total input tokens
        total_completion_tokens: Total output tokens
        total_tokens: Total tokens used
        estimated_cost: Estimated API cost
        
        last_message_at: Last message timestamp
        last_assistant_message: Last AI response preview
        
        context_window_tokens: Current context size
        documents: Referenced documents (many-to-many)
        
        settings: Session-specific settings
        metadata: Additional metadata
        
        is_pinned: Pinned by user
        is_shared: Shared with team
        
    Relationships:
        tenant: Parent tenant
        user: Session owner
        team: Shared team (optional)
        messages: Chat messages
        documents: Referenced documents
    """
    
    __tablename__ = "chat_sessions"
    
    # =========================================================================
    # IDENTITY
    # =========================================================================
    
    title = Column(
        String(MAX_CHAT_TITLE_LENGTH),
        nullable=False,
        comment="Conversation title (auto-generated or user-defined)",
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Session description or summary",
    )
    
    # =========================================================================
    # OWNERSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session owner",
    )
    
    user = relationship(
        "User",
        back_populates="chat_sessions",
    )
    
    team_id = Column(
        UUID(as_uuid=True),
        ForeignKey("teams.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Shared with team (optional)",
    )
    
    team = relationship(
        "Team",
        back_populates="chat_sessions",
    )
    
    # =========================================================================
    # STATUS & MODE
    # =========================================================================
    
    status = Column(
        Enum(ChatStatus, native_enum=False, length=50),
        nullable=False,
        default=ChatStatus.ACTIVE,
        index=True,
        comment="Session status",
    )
    
    mode = Column(
        Enum(ChatMode, native_enum=False, length=50),
        nullable=False,
        default=ChatMode.GENERAL,
        comment="Chat interaction mode",
    )
    
    # =========================================================================
    # AI CONFIGURATION
    # =========================================================================
    
    system_prompt = Column(
        Text,
        nullable=True,
        comment="AI system instructions (role, behavior, constraints)",
    )
    
    model = Column(
        String(100),
        nullable=False,
        default=DEFAULT_MODEL,
        comment="AI model identifier (claude-sonnet-4.5, gpt-4, etc.)",
    )
    
    temperature = Column(
        Numeric(3, 2),
        nullable=False,
        default=0.7,
        comment="Model temperature (0.0-1.0, higher = more creative)",
    )
    
    max_tokens = Column(
        Integer,
        nullable=True,
        comment="Maximum completion tokens per message",
    )
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    message_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of messages in session",
    )
    
    total_prompt_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total input tokens used",
    )
    
    total_completion_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total output tokens used",
    )
    
    total_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total tokens (prompt + completion)",
    )
    
    estimated_cost = Column(
        Numeric(10, 4),
        nullable=False,
        default=0.0,
        comment="Estimated API cost (USD)",
    )
    
    # =========================================================================
    # ACTIVITY TRACKING
    # =========================================================================
    
    last_message_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Timestamp of last message",
    )
    
    last_assistant_message = Column(
        Text,
        nullable=True,
        comment="Preview of last AI response (first 500 chars)",
    )
    
    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================
    
    context_window_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Current context window size (tokens)",
    )
    
    # =========================================================================
    # SETTINGS & METADATA
    # =========================================================================
    
    settings = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Session-specific settings (streaming, citations, etc.)",
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (tags, categories, etc.)",
    )
    
    # =========================================================================
    # FLAGS
    # =========================================================================
    
    is_pinned = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Pinned by user (quick access)",
    )
    
    is_shared = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Shared with team",
    )
    
    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================
    
    # Messages relationship
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="dynamic",
    )
    
    # Documents relationship (many-to-many)
    documents = relationship(
        "Document",
        secondary=chat_session_documents,
        back_populates="chat_sessions",
        lazy="dynamic",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user's active sessions
        Index(
            "ix_chat_sessions_user_active",
            "user_id",
            "status",
            postgresql_where="status = 'active' AND deleted_at IS NULL",
        ),
        
        # Index for recent sessions
        Index(
            "ix_chat_sessions_recent",
            "user_id",
            "last_message_at",
        ),
        
        # Index for team sessions
        Index(
            "ix_chat_sessions_team",
            "team_id",
            "is_shared",
            postgresql_where="is_shared = true",
        ),
        
        # Index for pinned sessions
        Index(
            "ix_chat_sessions_pinned",
            "user_id",
            "is_pinned",
            postgresql_where="is_pinned = true",
        ),
        
        # Check: message count non-negative
        CheckConstraint(
            "message_count >= 0",
            name="ck_chat_sessions_message_count",
        ),
        
        # Check: token counts non-negative
        CheckConstraint(
            "total_prompt_tokens >= 0 AND total_completion_tokens >= 0 AND total_tokens >= 0",
            name="ck_chat_sessions_token_counts",
        ),
        
        # Check: temperature range
        CheckConstraint(
            "temperature >= 0.0 AND temperature <= 1.0",
            name="ck_chat_sessions_temperature_range",
        ),
        
        # Check: estimated cost non-negative
        CheckConstraint(
            "estimated_cost >= 0",
            name="ck_chat_sessions_cost",
        ),
    )
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    @classmethod
    def create_session(
        cls,
        user_id: UUIDType,
        tenant_id: UUIDType,
        title: str | None = None,
        mode: ChatMode = ChatMode.GENERAL,
        system_prompt: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> "ChatSession":
        """
        Create a new chat session.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant UUID
            title: Session title (auto-generated if None)
            mode: Chat mode
            system_prompt: AI system instructions
            model: AI model to use
            
        Returns:
            ChatSession: New session instance
            
        Example:
            >>> session = ChatSession.create_session(
            ...     user_id=user.id,
            ...     tenant_id=user.tenant_id,
            ...     title="İş Sözleşmesi Danışma",
            ...     mode=ChatMode.DOCUMENT,
            ...     system_prompt="Sen Türk hukuku uzmanı bir yapay zeka asistanısın."
            ... )
        """
        # Generate title if not provided
        if not title:
            now = datetime.now(timezone.utc)
            title = f"Yeni Sohbet - {now.strftime('%d.%m.%Y %H:%M')}"
        
        # Default system prompt for Turkish legal AI
        if not system_prompt:
            system_prompt = (
                "Sen Türk hukuku konusunda uzmanlaşmış bir yapay zeka asistanısın. "
                "Kullanıcılara hukuki sorularında yardımcı olursun. "
                "Yanıtlarını net, anlaşılır ve profesyonel bir dille verirsin."
            )
        
        session = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            title=title,
            mode=mode,
            system_prompt=system_prompt,
            model=model,
            status=ChatStatus.ACTIVE,
        )
        
        logger.info(
            "Chat session created",
            session_id=str(session.id),
            user_id=str(user_id),
            mode=mode.value,
            model=model,
        )
        
        return session
    
    def can_add_message(self) -> bool:
        """
        Check if session can accept more messages.
        
        Returns:
            bool: True if within limit
        """
        return self.message_count < MAX_MESSAGES_PER_SESSION
    
    def require_can_add_message(self) -> None:
        """
        Require session can accept more messages (raises if not).
        
        Raises:
            ChatLimitExceededError: If message limit reached
        """
        if not self.can_add_message():
            raise ChatLimitExceededError(
                message=f"Mesaj limiti aşıldı. Maksimum: {MAX_MESSAGES_PER_SESSION}",
                limit=MAX_MESSAGES_PER_SESSION,
                current=self.message_count,
            )
    
    # =========================================================================
    # TOKEN TRACKING
    # =========================================================================
    
    def add_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """
        Track token usage for a message.
        
        Args:
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens generated
            
        Example:
            >>> session.add_token_usage(
            ...     prompt_tokens=500,
            ...     completion_tokens=300
            ... )
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        
        # Update context window
        self.context_window_tokens = self._calculate_context_window()
        
        # Update estimated cost
        self.estimated_cost = self._calculate_cost()
        
        logger.debug(
            "Token usage tracked",
            session_id=str(self.id),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=self.total_tokens,
        )
    
    def _calculate_context_window(self) -> int:
        """
        Calculate current context window size.
        
        Includes:
        - System prompt tokens
        - Recent message tokens (sliding window)
        - Document context tokens
        
        Returns:
            int: Context window size in tokens
        """
        # Simplified calculation
        # Production would calculate actual token counts
        
        # System prompt (~200 tokens)
        system_tokens = 200
        
        # Recent messages (last 10 messages)
        # Approximation: average 150 tokens per message
        recent_messages = min(10, self.message_count)
        message_tokens = recent_messages * 150
        
        # Document context (if any)
        # Would query actual document token counts
        document_tokens = 0
        
        total = system_tokens + message_tokens + document_tokens
        
        return total
    
    def _calculate_cost(self) -> float:
        """
        Calculate estimated API cost.
        
        Pricing (example for Claude Sonnet 4.5):
        - Input: $3 per 1M tokens
        - Output: $15 per 1M tokens
        
        Returns:
            float: Estimated cost in USD
        """
        # Pricing per 1M tokens
        input_price_per_million = 3.0
        output_price_per_million = 15.0
        
        input_cost = (self.total_prompt_tokens / 1_000_000) * input_price_per_million
        output_cost = (self.total_completion_tokens / 1_000_000) * output_price_per_million
        
        return round(input_cost + output_cost, 4)
    
    def check_token_limit(self) -> bool:
        """
        Check if context window is within limit.
        
        Returns:
            bool: True if within limit
        """
        return self.context_window_tokens <= MAX_CONTEXT_WINDOW_TOKENS
    
    # =========================================================================
    # DOCUMENT MANAGEMENT
    # =========================================================================
    
    def add_document(
        self,
        document_id: UUIDType,
        relevance_score: float | None = None,
    ) -> None:
        """
        Add a document to session context.
        
        Args:
            document_id: Document UUID
            relevance_score: Relevance score (0.0-1.0)
            
        Example:
            >>> session.add_document(
            ...     document_id=doc.id,
            ...     relevance_score=0.95
            ... )
        """
        # This would be implemented in service layer with DB session
        # to insert into chat_session_documents table
        
        logger.info(
            "Document added to chat session",
            session_id=str(self.id),
            document_id=str(document_id),
            relevance_score=relevance_score,
        )
    
    def remove_document(self, document_id: UUIDType) -> None:
        """Remove a document from session context."""
        logger.info(
            "Document removed from chat session",
            session_id=str(self.id),
            document_id=str(document_id),
        )
    
    def get_document_count(self) -> int:
        """
        Get number of documents in context.
        
        Returns:
            int: Document count
        """
        return self.documents.count()
    
    # =========================================================================
    # MESSAGE TRACKING
    # =========================================================================
    
    def increment_message_count(self) -> None:
        """Increment message count (called when message added)."""
        self.message_count += 1
        self.last_message_at = datetime.now(timezone.utc)
    
    def update_last_assistant_message(self, content: str) -> None:
        """
        Update last assistant message preview.
        
        Args:
            content: Assistant message content
        """
        # Store first 500 characters as preview
        self.last_assistant_message = content[:500] if content else None
    
    # =========================================================================
    # SESSION STATUS
    # =========================================================================
    
    def pause(self) -> None:
        """Pause conversation."""
        self.status = ChatStatus.PAUSED
        
        logger.info(
            "Chat session paused",
            session_id=str(self.id),
        )
    
    def resume(self) -> None:
        """Resume conversation."""
        if self.status == ChatStatus.PAUSED:
            self.status = ChatStatus.ACTIVE
            
            logger.info(
                "Chat session resumed",
                session_id=str(self.id),
            )
    
    def complete(self) -> None:
        """Mark conversation as completed."""
        self.status = ChatStatus.COMPLETED
        
        logger.info(
            "Chat session completed",
            session_id=str(self.id),
        )
    
    def archive(self) -> None:
        """Archive conversation."""
        self.status = ChatStatus.ARCHIVED
        
        logger.info(
            "Chat session archived",
            session_id=str(self.id),
        )
    
    def mark_error(self, error_message: str | None = None) -> None:
        """
        Mark session as error state.
        
        Args:
            error_message: Error description
        """
        self.status = ChatStatus.ERROR
        
        if error_message:
            if "error" not in self.metadata:
                self.metadata["error"] = {}
            self.metadata["error"]["message"] = error_message
            self.metadata["error"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "metadata")
        
        logger.error(
            "Chat session error",
            session_id=str(self.id),
            error=error_message,
        )
    
    # =========================================================================
    # SHARING
    # =========================================================================
    
    def share_with_team(self, team_id: UUIDType) -> None:
        """
        Share session with team.
        
        Args:
            team_id: Team UUID
        """
        self.team_id = team_id
        self.is_shared = True
        
        logger.info(
            "Chat session shared with team",
            session_id=str(self.id),
            team_id=str(team_id),
        )
    
    def unshare(self) -> None:
        """Remove team sharing."""
        self.team_id = None
        self.is_shared = False
        
        logger.info(
            "Chat session unshared",
            session_id=str(self.id),
        )
    
    # =========================================================================
    # PINNING
    # =========================================================================
    
    def pin(self) -> None:
        """Pin session for quick access."""
        self.is_pinned = True
        
        logger.debug(
            "Chat session pinned",
            session_id=str(self.id),
        )
    
    def unpin(self) -> None:
        """Unpin session."""
        self.is_pinned = False
        
        logger.debug(
            "Chat session unpinned",
            session_id=str(self.id),
        )
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_conversation(self, format: str = "json") -> dict[str, Any]:
        """
        Export conversation in specified format.
        
        Args:
            format: Export format (json, markdown, pdf)
            
        Returns:
            dict: Exported conversation data
            
        Example:
            >>> export_data = session.export_conversation(format="json")
            >>> with open("conversation.json", "w") as f:
            ...     json.dump(export_data, f)
        """
        messages = []
        for msg in self.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
            })
        
        export_data = {
            "session_id": str(self.id),
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "message_count": self.message_count,
            "messages": messages,
            "statistics": {
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "estimated_cost": float(self.estimated_cost),
            },
        }
        
        logger.info(
            "Chat session exported",
            session_id=str(self.id),
            format=format,
        )
        
        return export_data
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("title")
    def validate_title(self, key: str, title: str) -> str:
        """Validate title."""
        if not title or not title.strip():
            raise ValidationError(
                message="Başlık boş olamaz",
                field="title",
            )
        
        return title.strip()
    
    @validates("temperature")
    def validate_temperature(self, key: str, temperature: float) -> float:
        """Validate temperature range."""
        if not 0.0 <= temperature <= 1.0:
            raise ValidationError(
                message="Temperature 0.0 ile 1.0 arasında olmalıdır",
                field="temperature",
            )
        
        return temperature
    
    @validates("model")
    def validate_model(self, key: str, model: str) -> str:
        """Validate model identifier."""
        # List of supported models
        supported_models = [
            "claude-sonnet-4.5",
            "claude-sonnet-4",
            "claude-opus-4",
            "gpt-4",
            "gpt-4-turbo",
        ]
        
        if model not in supported_models:
            logger.warning(
                "Unsupported model specified",
                model=model,
                supported=supported_models,
            )
        
        return model
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ChatSession("
            f"id={self.id}, "
            f"title={self.title}, "
            f"messages={self.message_count}"
            f")>"
        )
    
    def to_dict(self, include_messages: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_messages: Include message list (default: False)
            
        Returns:
            dict: Session data
        """
        data = super().to_dict()
        
        # Add computed fields
        data["status_display"] = self.status.display_name_tr
        data["mode_display"] = self.mode.display_name_tr
        data["document_count"] = self.get_document_count()
        data["cost_usd"] = float(self.estimated_cost)
        
        # Add time since last message
        if self.last_message_at:
            delta = datetime.now(timezone.utc) - self.last_message_at
            data["minutes_since_last_message"] = int(delta.total_seconds() / 60)
        
        # Include messages if requested
        if include_messages:
            data["messages"] = [
                msg.to_dict() for msg in self.messages
            ]
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ChatSession",
    "ChatStatus",
    "ChatMode",
    "chat_session_documents",
]
