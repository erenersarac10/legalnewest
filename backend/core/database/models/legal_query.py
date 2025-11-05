"""
Legal Query model for AI-powered legal research in Turkish Legal AI.

This module provides the LegalQuery model for tracking legal research queries:
- Legal research questions (Turkish law)
- AI-powered legal analysis
- Case law search
- Statute research
- Legal precedent tracking
- Multi-source legal research
- Citation tracking
- Query history and reuse

Query Types:
    - GENERAL: General legal question
    - CASE_LAW: Case law research (Yargıtay, Danıştay)
    - STATUTE: Statute/law research (Kanun araştırması)
    - PRECEDENT: Legal precedent research
    - INTERPRETATION: Law interpretation question
    - COMPLIANCE: Compliance check (KVKK, TBK, etc.)
    - COMPARISON: Comparative law analysis
    - PROCEDURE: Legal procedure question

Legal Sources (Turkish):
    - Yargıtay (Supreme Court of Appeals)
    - Danıştay (Council of State)
    - Anayasa Mahkemesi (Constitutional Court)
    - TBK (Turkish Code of Obligations)
    - TTK (Turkish Commercial Code)
    - KVKK (Data Protection Law)
    - İş Kanunu (Labor Law)

Example:
    >>> # User asks legal question
    >>> query = LegalQuery.create_query(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     query_type=QueryType.CASE_LAW,
    ...     question="İşten çıkarma için geçerli sebepler nelerdir?",
    ...     context={"industry": "tech", "position": "senior"}
    ... )
    >>> 
    >>> # AI processes query
    >>> query.complete(
    ...     answer="İş Kanunu Madde 25/II'ye göre...",
    ...     sources=["Yargıtay 9. HD., 2023/4567"],
    ...     confidence_score=0.92
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


class QueryType(str, enum.Enum):
    """
    Legal query type classification.
    
    Types:
    - GENERAL: General legal question
    - CASE_LAW: Case law research (Yargıtay, Danıştay)
    - STATUTE: Statute/legislation research
    - PRECEDENT: Legal precedent analysis
    - INTERPRETATION: Law interpretation question
    - COMPLIANCE: Compliance check
    - COMPARISON: Comparative law
    - PROCEDURE: Legal procedure
    - DEFINITION: Legal term definition
    """
    
    GENERAL = "general"
    CASE_LAW = "case_law"
    STATUTE = "statute"
    PRECEDENT = "precedent"
    INTERPRETATION = "interpretation"
    COMPLIANCE = "compliance"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    DEFINITION = "definition"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.GENERAL: "Genel Hukuki Soru",
            self.CASE_LAW: "İçtihat Araştırması",
            self.STATUTE: "Kanun Araştırması",
            self.PRECEDENT: "Emsal Karar Araştırması",
            self.INTERPRETATION: "Yorum Sorusu",
            self.COMPLIANCE: "Uyumluluk Kontrolü",
            self.COMPARISON: "Karşılaştırmalı Hukuk",
            self.PROCEDURE: "Usul Sorusu",
            self.DEFINITION: "Terim Tanımı",
        }
        return names.get(self, self.value)


class QueryStatus(str, enum.Enum):
    """Query processing status."""
    
    PENDING = "pending"          # Queued for processing
    PROCESSING = "processing"    # AI analyzing
    COMPLETED = "completed"      # Successfully answered
    FAILED = "failed"            # Processing failed
    CANCELLED = "cancelled"      # User cancelled
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.PENDING: "Bekliyor",
            self.PROCESSING: "İşleniyor",
            self.COMPLETED: "Tamamlandı",
            self.FAILED: "Başarısız",
            self.CANCELLED: "İptal Edildi",
        }
        return names.get(self, self.value)


class LegalArea(str, enum.Enum):
    """Legal area/practice area."""
    
    CIVIL = "civil"                      # Medeni Hukuk
    COMMERCIAL = "commercial"            # Ticaret Hukuku
    LABOR = "labor"                      # İş Hukuku
    CRIMINAL = "criminal"                # Ceza Hukuku
    ADMINISTRATIVE = "administrative"    # İdare Hukuku
    TAX = "tax"                          # Vergi Hukuku
    FAMILY = "family"                    # Aile Hukuku
    PROPERTY = "property"                # Eşya Hukuku
    CONSUMER = "consumer"                # Tüketici Hukuku
    DATA_PROTECTION = "data_protection"  # Veri Koruma (KVKK)
    INTELLECTUAL_PROPERTY = "intellectual_property"  # Fikri Mülkiyet
    BANKRUPTCY = "bankruptcy"            # İflas ve İcra
    INTERNATIONAL = "international"      # Milletlerarası Hukuk
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.CIVIL: "Medeni Hukuk",
            self.COMMERCIAL: "Ticaret Hukuku",
            self.LABOR: "İş Hukuku",
            self.CRIMINAL: "Ceza Hukuku",
            self.ADMINISTRATIVE: "İdare Hukuku",
            self.TAX: "Vergi Hukuku",
            self.FAMILY: "Aile Hukuku",
            self.PROPERTY: "Eşya Hukuku",
            self.CONSUMER: "Tüketici Hukuku",
            self.DATA_PROTECTION: "Veri Koruma Hukuku",
            self.INTELLECTUAL_PROPERTY: "Fikri Mülkiyet Hukuku",
            self.BANKRUPTCY: "İflas ve İcra Hukuku",
            self.INTERNATIONAL: "Milletlerarası Hukuk",
        }
        return names.get(self, self.value)


# =============================================================================
# LEGAL QUERY MODEL
# =============================================================================


class LegalQuery(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Legal Query model for AI-powered legal research.
    
    Tracks legal research queries:
    - User questions
    - AI-generated answers
    - Source citations
    - Confidence scoring
    - Follow-up queries
    
    Query Lifecycle:
    1. User submits legal question
    2. System processes with AI (RAG + legal databases)
    3. AI generates answer with citations
    4. User reviews and rates answer
    5. Optional: Follow-up questions
    
    AI Research Process:
        - Query analysis (intent, entities, legal area)
        - Multi-source search (case law, statutes, precedents)
        - RAG retrieval (relevant legal documents)
        - Answer generation with citations
        - Confidence scoring
    
    Attributes:
        user_id: User asking question
        user: User relationship
        
        query_type: Type of legal query
        legal_area: Legal practice area
        
        question: User's question (original)
        question_normalized: Normalized/cleaned question
        
        context: Additional context (JSON)
        
        status: Processing status
        
        answer: AI-generated answer
        answer_summary: Brief answer summary
        
        sources: Legal sources cited (array)
        citations: Detailed citation information (JSON)
        
        confidence_score: Answer confidence (0-1)
        
        model_used: AI model identifier
        model_version: Model version
        
        search_strategy: Search strategy used (JSON)
        
        started_at: Processing start timestamp
        completed_at: Processing completion timestamp
        processing_time_seconds: Duration
        
        token_count: Tokens used
        estimated_cost: Processing cost
        
        user_rating: User satisfaction (1-5)
        user_feedback: User feedback text
        
        is_helpful: User marked as helpful
        
        related_queries: Related query IDs (array)
        parent_query_id: Parent query (for follow-ups)
        parent_query: Parent relationship
        
        metadata: Additional context
        
        error_message: Error details if failed
        
    Relationships:
        tenant: Parent tenant
        user: User who asked question
        parent_query: Parent query (for follow-ups)
    """
    
    __tablename__ = "legal_queries"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who asked the legal question",
    )
    
    user = relationship(
        "User",
        back_populates="legal_queries",
    )
    
    # =========================================================================
    # QUERY CLASSIFICATION
    # =========================================================================
    
    query_type = Column(
        Enum(QueryType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of legal query",
    )
    
    legal_area = Column(
        Enum(LegalArea, native_enum=False, length=50),
        nullable=True,
        index=True,
        comment="Legal practice area",
    )
    
    # =========================================================================
    # QUESTION
    # =========================================================================
    
    question = Column(
        Text,
        nullable=False,
        comment="User's legal question (original)",
    )
    
    question_normalized = Column(
        Text,
        nullable=True,
        comment="Normalized/cleaned question for processing",
    )
    
    context = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (industry, position, prior knowledge, etc.)",
    )
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    status = Column(
        Enum(QueryStatus, native_enum=False, length=50),
        nullable=False,
        default=QueryStatus.PENDING,
        index=True,
        comment="Query processing status",
    )
    
    # =========================================================================
    # ANSWER
    # =========================================================================
    
    answer = Column(
        Text,
        nullable=True,
        comment="AI-generated answer with legal analysis",
    )
    
    answer_summary = Column(
        Text,
        nullable=True,
        comment="Brief answer summary (TL;DR)",
    )
    
    # =========================================================================
    # SOURCES & CITATIONS
    # =========================================================================
    
    sources = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Legal sources cited (Yargıtay kararları, kanun maddeleri, etc.)",
    )
    
    citations = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Detailed citation information (array of citation objects)",
    )
    
    # =========================================================================
    # CONFIDENCE
    # =========================================================================
    
    confidence_score = Column(
        Float,
        nullable=True,
        comment="Answer confidence score (0.0-1.0)",
    )
    
    # =========================================================================
    # AI MODEL INFORMATION
    # =========================================================================
    
    model_used = Column(
        String(100),
        nullable=True,
        comment="AI model used (claude-sonnet-4.5, etc.)",
    )
    
    model_version = Column(
        String(50),
        nullable=True,
        comment="Model version",
    )
    
    search_strategy = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Search strategy and parameters used",
    )
    
    # =========================================================================
    # TIMING
    # =========================================================================
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing started",
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When processing completed",
    )
    
    processing_time_seconds = Column(
        Integer,
        nullable=True,
        comment="Processing duration in seconds",
    )
    
    # =========================================================================
    # COST TRACKING
    # =========================================================================
    
    token_count = Column(
        Integer,
        nullable=True,
        comment="Total tokens used",
    )
    
    estimated_cost = Column(
        Float,
        nullable=True,
        comment="Estimated processing cost (USD)",
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
    
    is_helpful = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="User marked answer as helpful",
    )
    
    # =========================================================================
    # QUERY RELATIONSHIPS (Follow-ups)
    # =========================================================================
    
    related_queries = Column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list,
        comment="Related query IDs (similar queries)",
    )
    
    parent_query_id = Column(
        UUID(as_uuid=True),
        ForeignKey("legal_queries.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Parent query (if this is a follow-up)",
    )
    
    parent_query = relationship(
        "LegalQuery",
        remote_side="LegalQuery.id",
        backref="follow_up_queries",
        foreign_keys=[parent_query_id],
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (search results count, databases used, etc.)",
    )
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if processing failed",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user's queries
        Index(
            "ix_legal_queries_user",
            "user_id",
            "created_at",
        ),
        
        # Index for query type analytics
        Index(
            "ix_legal_queries_type",
            "tenant_id",
            "query_type",
            "legal_area",
        ),
        
        # Index for completed queries
        Index(
            "ix_legal_queries_completed",
            "status",
            "completed_at",
            postgresql_where="status = 'completed'",
        ),
        
        # Index for helpful queries (knowledge base)
        Index(
            "ix_legal_queries_helpful",
            "is_helpful",
            "user_rating",
            postgresql_where="is_helpful = true AND status = 'completed'",
        ),
        
        # Index for parent-child queries
        Index(
            "ix_legal_queries_parent",
            "parent_query_id",
            postgresql_where="parent_query_id IS NOT NULL",
        ),
        
        # Check: user rating range
        CheckConstraint(
            "user_rating IS NULL OR (user_rating >= 1 AND user_rating <= 5)",
            name="ck_legal_queries_rating",
        ),
        
        # Check: confidence score range
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)",
            name="ck_legal_queries_confidence",
        ),
        
        # Check: token count non-negative
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="ck_legal_queries_tokens",
        ),
    )
    
    # =========================================================================
    # QUERY CREATION
    # =========================================================================
    
    @classmethod
    def create_query(
        cls,
        user_id: UUIDType,
        tenant_id: UUIDType,
        query_type: QueryType,
        question: str,
        legal_area: LegalArea | None = None,
        context: dict[str, Any] | None = None,
        parent_query_id: UUIDType | None = None,
    ) -> "LegalQuery":
        """
        Create a new legal query.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant UUID
            query_type: Query type
            question: Legal question
            legal_area: Legal practice area
            context: Additional context
            parent_query_id: Parent query (for follow-ups)
            
        Returns:
            LegalQuery: New query instance
            
        Example:
            >>> query = LegalQuery.create_query(
            ...     user_id=user.id,
            ...     tenant_id=tenant.id,
            ...     query_type=QueryType.CASE_LAW,
            ...     question="İşten çıkarma için geçerli sebepler nelerdir?",
            ...     legal_area=LegalArea.LABOR,
            ...     context={
            ...         "industry": "technology",
            ...         "company_size": "50+",
            ...         "position": "senior_developer"
            ...     }
            ... )
        """
        query = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            query_type=query_type,
            question=question,
            legal_area=legal_area,
            context=context or {},
            parent_query_id=parent_query_id,
            status=QueryStatus.PENDING,
        )
        
        logger.info(
            "Legal query created",
            query_id=str(query.id),
            user_id=str(user_id),
            query_type=query_type.value,
            is_follow_up=parent_query_id is not None,
        )
        
        return query
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def start_processing(self, model: str, model_version: str | None = None) -> None:
        """
        Start processing query.
        
        Args:
            model: AI model identifier
            model_version: Model version
        """
        self.status = QueryStatus.PROCESSING
        self.started_at = datetime.now(timezone.utc)
        self.model_used = model
        self.model_version = model_version
        
        logger.info(
            "Legal query processing started",
            query_id=str(self.id),
            model=model,
        )
    
    def complete(
        self,
        answer: str,
        sources: list[str],
        confidence_score: float,
        answer_summary: str | None = None,
        citations: list[dict[str, Any]] | None = None,
        token_count: int | None = None,
    ) -> None:
        """
        Complete query with answer.
        
        Args:
            answer: AI-generated answer
            sources: Legal sources cited
            confidence_score: Confidence score
            answer_summary: Brief summary
            citations: Detailed citations
            token_count: Tokens used
            
        Example:
            >>> query.complete(
            ...     answer="İş Kanunu Madde 25/II'ye göre...",
            ...     sources=[
            ...         "Yargıtay 9. HD., 2023/4567",
            ...         "İş Kanunu Madde 25/II"
            ...     ],
            ...     confidence_score=0.92,
            ...     answer_summary="Geçerli sebepler: performans, devamsızlık...",
            ...     citations=[
            ...         {
            ...             "source": "Yargıtay 9. HD., 2023/4567",
            ...             "text": "İşçinin performans düşüklüğü...",
            ...             "relevance": 0.95
            ...         }
            ...     ],
            ...     token_count=1500
            ... )
        """
        self.status = QueryStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.answer = answer
        self.answer_summary = answer_summary
        self.sources = sources
        self.citations = citations or []
        self.confidence_score = confidence_score
        self.token_count = token_count
        
        # Calculate processing time
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.processing_time_seconds = int(delta.total_seconds())
        
        logger.info(
            "Legal query completed",
            query_id=str(self.id),
            sources_count=len(sources),
            confidence_score=confidence_score,
            processing_time_seconds=self.processing_time_seconds,
        )
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark query as failed.
        
        Args:
            error_message: Error description
        """
        self.status = QueryStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        
        logger.error(
            "Legal query failed",
            query_id=str(self.id),
            error=error_message,
        )
    
    def cancel(self) -> None:
        """Cancel query processing."""
        self.status = QueryStatus.CANCELLED
        
        logger.info(
            "Legal query cancelled",
            query_id=str(self.id),
        )
    
    # =========================================================================
    # USER FEEDBACK
    # =========================================================================
    
    def add_user_feedback(
        self,
        rating: int,
        feedback: str | None = None,
        is_helpful: bool = True,
    ) -> None:
        """
        Add user feedback.
        
        Args:
            rating: User rating (1-5)
            feedback: Feedback text
            is_helpful: Helpful flag
            
        Example:
            >>> query.add_user_feedback(
            ...     rating=5,
            ...     feedback="Çok kapsamlı ve yardımcı bir yanıt",
            ...     is_helpful=True
            ... )
        """
        self.user_rating = rating
        self.user_feedback = feedback
        self.is_helpful = is_helpful
        
        logger.info(
            "Feedback added to legal query",
            query_id=str(self.id),
            rating=rating,
            is_helpful=is_helpful,
        )
    
    # =========================================================================
    # CITATIONS
    # =========================================================================
    
    def add_citation(
        self,
        source: str,
        text: str | None = None,
        url: str | None = None,
        relevance: float | None = None,
    ) -> None:
        """
        Add a citation to the answer.
        
        Args:
            source: Source identifier
            text: Cited text excerpt
            url: Source URL
            relevance: Relevance score
            
        Example:
            >>> query.add_citation(
            ...     source="Yargıtay 9. HD., 2023/4567 E., 2023/5678 K.",
            ...     text="İşverenin fesih hakkı...",
            ...     relevance=0.95
            ... )
        """
        citation = {
            "source": source,
            "text": text,
            "url": url,
            "relevance": relevance,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if not isinstance(self.citations, list):
            self.citations = []
        
        self.citations.append(citation)
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "citations")
        
        logger.debug(
            "Citation added to query",
            query_id=str(self.id),
            source=source,
        )
    
    def get_primary_sources(self) -> list[str]:
        """
        Get primary legal sources (statutes, codes).
        
        Returns:
            list: Primary sources
        """
        primary_keywords = ["kanun", "madde", "tbk", "ttk", "kvkk", "anayasa"]
        
        primary = []
        for source in self.sources:
            if any(keyword in source.lower() for keyword in primary_keywords):
                primary.append(source)
        
        return primary
    
    def get_case_law_sources(self) -> list[str]:
        """
        Get case law sources (Yargıtay, Danıştay).
        
        Returns:
            list: Case law sources
        """
        case_keywords = ["yargıtay", "danıştay", "anayasa mahkemesi"]
        
        cases = []
        for source in self.sources:
            if any(keyword in source.lower() for keyword in case_keywords):
                cases.append(source)
        
        return cases
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("question")
    def validate_question(self, key: str, question: str) -> str:
        """Validate question."""
        if not question or not question.strip():
            raise ValidationError(
                message="Question cannot be empty",
                field="question",
            )
        
        return question.strip()
    
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
            f"<LegalQuery("
            f"id={self.id}, "
            f"type={self.query_type.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self, include_answer: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_answer: Include full answer (can be large)
            
        Returns:
            dict: Query data
        """
        data = super().to_dict()
        
        # Remove large answer by default for list views
        if not include_answer:
            data.pop("answer", None)
        
        # Add display names
        data["query_type_display"] = self.query_type.display_name_tr
        
        if self.legal_area:
            data["legal_area_display"] = self.legal_area.display_name_tr
        
        data["status_display"] = self.status.display_name_tr
        
        # Add computed fields
        data["sources_count"] = len(self.sources) if self.sources else 0
        data["citation_count"] = len(self.citations) if self.citations else 0
        data["primary_sources_count"] = len(self.get_primary_sources())
        data["case_law_count"] = len(self.get_case_law_sources())
        data["has_user_feedback"] = self.user_rating is not None
        data["is_follow_up"] = self.parent_query_id is not None
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "LegalQuery",
    "QueryType",
    "QueryStatus",
    "LegalArea",
]
