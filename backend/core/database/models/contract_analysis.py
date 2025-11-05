"""
Contract Analysis model for AI-powered contract analysis in Turkish Legal AI.

This module provides the ContractAnalysis model for storing AI analysis results:
- Contract risk assessment
- Clause extraction and classification
- Compliance checking (Turkish law)
- Missing clause detection
- Term analysis and recommendations
- Multi-version comparison
- Legal issue flagging
- Custom analysis requests

Analysis Types:
    - FULL: Complete contract analysis
    - RISK: Risk assessment only
    - COMPLIANCE: Compliance check
    - CLAUSE: Clause extraction
    - COMPARISON: Multi-contract comparison
    - CUSTOM: Custom analysis request

Risk Levels:
    - LOW: Minor issues, non-critical
    - MEDIUM: Notable issues, review recommended
    - HIGH: Significant risks, action required
    - CRITICAL: Severe issues, immediate attention

Turkish Law Focus:
    - TBK (Turkish Code of Obligations)
    - TTK (Turkish Commercial Code)
    - İş Kanunu (Labor Law)
    - Tüketici Kanunu (Consumer Protection Law)
    - KVKK (Data Protection Law)

Example:
    >>> # Start contract analysis
    >>> analysis = ContractAnalysis.create_analysis(
    ...     document_id=contract.id,
    ...     tenant_id=tenant.id,
    ...     initiated_by_id=user.id,
    ...     analysis_type=AnalysisType.FULL,
    ...     parameters={"focus": "risk", "depth": "detailed"}
    ... )
    >>> 
    >>> # Complete analysis with results
    >>> analysis.complete(
    ...     findings={
    ...         "risk_score": 65,
    ...         "issues": [...],
    ...         "recommendations": [...]
    ...     },
    ...     risk_level=RiskLevel.MEDIUM
    ... )
"""

import enum
from datetime import datetime, timezone
from typing import Any
from uuid import UUID as UUIDType

from sqlalchemy import (
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


class AnalysisType(str, enum.Enum):
    """
    Contract analysis type.
    
    Types:
    - FULL: Complete contract analysis (all aspects)
    - RISK: Risk assessment only
    - COMPLIANCE: Compliance checking
    - CLAUSE: Clause extraction and classification
    - COMPARISON: Multi-contract comparison
    - TERMS: Payment terms and conditions
    - OBLIGATIONS: Rights and obligations analysis
    - CUSTOM: Custom analysis request
    """
    
    FULL = "full"
    RISK = "risk"
    COMPLIANCE = "compliance"
    CLAUSE = "clause"
    COMPARISON = "comparison"
    TERMS = "terms"
    OBLIGATIONS = "obligations"
    CUSTOM = "custom"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.FULL: "Tam Analiz",
            self.RISK: "Risk Değerlendirmesi",
            self.COMPLIANCE: "Uyumluluk Kontrolü",
            self.CLAUSE: "Madde Analizi",
            self.COMPARISON: "Karşılaştırmalı Analiz",
            self.TERMS: "Koşullar Analizi",
            self.OBLIGATIONS: "Yükümlülükler Analizi",
            self.CUSTOM: "Özel Analiz",
        }
        return names.get(self, self.value)


class AnalysisStatus(str, enum.Enum):
    """Analysis processing status."""
    
    PENDING = "pending"          # Queued for processing
    PROCESSING = "processing"    # Currently being analyzed
    COMPLETED = "completed"      # Successfully completed
    FAILED = "failed"            # Analysis failed
    CANCELLED = "cancelled"      # Cancelled by user
    
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


class RiskLevel(str, enum.Enum):
    """Overall contract risk level."""
    
    LOW = "low"              # Minor issues, acceptable
    MEDIUM = "medium"        # Notable issues, review recommended
    HIGH = "high"            # Significant risks, action required
    CRITICAL = "critical"    # Severe issues, do not sign
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.LOW: "Düşük",
            self.MEDIUM: "Orta",
            self.HIGH: "Yüksek",
            self.CRITICAL: "Kritik",
        }
        return names.get(self, self.value)
    
    @property
    def color(self) -> str:
        """UI color code."""
        colors = {
            self.LOW: "green",
            self.MEDIUM: "yellow",
            self.HIGH: "orange",
            self.CRITICAL: "red",
        }
        return colors.get(self, "gray")


class IssueCategory(str, enum.Enum):
    """Legal issue category."""
    
    MISSING_CLAUSE = "missing_clause"          # Required clause missing
    UNFAIR_TERM = "unfair_term"                # Potentially unfair term
    AMBIGUOUS = "ambiguous"                    # Ambiguous language
    COMPLIANCE = "compliance"                  # Non-compliance with law
    RISK = "risk"                              # General risk
    PENALTY = "penalty"                        # Excessive penalties
    TERMINATION = "termination"                # Problematic termination
    LIABILITY = "liability"                    # Liability concerns
    PAYMENT = "payment"                        # Payment terms issues
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.MISSING_CLAUSE: "Eksik Madde",
            self.UNFAIR_TERM: "Haksız Şart",
            self.AMBIGUOUS: "Belirsiz İfade",
            self.COMPLIANCE: "Uyumsuzluk",
            self.RISK: "Risk",
            self.PENALTY: "Yüksek Ceza",
            self.TERMINATION: "Fesih Sorunu",
            self.LIABILITY: "Sorumluluk",
            self.PAYMENT: "Ödeme Sorunu",
        }
        return names.get(self, self.value)


# =============================================================================
# CONTRACT ANALYSIS MODEL
# =============================================================================


class ContractAnalysis(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Contract Analysis model for AI-powered contract analysis.
    
    Stores comprehensive contract analysis results:
    - AI-generated insights
    - Risk assessment
    - Compliance checking
    - Clause extraction
    - Issue identification
    - Recommendations
    
    Analysis Process:
    1. User initiates analysis (document + parameters)
    2. AI processes contract (Claude API)
    3. Results stored with detailed findings
    4. User reviews and takes action
    
    Attributes:
        document_id: Contract being analyzed
        document: Document relationship
        
        analysis_type: Type of analysis
        status: Processing status
        
        initiated_by_id: User who requested analysis
        initiated_by: User relationship
        
        started_at: When analysis started
        completed_at: When analysis completed
        processing_time_seconds: Duration
        
        model_used: AI model identifier
        model_version: Model version
        
        parameters: Analysis parameters (JSON)
        
        risk_level: Overall risk assessment
        risk_score: Numeric risk score (0-100)
        confidence_score: Analysis confidence (0-1)
        
        findings: Detailed findings (JSON)
        issues: Identified issues (JSON array)
        recommendations: Recommendations (JSON array)
        
        clauses_extracted: Extracted clauses (JSON)
        missing_clauses: Missing required clauses (array)
        
        compliance_check: Compliance results (JSON)
        applicable_laws: Turkish laws referenced (array)
        
        summary: Executive summary text
        
        comparison_analysis: Comparison results (JSON, if applicable)
        
        error_message: Error details if failed
        
        metadata: Additional context
        
    Relationships:
        tenant: Parent tenant
        document: Contract being analyzed
        initiated_by: User who requested analysis
    """
    
    __tablename__ = "contract_analyses"
    
    # =========================================================================
    # DOCUMENT RELATIONSHIP
    # =========================================================================
    
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Contract/document being analyzed",
    )
    
    document = relationship(
        "Document",
        back_populates="contract_analyses",
    )
    
    # =========================================================================
    # ANALYSIS CLASSIFICATION
    # =========================================================================
    
    analysis_type = Column(
        Enum(AnalysisType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of analysis performed",
    )
    
    status = Column(
        Enum(AnalysisStatus, native_enum=False, length=50),
        nullable=False,
        default=AnalysisStatus.PENDING,
        index=True,
        comment="Analysis processing status",
    )
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    initiated_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who requested the analysis",
    )
    
    initiated_by = relationship(
        "User",
        back_populates="contract_analyses",
    )
    
    # =========================================================================
    # TIMING
    # =========================================================================
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When analysis processing started",
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When analysis completed",
    )
    
    processing_time_seconds = Column(
        Integer,
        nullable=True,
        comment="Analysis duration in seconds",
    )
    
    # =========================================================================
    # AI MODEL INFORMATION
    # =========================================================================
    
    model_used = Column(
        String(100),
        nullable=True,
        comment="AI model identifier (claude-sonnet-4.5, etc.)",
    )
    
    model_version = Column(
        String(50),
        nullable=True,
        comment="Model version/date",
    )
    
    # =========================================================================
    # ANALYSIS PARAMETERS
    # =========================================================================
    
    parameters = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Analysis parameters (focus areas, depth, language)",
    )
    
    # =========================================================================
    # RISK ASSESSMENT
    # =========================================================================
    
    risk_level = Column(
        Enum(RiskLevel, native_enum=False, length=50),
        nullable=True,
        index=True,
        comment="Overall risk level assessment",
    )
    
    risk_score = Column(
        Integer,
        nullable=True,
        comment="Numeric risk score (0-100, higher = more risky)",
    )
    
    confidence_score = Column(
        Float,
        nullable=True,
        comment="Analysis confidence score (0.0-1.0)",
    )
    
    # =========================================================================
    # FINDINGS & RESULTS
    # =========================================================================
    
    findings = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed analysis findings (structured data)",
    )
    
    issues = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Identified issues (array of issue objects)",
    )
    
    recommendations = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Recommendations for improvement (array)",
    )
    
    # =========================================================================
    # CLAUSE ANALYSIS
    # =========================================================================
    
    clauses_extracted = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Extracted and classified clauses (array)",
    )
    
    missing_clauses = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Missing required clauses (e.g., 'force majeure', 'arbitration')",
    )
    
    # =========================================================================
    # COMPLIANCE
    # =========================================================================
    
    compliance_check = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Compliance check results (per law/regulation)",
    )
    
    applicable_laws = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Turkish laws referenced (TBK, TTK, İş Kanunu, etc.)",
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    summary = Column(
        Text,
        nullable=True,
        comment="Executive summary of analysis (human-readable)",
    )
    
    # =========================================================================
    # COMPARISON (if applicable)
    # =========================================================================
    
    comparison_analysis = Column(
        JSONB,
        nullable=True,
        comment="Comparison results if comparing multiple contracts",
    )
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if analysis failed",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (token usage, prompts, etc.)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for document's analyses
        Index(
            "ix_contract_analyses_document",
            "document_id",
            "created_at",
        ),
        
        # Index for user's analyses
        Index(
            "ix_contract_analyses_user",
            "initiated_by_id",
            "created_at",
        ),
        
        # Index for completed analyses
        Index(
            "ix_contract_analyses_completed",
            "status",
            "completed_at",
            postgresql_where="status = 'completed'",
        ),
        
        # Index for risk level filtering
        Index(
            "ix_contract_analyses_risk",
            "tenant_id",
            "risk_level",
            postgresql_where="status = 'completed' AND risk_level IS NOT NULL",
        ),
        
        # Check: risk score range
        CheckConstraint(
            "risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)",
            name="ck_contract_analyses_risk_score",
        ),
        
        # Check: confidence score range
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)",
            name="ck_contract_analyses_confidence",
        ),
        
        # Check: processing time non-negative
        CheckConstraint(
            "processing_time_seconds IS NULL OR processing_time_seconds >= 0",
            name="ck_contract_analyses_time",
        ),
    )
    
    # =========================================================================
    # ANALYSIS CREATION
    # =========================================================================
    
    @classmethod
    def create_analysis(
        cls,
        document_id: UUIDType,
        tenant_id: UUIDType,
        initiated_by_id: UUIDType,
        analysis_type: AnalysisType,
        parameters: dict[str, Any] | None = None,
    ) -> "ContractAnalysis":
        """
        Create a new contract analysis request.
        
        Args:
            document_id: Document UUID
            tenant_id: Tenant UUID
            initiated_by_id: User UUID
            analysis_type: Type of analysis
            parameters: Analysis parameters
            
        Returns:
            ContractAnalysis: New analysis instance
            
        Example:
            >>> analysis = ContractAnalysis.create_analysis(
            ...     document_id=contract.id,
            ...     tenant_id=tenant.id,
            ...     initiated_by_id=user.id,
            ...     analysis_type=AnalysisType.FULL,
            ...     parameters={"focus": ["risk", "compliance"], "depth": "detailed"}
            ... )
        """
        analysis = cls(
            document_id=document_id,
            tenant_id=tenant_id,
            initiated_by_id=initiated_by_id,
            analysis_type=analysis_type,
            status=AnalysisStatus.PENDING,
            parameters=parameters or {},
        )
        
        logger.info(
            "Contract analysis created",
            analysis_id=str(analysis.id),
            document_id=str(document_id),
            analysis_type=analysis_type.value,
        )
        
        return analysis
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def start_processing(self, model_used: str, model_version: str | None = None) -> None:
        """
        Mark analysis as processing.
        
        Args:
            model_used: AI model identifier
            model_version: Model version
        """
        self.status = AnalysisStatus.PROCESSING
        self.started_at = datetime.now(timezone.utc)
        self.model_used = model_used
        self.model_version = model_version
        
        logger.info(
            "Contract analysis processing started",
            analysis_id=str(self.id),
            model=model_used,
        )
    
    def complete(
        self,
        findings: dict[str, Any],
        risk_level: RiskLevel | None = None,
        risk_score: int | None = None,
        confidence_score: float | None = None,
        issues: list[dict[str, Any]] | None = None,
        recommendations: list[dict[str, Any]] | None = None,
        clauses_extracted: list[dict[str, Any]] | None = None,
        missing_clauses: list[str] | None = None,
        compliance_check: dict[str, Any] | None = None,
        applicable_laws: list[str] | None = None,
        summary: str | None = None,
    ) -> None:
        """
        Complete analysis with results.
        
        Args:
            findings: Detailed findings
            risk_level: Overall risk level
            risk_score: Numeric risk score
            confidence_score: Confidence score
            issues: Identified issues
            recommendations: Recommendations
            clauses_extracted: Extracted clauses
            missing_clauses: Missing clauses
            compliance_check: Compliance results
            applicable_laws: Referenced laws
            summary: Executive summary
            
        Example:
            >>> analysis.complete(
            ...     findings={"contract_type": "employment", "parties": 2},
            ...     risk_level=RiskLevel.MEDIUM,
            ...     risk_score=65,
            ...     confidence_score=0.92,
            ...     issues=[
            ...         {
            ...             "category": "missing_clause",
            ...             "severity": "high",
            ...             "description": "Eksik gizlilik maddesi"
            ...         }
            ...     ],
            ...     recommendations=[
            ...         "Gizlilik maddesi eklenmeli"
            ...     ],
            ...     summary="Orta risk seviyeli iş sözleşmesi..."
            ... )
        """
        self.status = AnalysisStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        
        # Calculate processing time
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.processing_time_seconds = int(delta.total_seconds())
        
        # Store results
        self.findings = findings
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.confidence_score = confidence_score
        self.issues = issues or []
        self.recommendations = recommendations or []
        self.clauses_extracted = clauses_extracted or []
        self.missing_clauses = missing_clauses or []
        self.compliance_check = compliance_check or {}
        self.applicable_laws = applicable_laws or []
        self.summary = summary
        
        logger.info(
            "Contract analysis completed",
            analysis_id=str(self.id),
            risk_level=risk_level.value if risk_level else None,
            risk_score=risk_score,
            processing_time_seconds=self.processing_time_seconds,
        )
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark analysis as failed.
        
        Args:
            error_message: Error description
        """
        self.status = AnalysisStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        
        # Calculate processing time
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.processing_time_seconds = int(delta.total_seconds())
        
        logger.error(
            "Contract analysis failed",
            analysis_id=str(self.id),
            error=error_message,
        )
    
    def cancel(self) -> None:
        """Cancel analysis."""
        self.status = AnalysisStatus.CANCELLED
        
        logger.info(
            "Contract analysis cancelled",
            analysis_id=str(self.id),
        )
    
    # =========================================================================
    # ISSUE MANAGEMENT
    # =========================================================================
    
    def add_issue(
        self,
        category: IssueCategory,
        severity: RiskLevel,
        description: str,
        clause_reference: str | None = None,
        recommendation: str | None = None,
    ) -> None:
        """
        Add an identified issue.
        
        Args:
            category: Issue category
            severity: Issue severity
            description: Issue description
            clause_reference: Related clause
            recommendation: How to fix
            
        Example:
            >>> analysis.add_issue(
            ...     category=IssueCategory.MISSING_CLAUSE,
            ...     severity=RiskLevel.HIGH,
            ...     description="Gizlilik maddesi eksik",
            ...     recommendation="TBK'ya uygun gizlilik maddesi eklenmeli"
            ... )
        """
        issue = {
            "category": category.value,
            "severity": severity.value,
            "description": description,
            "clause_reference": clause_reference,
            "recommendation": recommendation,
            "identified_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if not isinstance(self.issues, list):
            self.issues = []
        
        self.issues.append(issue)
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "issues")
        
        logger.debug(
            "Issue added to analysis",
            analysis_id=str(self.id),
            category=category.value,
            severity=severity.value,
        )
    
    def get_high_priority_issues(self) -> list[dict[str, Any]]:
        """
        Get high and critical priority issues.
        
        Returns:
            list: High/critical issues
        """
        if not self.issues:
            return []
        
        return [
            issue for issue in self.issues
            if issue.get("severity") in ["high", "critical"]
        ]
    
    def get_issue_count_by_severity(self) -> dict[str, int]:
        """
        Get issue count by severity.
        
        Returns:
            dict: Count per severity level
        """
        if not self.issues:
            return {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for issue in self.issues:
            severity = issue.get("severity", "low")
            counts[severity] = counts.get(severity, 0) + 1
        
        return counts
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("risk_score")
    def validate_risk_score(self, key: str, risk_score: int | None) -> int | None:
        """Validate risk score range."""
        if risk_score is not None and not 0 <= risk_score <= 100:
            raise ValidationError(
                message="Risk score must be between 0 and 100",
                field="risk_score",
            )
        
        return risk_score
    
    @validates("confidence_score")
    def validate_confidence_score(
        self,
        key: str,
        confidence_score: float | None,
    ) -> float | None:
        """Validate confidence score range."""
        if confidence_score is not None and not 0.0 <= confidence_score <= 1.0:
            raise ValidationError(
                message="Confidence score must be between 0.0 and 1.0",
                field="confidence_score",
            )
        
        return confidence_score
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ContractAnalysis("
            f"id={self.id}, "
            f"type={self.analysis_type.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        
        # Add display names
        data["analysis_type_display"] = self.analysis_type.display_name_tr
        data["status_display"] = self.status.display_name_tr
        
        if self.risk_level:
            data["risk_level_display"] = self.risk_level.display_name_tr
            data["risk_level_color"] = self.risk_level.color
        
        # Add computed fields
        data["issue_count"] = len(self.issues) if self.issues else 0
        data["high_priority_issue_count"] = len(self.get_high_priority_issues())
        data["issue_count_by_severity"] = self.get_issue_count_by_severity()
        data["missing_clause_count"] = len(self.missing_clauses) if self.missing_clauses else 0
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ContractAnalysis",
    "AnalysisType",
    "AnalysisStatus",
    "RiskLevel",
    "IssueCategory",
]
