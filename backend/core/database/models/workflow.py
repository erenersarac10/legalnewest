"""
SQLAlchemy ORM Model: Workflow Definition
==========================================

Harvey/Legora CTO-Level Implementation
Turkish Legal AI Platform - Workflow Orchestration Layer

Bu modl workflow tan1mlar1n1n veritaban1 modelini salar.

SORUMLULUU:
-----------
- Workflow blueprint'lerinin saklanmas1 ve ynetimi
- DAG (Directed Acyclic Graph) yap1lar1n1n depolanmas1
- Trigger ve step tan1mlar1n1n JSONB format1nda tutulmas1
- Multi-tenant izolasyonu ve KVKK uyumluluu
- Versiyonlama ve lifecycle ynetimi
- Workflow metadata ve SLA takibi

KVKK UYUMLULUK:
--------------
 Sadece workflow tan1mlar1 (blueprint) saklan1r
 PII iermez, sadece yap1land1rma bilgileri
 Audit trail iin created_by/updated_by (user_id kullan1l1r)
L 0sim, email, telefon gibi ki_isel bilgiler saklanmaz

VERITABANASI ^EMASI:
-------------------
Table: workflows
- id (UUID, PK)
- tenant_id (String, indexed, RLS)
- name (String, unique per tenant)
- description (Text)
- version (Integer, default=1)
- status (Enum: draft, active, archived, deprecated)
- triggers (JSONB: [{type, config, enabled}])
- steps (JSONB: [{order, name, type, config, retry_policy}])
- tags (ARRAY: [str])
- practice_area (String: "i__hukuku", "ceza_hukuku", etc.)
- risk_profile (Enum: low, medium, high, critical)
- sla_minutes (Integer: max execution time)
- metadata (JSONB: feature flags, owner info, etc.)
- created_at, updated_at, deleted_at (audit)
- created_by, updated_by (audit)

0L0^K0LER:
---------
- workflow  workflow_executions (1:N)
- workflow  scheduled_jobs (1:N via workflow_id reference)

Author: Harvey/Legora CTO
Date: 2024-01-10
Version: 1.0.0
"""

from __future__ import annotations

import enum
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Column,
    Enum,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, validates

from backend.core.database.models.base import (
    AuditMixin,
    Base,
    BaseModelMixin,
    SoftDeleteMixin,
    TenantMixin,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class WorkflowStatus(str, enum.Enum):
    """
    Workflow lifecycle durumlar1

    - DRAFT: Henz tamamlanmam1_, test a_amas1nda
    - ACTIVE: retimde aktif, trigger edilebilir
    - ARCHIVED: Art1k kullan1lm1yor ama tarihsel kay1t iin saklan1yor
    - DEPRECATED: Yak1nda kald1r1lacak, yeni execution'lara kapal1
    """
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class WorkflowTriggerType(str, enum.Enum):
    """
    Workflow tetikleyici tipleri

    7 farkl1 trigger mekanizmas1 desteklenir:
    """
    MANUAL = "manual"  # API veya UI'den manuel tetikleme
    SCHEDULED = "scheduled"  # Cron-based zamanl1 tetikleme
    DOCUMENT_UPLOADED = "document_uploaded"  # Dokman yklendiinde
    CASE_CREATED = "case_created"  # Yeni dava a1ld11nda
    SLACK_COMMAND = "slack_command"  # Slack slash command
    TEAMS_MESSAGE = "teams_message"  # MS Teams mesaj1
    SHAREPOINT_WEBHOOK = "sharepoint_webhook"  # SharePoint webhook


class WorkflowStepType(str, enum.Enum):
    """
    Workflow ad1m tipleri

    8 farkl1 step tr desteklenir:
    """
    RAG_QUERY = "rag_query"  # RAG sorgusu (retrieval + generation)
    LEGAL_REASONING = "legal_reasoning"  # LLM-based hukuki analiz
    TIMELINE_ANALYSIS = "timeline_analysis"  # Zaman izelgesi 1karma
    COMPLIANCE_CHECK = "compliance_check"  # Uyumluluk kontrolleri
    REPORT_GENERATION = "report_generation"  # PDF/DOCX rapor retimi
    NOTIFICATION = "notification"  # Email/Slack/Teams bildirimi
    WEBHOOK_CALL = "webhook_call"  # Harici API ar1s1
    BULK_PROCESSING = "bulk_processing"  # Toplu i_lem (sub-workflow)


class RiskProfile(str, enum.Enum):
    """
    Workflow risk profili

    SLA ve nceliklendirme iin kullan1l1r
    """
    LOW = "low"  # D_k risk (rn: basit rapor)
    MEDIUM = "medium"  # Orta risk (rn: compliance check)
    HIGH = "high"  # Yksek risk (rn: mahkeme belgesi)
    CRITICAL = "critical"  # Kritik (rn: sre bitmek zere)


# ============================================================================
# WORKFLOW MODEL
# ============================================================================


class Workflow(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Workflow Definition ORM Model
    =============================

    Bir workflow'un tam tan1m1n1 saklar:
    - Tetikleyiciler (triggers)
    - Ad1mlar (steps) ve DAG yap1s1
    - Metadata ve konfigrasyon
    - Lifecycle durumu (draft, active, etc.)

    RNEK KULLANIM:
    -------------
    ```python
    workflow = Workflow(
        tenant_id="acme-law-firm",
        name="Yeni Dava Analiz Workflow",
        description="Yeni a1lan davalar1 otomatik analiz eder",
        status=WorkflowStatus.ACTIVE,
        triggers=[{
            "type": "case_created",
            "config": {"case_types": ["i__hukuku", "ticaret_hukuku"]},
            "enabled": True
        }],
        steps=[
            {
                "order": 1,
                "name": "extract_parties",
                "type": "timeline_analysis",
                "config": {"extract_entities": True},
                "retry_policy": {"max_attempts": 3, "backoff_seconds": 60}
            },
            {
                "order": 2,
                "name": "legal_analysis",
                "type": "legal_reasoning",
                "config": {"reasoning_style": "FULL_OPINION"},
                "depends_on": ["extract_parties"]
            }
        ],
        tags=["dava", "otomatik"],
        practice_area="i__hukuku",
        risk_profile=RiskProfile.HIGH,
        sla_minutes=30
    )
    session.add(workflow)
    await session.commit()
    ```

    KVKK NOTU:
    ---------
    Bu model sadece workflow blueprint'i saklar, execution-time'da
    i_lenen ki_isel verileri iermez.
    """

    __tablename__ = "workflows"

    # ========================================================================
    # COLUMNS
    # ========================================================================

    # Primary Key (UUID)
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Workflow unique identifier"
    )

    # Basic Info
    name = Column(
        String(255),
        nullable=False,
        comment="Workflow ad1 (rn: 'Yeni Dava Analiz Workflow')"
    )

    description = Column(
        Text,
        nullable=True,
        comment="Workflow a1klamas1 (markdown destekler)"
    )

    version = Column(
        Integer,
        nullable=False,
        default=1,
        comment="Workflow versiyonu (otomatik artar)"
    )

    status = Column(
        Enum(WorkflowStatus, name="workflow_status_enum"),
        nullable=False,
        default=WorkflowStatus.DRAFT,
        index=True,
        comment="Workflow lifecycle durumu"
    )

    # Workflow Definition (JSONB for flexibility)
    triggers = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Tetikleyici listesi: [{type, config, enabled}]"
    )

    steps = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Ad1m listesi: [{order, name, type, config, retry_policy, depends_on}]"
    )

    # Organization & Classification
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Arama ve filtreleme iin etiketler"
    )

    practice_area = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Hukuk dal1 (i__hukuku, ceza_hukuku, ticaret_hukuku, etc.)"
    )

    risk_profile = Column(
        Enum(RiskProfile, name="risk_profile_enum"),
        nullable=False,
        default=RiskProfile.MEDIUM,
        index=True,
        comment="Risk seviyesi (SLA nceliklendirme iin)"
    )

    # SLA & Performance
    sla_minutes = Column(
        Integer,
        nullable=True,
        comment="Maksimum execution sresi (dakika). None = s1n1rs1z"
    )

    # Metadata (feature flags, owner info, etc.)
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Ek bilgiler: feature_flags, owner_email (hash'li), cost_center, etc."
    )

    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================

    executions = relationship(
        "WorkflowExecution",
        back_populates="workflow",
        cascade="all, delete-orphan",
        lazy="dynamic",  # ok say1da execution olabilir, lazy load
        primaryjoin="and_(Workflow.id == foreign(WorkflowExecution.workflow_id), "
                    "WorkflowExecution.deleted_at.is_(None))"
    )

    # ========================================================================
    # CONSTRAINTS
    # ========================================================================

    __table_args__ = (
        # Unique constraint: tenant_id + name (ayn1 tenant'ta ayn1 isimde 2 workflow olamaz)
        UniqueConstraint(
            "tenant_id",
            "name",
            name="uq_workflow_tenant_name",
            postgresql_where=(Column("deleted_at").is_(None))  # Soft delete aware
        ),

        # Check constraint: sla_minutes must be positive
        CheckConstraint(
            "sla_minutes IS NULL OR sla_minutes > 0",
            name="ck_workflow_sla_positive"
        ),

        # Check constraint: version must be positive
        CheckConstraint(
            "version > 0",
            name="ck_workflow_version_positive"
        ),

        # Check constraint: triggers must be non-empty array
        CheckConstraint(
            "jsonb_array_length(triggers) > 0",
            name="ck_workflow_triggers_nonempty"
        ),

        # Check constraint: steps must be non-empty array
        CheckConstraint(
            "jsonb_array_length(steps) > 0",
            name="ck_workflow_steps_nonempty"
        ),

        # Composite index for common queries
        Index("ix_workflow_tenant_status", "tenant_id", "status"),
        Index("ix_workflow_tenant_practice_area", "tenant_id", "practice_area"),
        Index("ix_workflow_created_at", "created_at"),

        # GIN index for JSONB columns (fast search in triggers/steps/metadata)
        Index("ix_workflow_triggers_gin", "triggers", postgresql_using="gin"),
        Index("ix_workflow_steps_gin", "steps", postgresql_using="gin"),
        Index("ix_workflow_metadata_gin", "metadata", postgresql_using="gin"),

        # GIN index for tags array (fast tag search)
        Index("ix_workflow_tags_gin", "tags", postgresql_using="gin"),
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @validates("name")
    def validate_name(self, key: str, value: str) -> str:
        """Workflow isminin bo_ olmamas1n1 sala"""
        if not value or not value.strip():
            raise ValueError("Workflow name cannot be empty")

        # Trim whitespace
        value = value.strip()

        # Max length check (database constraint 255)
        if len(value) > 255:
            raise ValueError(f"Workflow name too long: {len(value)} > 255")

        return value

    @validates("version")
    def validate_version(self, key: str, value: int) -> int:
        """Version pozitif olmal1"""
        if value <= 0:
            raise ValueError(f"Workflow version must be positive, got {value}")
        return value

    @validates("triggers")
    def validate_triggers(self, key: str, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Triggers validation:
        - En az 1 trigger olmal1
        - Her trigger'da type, config, enabled olmal1
        - Type geerli bir WorkflowTriggerType olmal1
        """
        if not value:
            raise ValueError("Workflow must have at least 1 trigger")

        for idx, trigger in enumerate(value):
            # Required fields
            if "type" not in trigger:
                raise ValueError(f"Trigger {idx}: missing 'type' field")
            if "config" not in trigger:
                raise ValueError(f"Trigger {idx}: missing 'config' field")
            if "enabled" not in trigger:
                raise ValueError(f"Trigger {idx}: missing 'enabled' field")

            # Validate type
            try:
                WorkflowTriggerType(trigger["type"])
            except ValueError:
                raise ValueError(
                    f"Trigger {idx}: invalid type '{trigger['type']}'. "
                    f"Valid: {[t.value for t in WorkflowTriggerType]}"
                )

            # Config must be dict
            if not isinstance(trigger["config"], dict):
                raise ValueError(f"Trigger {idx}: config must be a dict")

            # Enabled must be bool
            if not isinstance(trigger["enabled"], bool):
                raise ValueError(f"Trigger {idx}: enabled must be a boolean")

        return value

    @validates("steps")
    def validate_steps(self, key: str, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Steps validation:
        - En az 1 step olmal1
        - Her step'te order, name, type, config olmal1
        - Order'lar unique ve sequential olmal1 (1, 2, 3, ...)
        - Name'ler unique olmal1
        - Type geerli bir WorkflowStepType olmal1
        - depends_on referanslar1 geerli olmal1 (ileri referans yok)
        """
        if not value:
            raise ValueError("Workflow must have at least 1 step")

        orders = []
        names = []

        for idx, step in enumerate(value):
            # Required fields
            if "order" not in step:
                raise ValueError(f"Step {idx}: missing 'order' field")
            if "name" not in step:
                raise ValueError(f"Step {idx}: missing 'name' field")
            if "type" not in step:
                raise ValueError(f"Step {idx}: missing 'type' field")
            if "config" not in step:
                raise ValueError(f"Step {idx}: missing 'config' field")

            # Validate order
            order = step["order"]
            if not isinstance(order, int) or order <= 0:
                raise ValueError(f"Step {idx}: order must be positive integer")

            if order in orders:
                raise ValueError(f"Step {idx}: duplicate order {order}")
            orders.append(order)

            # Validate name
            name = step["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Step {idx}: name must be non-empty string")

            if name in names:
                raise ValueError(f"Step {idx}: duplicate name '{name}'")
            names.append(name)

            # Validate type
            try:
                WorkflowStepType(step["type"])
            except ValueError:
                raise ValueError(
                    f"Step {idx}: invalid type '{step['type']}'. "
                    f"Valid: {[t.value for t in WorkflowStepType]}"
                )

            # Config must be dict
            if not isinstance(step["config"], dict):
                raise ValueError(f"Step {idx}: config must be a dict")

            # Validate depends_on (if exists)
            if "depends_on" in step:
                depends_on = step["depends_on"]
                if not isinstance(depends_on, list):
                    raise ValueError(f"Step {idx}: depends_on must be a list")

                for dep in depends_on:
                    if dep not in names:
                        raise ValueError(
                            f"Step {idx}: depends_on references unknown step '{dep}'"
                        )

        # Validate orders are sequential (1, 2, 3, ...)
        orders.sort()
        expected_orders = list(range(1, len(orders) + 1))
        if orders != expected_orders:
            raise ValueError(
                f"Step orders must be sequential starting from 1. Got {orders}"
            )

        return value

    @validates("tags")
    def validate_tags(self, key: str, value: List[str]) -> List[str]:
        """Tags validation: bo_ tag olamaz"""
        if not isinstance(value, list):
            raise ValueError("Tags must be a list")

        for idx, tag in enumerate(value):
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(f"Tag {idx}: must be non-empty string")

        # Deduplicate and lowercase
        return list(set(t.strip().lower() for t in value))

    @validates("sla_minutes")
    def validate_sla_minutes(self, key: str, value: Optional[int]) -> Optional[int]:
        """SLA pozitif olmal1 veya None"""
        if value is not None and value <= 0:
            raise ValueError(f"SLA must be positive or None, got {value}")
        return value

    @validates("metadata")
    def validate_metadata(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Metadata validation: KVKK uyumluluk kontrol

        L 0sim, email, telefon gibi PII iermemeli
         Feature flags, cost_center, owner_id (hash'li) gibi meta bilgiler OK
        """
        if not isinstance(value, dict):
            raise ValueError("Metadata must be a dict")

        # KVKK: PII field kontrol
        pii_fields = ["name", "email", "phone", "tc_kimlik_no", "address"]
        for field in pii_fields:
            if field in value:
                raise ValueError(
                    f"Metadata contains PII field '{field}'. "
                    f"Use IDs or hashed identifiers only (KVKK compliance)."
                )

        return value

    # ========================================================================
    # HYBRID PROPERTIES
    # ========================================================================

    @hybrid_property
    def is_active(self) -> bool:
        """Workflow aktif mi?"""
        return self.status == WorkflowStatus.ACTIVE

    @hybrid_property
    def is_draft(self) -> bool:
        """Workflow draft m1?"""
        return self.status == WorkflowStatus.DRAFT

    @hybrid_property
    def step_count(self) -> int:
        """Ka ad1m var?"""
        return len(self.steps)

    @hybrid_property
    def trigger_count(self) -> int:
        """Ka tetikleyici var?"""
        return len(self.triggers)

    # ========================================================================
    # METHODS
    # ========================================================================

    def to_dict(self, include_executions: bool = False) -> Dict[str, Any]:
        """
        Workflow'u dictionary'ye dn_tr

        Args:
            include_executions: Execution istatistiklerini dahil et

        Returns:
            Dict representation
        """
        data = {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "triggers": self.triggers,
            "steps": self.steps,
            "tags": self.tags,
            "practice_area": self.practice_area,
            "risk_profile": self.risk_profile.value,
            "sla_minutes": self.sla_minutes,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
        }

        if include_executions:
            # Lazy load executions ve istatistikleri hesapla
            total_executions = self.executions.count()
            successful_executions = self.executions.filter_by(status="success").count()
            failed_executions = self.executions.filter_by(status="failed").count()

            data["execution_stats"] = {
                "total": total_executions,
                "successful": successful_executions,
                "failed": failed_executions,
                "success_rate": (
                    successful_executions / total_executions
                    if total_executions > 0
                    else 0.0
                ),
            }

        return data

    def get_step_by_name(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Step'i ismine gre bul

        Args:
            step_name: Step ad1

        Returns:
            Step dict veya None
        """
        for step in self.steps:
            if step.get("name") == step_name:
                return step
        return None

    def get_step_by_order(self, order: int) -> Optional[Dict[str, Any]]:
        """
        Step'i s1ras1na gre bul

        Args:
            order: Step order (1, 2, 3, ...)

        Returns:
            Step dict veya None
        """
        for step in self.steps:
            if step.get("order") == order:
                return step
        return None

    def get_enabled_triggers(self) -> List[Dict[str, Any]]:
        """Sadece enabled=True olan trigger'lar1 dndr"""
        return [t for t in self.triggers if t.get("enabled", False)]

    def has_trigger_type(self, trigger_type: WorkflowTriggerType) -> bool:
        """Belirli bir trigger type'1 var m1?"""
        return any(
            t.get("type") == trigger_type.value
            for t in self.triggers
        )

    def activate(self) -> None:
        """
        Workflow'u aktif et

        Raises:
            ValueError: Draft deilse veya validation hatalar1 varsa
        """
        if self.status != WorkflowStatus.DRAFT:
            raise ValueError(
                f"Cannot activate workflow in status '{self.status.value}'. "
                f"Only draft workflows can be activated."
            )

        # Re-validate triggers and steps
        self.validate_triggers("triggers", self.triggers)
        self.validate_steps("steps", self.steps)

        self.status = WorkflowStatus.ACTIVE
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Workflow activated: {self.name} (id={self.id}, tenant={self.tenant_id})"
        )

    def deactivate(self) -> None:
        """Workflow'u deaktif et (archived)"""
        if self.status != WorkflowStatus.ACTIVE:
            raise ValueError(
                f"Cannot archive workflow in status '{self.status.value}'. "
                f"Only active workflows can be archived."
            )

        self.status = WorkflowStatus.ARCHIVED
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Workflow archived: {self.name} (id={self.id}, tenant={self.tenant_id})"
        )

    def deprecate(self) -> None:
        """Workflow'u deprecated olarak i_aretle"""
        if self.status not in [WorkflowStatus.ACTIVE, WorkflowStatus.DRAFT]:
            raise ValueError(
                f"Cannot deprecate workflow in status '{self.status.value}'"
            )

        self.status = WorkflowStatus.DEPRECATED
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Workflow deprecated: {self.name} (id={self.id}, tenant={self.tenant_id})"
        )

    def increment_version(self) -> None:
        """Version numaras1n1 art1r"""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Workflow version incremented: {self.name} v{self.version} "
            f"(id={self.id}, tenant={self.tenant_id})"
        )

    def __repr__(self) -> str:
        return (
            f"<Workflow(id={self.id}, tenant={self.tenant_id}, name='{self.name}', "
            f"version={self.version}, status={self.status.value}, "
            f"steps={self.step_count}, triggers={self.trigger_count})>"
        )


# ============================================================================
# EVENT LISTENERS
# ============================================================================


@event.listens_for(Workflow, "before_insert")
def workflow_before_insert(mapper, connection, target: Workflow) -> None:
    """
    Insert ncesi validation ve default deer ayarlar1
    """
    # Ensure UUID
    if target.id is None:
        target.id = uuid4()

    # Ensure version starts at 1
    if target.version is None or target.version < 1:
        target.version = 1

    # Ensure timestamps
    if target.created_at is None:
        target.created_at = datetime.now(timezone.utc)

    if target.updated_at is None:
        target.updated_at = datetime.now(timezone.utc)

    logger.debug(
        f"Workflow before_insert: {target.name} (tenant={target.tenant_id})"
    )


@event.listens_for(Workflow, "before_update")
def workflow_before_update(mapper, connection, target: Workflow) -> None:
    """
    Update ncesi timestamp gncelleme
    """
    target.updated_at = datetime.now(timezone.utc)

    logger.debug(
        f"Workflow before_update: {target.name} (id={target.id}, tenant={target.tenant_id})"
    )


@event.listens_for(Workflow, "after_insert")
def workflow_after_insert(mapper, connection, target: Workflow) -> None:
    """
    Insert sonras1 audit log
    """
    logger.info(
        f" Workflow created: {target.name} (id={target.id}, tenant={target.tenant_id}, "
        f"status={target.status.value}, steps={target.step_count})"
    )


@event.listens_for(Workflow, "after_delete")
def workflow_after_delete(mapper, connection, target: Workflow) -> None:
    """
    Delete sonras1 audit log (soft delete)
    """
    logger.warning(
        f"= Workflow deleted: {target.name} (id={target.id}, tenant={target.tenant_id})"
    )
