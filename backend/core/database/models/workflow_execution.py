"""
SQLAlchemy ORM Model: Workflow Execution
=========================================

Harvey/Legora CTO-Level Implementation
Turkish Legal AI Platform - Workflow Orchestration Layer

Bu modl workflow execution'lar1n1n (al1_ma kay1tlar1n1n) veritaban1 modelini salar.

SORUMLULUU:
-----------
- Workflow al1_t1rmalar1n1n takibi (execution tracking)
- Step-level sonular1n saklanmas1
- Hata ve performance metriklerinin kayd1
- SLA ihlallerinin tespiti
- Resume/retry mekanizmas1 iin checkpoint'ler
- Multi-tenant izolasyonu ve KVKK uyumluluu

KVKK UYUMLULUK:
--------------
 Execution metadata (step sonular1, hatalar) saklan1r
 PII ASLA saklanmaz - sadece referans ID'ler kullan1l1r
 rnek: "Davac1: Ahmet Y1lmaz" L  "davac1_id: uuid-123" 
 Audit trail iin user_id kullan1l1r (isim deil)
L Dokman ierikleri, isimler, TC kimlik no saklanmaz

VER0TABANASI ^EMASI:
-------------------
Table: workflow_executions
- id (UUID, PK)
- tenant_id (String, indexed, RLS)
- workflow_id (UUID, FK  workflows.id)
- triggered_by (String: user_id veya "system")
- trigger_type (Enum: manual, scheduled, event)
- payload (JSONB: input parametreleri - KVKK safe)
- status (Enum: pending, running, success, failed, cancelled, paused)
- started_at (DateTime)
- completed_at (DateTime, nullable)
- duration_ms (Integer: completed_at - started_at)
- step_results (JSONB: [{step_name, status, output, error, duration_ms}])
- error_message (Text, nullable)
- error_type (String, nullable)
- retry_count (Integer, default=0)
- sla_violated (Boolean)
- metadata (JSONB: feature flags, cost tracking, etc.)
- created_at, updated_at, deleted_at (audit)

0L0^K0LER:
---------
- workflow_execution  workflow (N:1)

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
    Boolean,
    CheckConstraint,
    Column,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
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


class ExecutionStatus(str, enum.Enum):
    """
    Workflow execution durumlar1

    - PENDING: Henz ba_lamad1, kuyrua al1nd1
    - RUNNING: ^u anda al1_1yor
    - SUCCESS: Ba_ar1yla tamamland1
    - FAILED: Hata ile sonland1
    - CANCELLED: Manuel olarak iptal edildi
    - PAUSED: Durduruldu, resume edilebilir
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionTriggerType(str, enum.Enum):
    """
    Execution'1 tetikleyen mekanizma

    - MANUAL: Kullan1c1 taraf1ndan manuel tetiklendi (UI veya API)
    - SCHEDULED: Zamanlanm1_ i_ (cron, Celery Beat)
    - EVENT: Sistem eventi (dokman yklendi, dava a1ld1, etc.)
    """
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"


class StepStatus(str, enum.Enum):
    """
    Workflow iindeki bir ad1m1n durumu

    - PENDING: Henz ba_lamad1
    - RUNNING: al1_1yor
    - SUCCESS: Ba_ar1l1
    - FAILED: Hata ald1
    - SKIPPED: Conditional logic nedeniyle atland1
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# WORKFLOW EXECUTION MODEL
# ============================================================================


class WorkflowExecution(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Workflow Execution ORM Model
    ============================

    Bir workflow'un tek bir al1_t1rmas1n1 (execution) takip eder.

    Her execution:
    - Hangi workflow'dan tredi? (workflow_id)
    - Kim tetikledi? (triggered_by)
    - Input neydi? (payload)
    - Durum nedir? (status)
    - Ne zaman ba_lad1/bitti? (started_at, completed_at)
    - Her step'in sonucu ne? (step_results)
    - Hata var m1? (error_message, error_type)

    RNEK KULLANIM:
    -------------
    ```python
    execution = WorkflowExecution(
        tenant_id="acme-law-firm",
        workflow_id=workflow.id,
        triggered_by="user-123",
        trigger_type=ExecutionTriggerType.MANUAL,
        payload={
            "case_id": "case-456",
            "document_ids": ["doc-1", "doc-2"]
        },
        status=ExecutionStatus.PENDING
    )
    session.add(execution)
    await session.commit()

    # Start execution
    execution.start()

    # Update step result
    execution.update_step_result(
        step_name="extract_parties",
        status=StepStatus.SUCCESS,
        output={"parties": ["davac1", "daval1"]},
        duration_ms=1250
    )

    # Complete execution
    execution.complete(success=True)
    ```

    KVKK NOTU:
    ---------
    Execution sonular1 ASLA PII iermemeli. rnek:

    L YANLI^:
    payload = {"user_name": "Ahmet Y1lmaz", "tc": "12345678901"}

     DORU:
    payload = {"user_id": "uuid-123", "case_id": "uuid-456"}
    """

    __tablename__ = "workflow_executions"

    # ========================================================================
    # COLUMNS
    # ========================================================================

    # Primary Key (UUID)
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Execution unique identifier"
    )

    # Foreign Key to Workflow
    workflow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Hangi workflow al1_t1r1ld1?"
    )

    # Trigger Info
    triggered_by = Column(
        String(255),
        nullable=False,
        comment="Kim tetikledi? (user_id veya 'system')"
    )

    trigger_type = Column(
        Enum(ExecutionTriggerType, name="execution_trigger_type_enum"),
        nullable=False,
        index=True,
        comment="Nas1l tetiklendi? (manual, scheduled, event)"
    )

    # Input Payload (KVKK-safe)
    payload = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Execution input parametreleri (KVKK-safe: sadece ID'ler)"
    )

    # Execution Status
    status = Column(
        Enum(ExecutionStatus, name="execution_status_enum"),
        nullable=False,
        default=ExecutionStatus.PENDING,
        index=True,
        comment="Execution durumu (pending, running, success, failed, etc.)"
    )

    # Timing
    started_at = Column(
        TIMESTAMP(timezone=True),
        nullable=True,
        comment="Ne zaman ba_lad1?"
    )

    completed_at = Column(
        TIMESTAMP(timezone=True),
        nullable=True,
        comment="Ne zaman tamamland1?"
    )

    duration_ms = Column(
        Integer,
        nullable=True,
        comment="Toplam sre (milisaniye): completed_at - started_at"
    )

    # Step-Level Results
    step_results = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Her step'in sonucu: [{step_name, status, output, error, duration_ms}]"
    )

    # Error Info
    error_message = Column(
        Text,
        nullable=True,
        comment="Hata mesaj1 (varsa)"
    )

    error_type = Column(
        String(255),
        nullable=True,
        comment="Hata tipi (rn: 'TimeoutError', 'ValidationError')"
    )

    retry_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Ka kere retry edildi?"
    )

    # SLA Tracking
    sla_violated = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="SLA ihlal edildi mi?"
    )

    # Metadata (cost tracking, feature flags, etc.)
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Ek bilgiler: cost_usd, llm_tokens, feature_flags, etc."
    )

    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================

    workflow = relationship(
        "Workflow",
        back_populates="executions",
        lazy="joined"  # Workflow bilgisi genelde laz1m, eager load
    )

    # ========================================================================
    # CONSTRAINTS
    # ========================================================================

    __table_args__ = (
        # Check constraint: duration_ms must be non-negative
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_execution_duration_nonnegative"
        ),

        # Check constraint: retry_count must be non-negative
        CheckConstraint(
            "retry_count >= 0",
            name="ck_execution_retry_nonnegative"
        ),

        # Check constraint: started_at <= completed_at
        CheckConstraint(
            "started_at IS NULL OR completed_at IS NULL OR started_at <= completed_at",
            name="ck_execution_times_logical"
        ),

        # Composite indexes for common queries
        Index("ix_execution_tenant_status", "tenant_id", "status"),
        Index("ix_execution_tenant_workflow", "tenant_id", "workflow_id"),
        Index("ix_execution_workflow_status", "workflow_id", "status"),
        Index("ix_execution_started_at", "started_at"),
        Index("ix_execution_sla_violated", "sla_violated"),

        # GIN indexes for JSONB
        Index("ix_execution_payload_gin", "payload", postgresql_using="gin"),
        Index("ix_execution_step_results_gin", "step_results", postgresql_using="gin"),
        Index("ix_execution_metadata_gin", "metadata", postgresql_using="gin"),
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @validates("triggered_by")
    def validate_triggered_by(self, key: str, value: str) -> str:
        """triggered_by bo_ olamaz"""
        if not value or not value.strip():
            raise ValueError("triggered_by cannot be empty")
        return value.strip()

    @validates("payload")
    def validate_payload(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Payload validation: KVKK uyumluluk kontrol

        L 0sim, email, telefon, TC gibi PII iermemeli
         Sadece ID'ler ve non-PII veriler
        """
        if not isinstance(value, dict):
            raise ValueError("Payload must be a dict")

        # KVKK: PII field kontrol
        pii_fields = [
            "name", "email", "phone", "tc_kimlik_no", "address",
            "user_name", "full_name", "phone_number", "tc_no"
        ]

        for field in pii_fields:
            if field in value:
                raise ValueError(
                    f"Payload contains PII field '{field}'. "
                    f"Use IDs only (e.g., 'user_id', 'case_id'). KVKK compliance required."
                )

        return value

    @validates("duration_ms")
    def validate_duration_ms(self, key: str, value: Optional[int]) -> Optional[int]:
        """Duration negatif olamaz"""
        if value is not None and value < 0:
            raise ValueError(f"duration_ms cannot be negative, got {value}")
        return value

    @validates("retry_count")
    def validate_retry_count(self, key: str, value: int) -> int:
        """Retry count negatif olamaz"""
        if value < 0:
            raise ValueError(f"retry_count cannot be negative, got {value}")
        return value

    @validates("step_results")
    def validate_step_results(self, key: str, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step results validation:
        - Her step result'1n step_name, status olmal1
        - Status geerli bir StepStatus olmal1
        """
        if not isinstance(value, list):
            raise ValueError("step_results must be a list")

        for idx, result in enumerate(value):
            # Required fields
            if "step_name" not in result:
                raise ValueError(f"Step result {idx}: missing 'step_name'")
            if "status" not in result:
                raise ValueError(f"Step result {idx}: missing 'status'")

            # Validate status
            try:
                StepStatus(result["status"])
            except ValueError:
                raise ValueError(
                    f"Step result {idx}: invalid status '{result['status']}'. "
                    f"Valid: {[s.value for s in StepStatus]}"
                )

            # Duration must be non-negative (if exists)
            if "duration_ms" in result:
                duration = result["duration_ms"]
                if not isinstance(duration, (int, float)) or duration < 0:
                    raise ValueError(
                        f"Step result {idx}: duration_ms must be non-negative, got {duration}"
                    )

        return value

    # ========================================================================
    # HYBRID PROPERTIES
    # ========================================================================

    @hybrid_property
    def is_running(self) -> bool:
        """Execution al1_1yor mu?"""
        return self.status == ExecutionStatus.RUNNING

    @hybrid_property
    def is_completed(self) -> bool:
        """Execution tamamland1 m1? (success, failed, cancelled)"""
        return self.status in [
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED
        ]

    @hybrid_property
    def is_successful(self) -> bool:
        """Execution ba_ar1l1 m1?"""
        return self.status == ExecutionStatus.SUCCESS

    @hybrid_property
    def is_failed(self) -> bool:
        """Execution ba_ar1s1z m1?"""
        return self.status == ExecutionStatus.FAILED

    @hybrid_property
    def step_count(self) -> int:
        """Ka step sonucu var?"""
        return len(self.step_results)

    @hybrid_property
    def successful_steps(self) -> int:
        """Ka step ba_ar1l1?"""
        return sum(
            1 for result in self.step_results
            if result.get("status") == StepStatus.SUCCESS.value
        )

    @hybrid_property
    def failed_steps(self) -> int:
        """Ka step ba_ar1s1z?"""
        return sum(
            1 for result in self.step_results
            if result.get("status") == StepStatus.FAILED.value
        )

    # ========================================================================
    # METHODS
    # ========================================================================

    def start(self) -> None:
        """
        Execution'1 ba_lat

        Raises:
            ValueError: Eer status PENDING deilse
        """
        if self.status != ExecutionStatus.PENDING:
            raise ValueError(
                f"Cannot start execution in status '{self.status.value}'. "
                f"Only PENDING executions can be started."
            )

        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Execution started: {self.id} (workflow={self.workflow_id}, "
            f"tenant={self.tenant_id}, triggered_by={self.triggered_by})"
        )

    def complete(self, success: bool, error_message: Optional[str] = None,
                 error_type: Optional[str] = None) -> None:
        """
        Execution'1 tamamla

        Args:
            success: Ba_ar1l1 m1?
            error_message: Hata mesaj1 (ba_ar1s1zsa)
            error_type: Hata tipi (ba_ar1s1zsa)

        Raises:
            ValueError: Eer status RUNNING deilse
        """
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(
                f"Cannot complete execution in status '{self.status.value}'. "
                f"Only RUNNING executions can be completed."
            )

        self.status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

        # Calculate duration
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)

        # Set error info (if failed)
        if not success:
            self.error_message = error_message
            self.error_type = error_type

        # Check SLA violation
        if self.workflow and self.workflow.sla_minutes:
            sla_ms = self.workflow.sla_minutes * 60 * 1000
            if self.duration_ms and self.duration_ms > sla_ms:
                self.sla_violated = True
                logger.warning(
                    f" SLA violated: execution {self.id} took {self.duration_ms}ms "
                    f"(SLA: {sla_ms}ms, workflow={self.workflow.name})"
                )

        logger.info(
            f"Execution completed: {self.id} (status={self.status.value}, "
            f"duration={self.duration_ms}ms, tenant={self.tenant_id})"
        )

    def cancel(self) -> None:
        """
        Execution'1 iptal et

        Raises:
            ValueError: Eer status RUNNING veya PENDING deilse
        """
        if self.status not in [ExecutionStatus.RUNNING, ExecutionStatus.PENDING]:
            raise ValueError(
                f"Cannot cancel execution in status '{self.status.value}'. "
                f"Only RUNNING/PENDING executions can be cancelled."
            )

        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

        # Calculate duration (if started)
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)

        logger.info(
            f"Execution cancelled: {self.id} (tenant={self.tenant_id})"
        )

    def pause(self) -> None:
        """
        Execution'1 duraklat

        Raises:
            ValueError: Eer status RUNNING deilse
        """
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(
                f"Cannot pause execution in status '{self.status.value}'. "
                f"Only RUNNING executions can be paused."
            )

        self.status = ExecutionStatus.PAUSED
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Execution paused: {self.id} (tenant={self.tenant_id})"
        )

    def resume(self) -> None:
        """
        Execution'1 devam ettir

        Raises:
            ValueError: Eer status PAUSED deilse
        """
        if self.status != ExecutionStatus.PAUSED:
            raise ValueError(
                f"Cannot resume execution in status '{self.status.value}'. "
                f"Only PAUSED executions can be resumed."
            )

        self.status = ExecutionStatus.RUNNING
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Execution resumed: {self.id} (tenant={self.tenant_id})"
        )

    def increment_retry(self) -> None:
        """Retry say1s1n1 art1r"""
        self.retry_count += 1
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Execution retry incremented: {self.id} (retry_count={self.retry_count})"
        )

    def update_step_result(self, step_name: str, status: StepStatus,
                           output: Optional[Dict[str, Any]] = None,
                           error: Optional[str] = None,
                           duration_ms: Optional[int] = None) -> None:
        """
        Bir step'in sonucunu gncelle

        Args:
            step_name: Step ad1
            status: Step durumu
            output: Step 1kt1s1 (KVKK-safe)
            error: Hata mesaj1 (varsa)
            duration_ms: Step sresi (ms)
        """
        # Find existing result
        existing_idx = None
        for idx, result in enumerate(self.step_results):
            if result.get("step_name") == step_name:
                existing_idx = idx
                break

        # Create result dict
        result = {
            "step_name": step_name,
            "status": status.value,
            "output": output or {},
            "error": error,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Update or append
        if existing_idx is not None:
            self.step_results[existing_idx] = result
        else:
            self.step_results.append(result)

        # Mark as modified for SQLAlchemy
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "step_results")

        self.updated_at = datetime.now(timezone.utc)

        logger.debug(
            f"Step result updated: execution={self.id}, step={step_name}, "
            f"status={status.value}, duration={duration_ms}ms"
        )

    def get_step_result(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Bir step'in sonucunu getir

        Args:
            step_name: Step ad1

        Returns:
            Step result dict veya None
        """
        for result in self.step_results:
            if result.get("step_name") == step_name:
                return result
        return None

    def to_dict(self, include_workflow: bool = False) -> Dict[str, Any]:
        """
        Execution'1 dictionary'ye dn_tr

        Args:
            include_workflow: Workflow bilgisini dahil et

        Returns:
            Dict representation
        """
        data = {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "workflow_id": str(self.workflow_id),
            "triggered_by": self.triggered_by,
            "trigger_type": self.trigger_type.value,
            "payload": self.payload,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "step_results": self.step_results,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "sla_violated": self.sla_violated,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        if include_workflow and self.workflow:
            data["workflow"] = {
                "id": str(self.workflow.id),
                "name": self.workflow.name,
                "version": self.workflow.version,
                "status": self.workflow.status.value
            }

        return data

    def __repr__(self) -> str:
        return (
            f"<WorkflowExecution(id={self.id}, workflow={self.workflow_id}, "
            f"status={self.status.value}, tenant={self.tenant_id}, "
            f"duration={self.duration_ms}ms)>"
        )


# ============================================================================
# EVENT LISTENERS
# ============================================================================


@event.listens_for(WorkflowExecution, "before_insert")
def execution_before_insert(mapper, connection, target: WorkflowExecution) -> None:
    """
    Insert ncesi validation ve default deer ayarlar1
    """
    # Ensure UUID
    if target.id is None:
        target.id = uuid4()

    # Ensure timestamps
    if target.created_at is None:
        target.created_at = datetime.now(timezone.utc)

    if target.updated_at is None:
        target.updated_at = datetime.now(timezone.utc)

    logger.debug(
        f"Execution before_insert: {target.id} (workflow={target.workflow_id}, "
        f"tenant={target.tenant_id})"
    )


@event.listens_for(WorkflowExecution, "before_update")
def execution_before_update(mapper, connection, target: WorkflowExecution) -> None:
    """
    Update ncesi timestamp gncelleme
    """
    target.updated_at = datetime.now(timezone.utc)

    logger.debug(
        f"Execution before_update: {target.id} (status={target.status.value})"
    )


@event.listens_for(WorkflowExecution, "after_insert")
def execution_after_insert(mapper, connection, target: WorkflowExecution) -> None:
    """
    Insert sonras1 audit log
    """
    logger.info(
        f" Execution created: {target.id} (workflow={target.workflow_id}, "
        f"tenant={target.tenant_id}, triggered_by={target.triggered_by}, "
        f"trigger_type={target.trigger_type.value})"
    )


@event.listens_for(WorkflowExecution, "after_delete")
def execution_after_delete(mapper, connection, target: WorkflowExecution) -> None:
    """
    Delete sonras1 audit log (soft delete)
    """
    logger.warning(
        f"= Execution deleted: {target.id} (workflow={target.workflow_id}, "
        f"tenant={target.tenant_id})"
    )
