"""
SQLAlchemy ORM Model: Scheduled Job
====================================

Harvey/Legora CTO-Level Implementation
Turkish Legal AI Platform - Workflow Orchestration Layer

Bu modl scheduled (zamanlanm) job execution'larnn veritaban modelini salar.

SORUMLULUU:
-----------
- Zamanlanm i altrmalarnn takibi (scheduled job tracking)
- SLA ihlallerinin kayd ve eskalasyonu
- Job metrics (cost, duration, success rate)
- Distributed lock durumu ve idempotency takibi
- Feature flag kontrolleri ve rate limit durumu
- Multi-tenant izolasyonu ve KVKK uyumluluu

KVKK UYUMLULUK:
--------------
 Job metadata ve metrikler saklanr
 PII ASLA saklanmaz - sadece job execution bilgileri
 Tenant ID ile izolasyon salanr
 Kullanc isimleri, email, telefon saklanmaz

VERTABANASI EMASI:
-------------------
Table: scheduled_jobs
- id (UUID, PK)
- tenant_id (String, indexed, RLS)
- job_type (Enum: 10 farkl scheduled job tipi)
- status (Enum: pending, running, success, failed, skipped)
- triggered_at (DateTime: ne zaman tetiklendi?)
- completed_at (DateTime, nullable)
- duration_ms (Integer: completed_at - triggered_at)
- sla_minutes (Integer: bu job iin tanml SLA)
- sla_violated (Boolean: SLA ihlal edildi mi?)
- next_run_at (DateTime: bir sonraki alma zaman)
- lock_acquired (Boolean: distributed lock alnd m?)
- lock_key (String: Redis lock key)
- feature_flag_enabled (Boolean: feature flag aktif miydi?)
- rate_limited (Boolean: rate limit nedeniyle atland m?)
- metrics (JSONB: documents_processed, cost_usd, llm_tokens, etc.)
- error_message (Text, nullable)
- error_type (String, nullable)
- retry_count (Integer, default=0)
- metadata (JSONB: ek bilgiler)
- created_at, updated_at, deleted_at (audit)

LKLER:
---------
Bu model bamszdr (no foreign keys), scheduled jobs tm tenant'lar iin genel takip salar.

Author: Harvey/Legora CTO
Date: 2024-01-10
Version: 1.0.0
"""

from __future__ import annotations

import enum
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Enum,
    Index,
    Integer,
    String,
    Text,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import validates

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


class ScheduledJobType(str, enum.Enum):
    """
    Scheduled Job Tipleri

    backend/core/queue/tasks/scheduled_workflow_tasks.py ile senkronize edilmeli.

    10 farkl scheduled job desteklenir:
    """
    # Maintenance Jobs
    DAILY_INDEX_HEALTH_CHECK = "daily_index_health_check"  # SLA: 5 dakika
    NIGHTLY_BULK_INGESTION = "nightly_bulk_ingestion"  # SLA: 2 saat
    WEEKLY_COMPLIANCE_REPORT = "weekly_compliance_report"  # SLA: 30 dakika
    MONTHLY_COST_OPTIMIZATION = "monthly_cost_optimization"  # SLA: 1 saat
    ORPHAN_WORKFLOW_CLEANUP = "orphan_workflow_cleanup"  # SLA: 15 dakika

    # Monitoring Jobs
    HOURLY_SLA_CHECK = "hourly_sla_check"  # SLA: 10 dakika
    DAILY_METRICS_AGGREGATION = "daily_metrics_aggregation"  # SLA: 20 dakika

    # Data Jobs
    NIGHTLY_EMBEDDING_REFRESH = "nightly_embedding_refresh"  # SLA: 3 saat
    WEEKLY_INDEX_OPTIMIZATION = "weekly_index_optimization"  # SLA: 1 saat

    # Audit Jobs
    DAILY_AUDIT_LOG_ROTATION = "daily_audit_log_rotation"  # SLA: 30 dakika


class ScheduledJobStatus(str, enum.Enum):
    """
    Scheduled Job Durumlar

    - PENDING: Kuyrua alnd, henz balamad
    - RUNNING: alyor
    - SUCCESS: Baaryla tamamland
    - FAILED: Hata ile sonland
    - SKIPPED: Feature flag veya rate limit nedeniyle atland
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# SCHEDULED JOB MODEL
# ============================================================================


class ScheduledJob(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Scheduled Job ORM Model
    =======================

    Zamanlanm bir iin (scheduled job) tek bir altrmasn takip eder.

    Her scheduled job execution:
    - Hangi job tipi? (job_type)
    - Ne zaman tetiklendi? (triggered_at)
    - Durum nedir? (status)
    - SLA ihlal edildi mi? (sla_violated)
    - Lock alnd m? (lock_acquired, lock_key)
    - Feature flag aktif miydi? (feature_flag_enabled)
    - Rate limit nedeniyle atland m? (rate_limited)
    - Metrics (JSONB: processed count, cost, tokens, etc.)

    RNEK KULLANIM:
    -------------
    ```python
    job = ScheduledJob(
        tenant_id="acme-law-firm",
        job_type=ScheduledJobType.NIGHTLY_BULK_INGESTION,
        status=ScheduledJobStatus.PENDING,
        triggered_at=datetime.now(timezone.utc),
        sla_minutes=120,  # 2 hours
        lock_key="lock:scheduled:nightly_bulk_ingestion:acme-law-firm",
        feature_flag_enabled=True
    )
    session.add(job)
    await session.commit()

    # Start job
    job.start(lock_acquired=True)

    # Complete job
    job.complete(
        success=True,
        metrics={
            "documents_processed": 10000,
            "duration_ms": 7200000,
            "cost_usd": 12.50,
            "llm_tokens": 150000
        }
    )
    ```

    KVKK NOTU:
    ---------
    Scheduled job metadata ASLA PII iermez.
    Sadece job execution bilgileri ve metrikler saklanr.
    """

    __tablename__ = "scheduled_jobs"

    # ========================================================================
    # COLUMNS
    # ========================================================================

    # Primary Key (UUID)
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Scheduled job execution unique identifier"
    )

    # Job Type
    job_type = Column(
        Enum(ScheduledJobType, name="scheduled_job_type_enum"),
        nullable=False,
        index=True,
        comment="Job tipi (rn: nightly_bulk_ingestion, daily_index_health_check)"
    )

    # Status
    status = Column(
        Enum(ScheduledJobStatus, name="scheduled_job_status_enum"),
        nullable=False,
        default=ScheduledJobStatus.PENDING,
        index=True,
        comment="Job durumu (pending, running, success, failed, skipped)"
    )

    # Timing
    triggered_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        index=True,
        comment="Job ne zaman tetiklendi?"
    )

    completed_at = Column(
        TIMESTAMP(timezone=True),
        nullable=True,
        comment="Job ne zaman tamamland?"
    )

    duration_ms = Column(
        Integer,
        nullable=True,
        comment="Job sresi (milisaniye): completed_at - triggered_at"
    )

    # SLA Tracking
    sla_minutes = Column(
        Integer,
        nullable=False,
        comment="Bu job iin tanml SLA (dakika)"
    )

    sla_violated = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="SLA ihlal edildi mi?"
    )

    # Next Run
    next_run_at = Column(
        TIMESTAMP(timezone=True),
        nullable=True,
        index=True,
        comment="Bir sonraki alma zaman (recurring jobs iin)"
    )

    # Distributed Lock (Idempotency)
    lock_acquired = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Distributed lock (Redis) alnd m?"
    )

    lock_key = Column(
        String(500),
        nullable=True,
        index=True,
        comment="Redis lock key (rn: lock:scheduled:job_type:tenant_id)"
    )

    # Feature Flags & Rate Limiting
    feature_flag_enabled = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Job altnda feature flag aktif miydi?"
    )

    rate_limited = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Rate limit nedeniyle atland m?"
    )

    # Metrics (JSONB)
    metrics = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Job metrikleri: {documents_processed, cost_usd, llm_tokens, etc.}"
    )

    # Error Info
    error_message = Column(
        Text,
        nullable=True,
        comment="Hata mesaj (varsa)"
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

    # Metadata
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Ek bilgiler: triggered_by, celery_task_id, etc."
    )

    # ========================================================================
    # CONSTRAINTS
    # ========================================================================

    __table_args__ = (
        # Check constraint: duration_ms must be non-negative
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_scheduled_job_duration_nonnegative"
        ),

        # Check constraint: retry_count must be non-negative
        CheckConstraint(
            "retry_count >= 0",
            name="ck_scheduled_job_retry_nonnegative"
        ),

        # Check constraint: sla_minutes must be positive
        CheckConstraint(
            "sla_minutes > 0",
            name="ck_scheduled_job_sla_positive"
        ),

        # Check constraint: triggered_at <= completed_at
        CheckConstraint(
            "completed_at IS NULL OR triggered_at <= completed_at",
            name="ck_scheduled_job_times_logical"
        ),

        # Composite indexes for common queries
        Index("ix_scheduled_job_tenant_type", "tenant_id", "job_type"),
        Index("ix_scheduled_job_tenant_status", "tenant_id", "status"),
        Index("ix_scheduled_job_type_status", "job_type", "status"),
        Index("ix_scheduled_job_triggered_at", "triggered_at"),
        Index("ix_scheduled_job_sla_violated", "sla_violated"),

        # GIN indexes for JSONB
        Index("ix_scheduled_job_metrics_gin", "metrics", postgresql_using="gin"),
        Index("ix_scheduled_job_metadata_gin", "metadata", postgresql_using="gin"),
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @validates("sla_minutes")
    def validate_sla_minutes(self, key: str, value: int) -> int:
        """SLA pozitif olmal"""
        if value <= 0:
            raise ValueError(f"sla_minutes must be positive, got {value}")
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

    @validates("lock_key")
    def validate_lock_key(self, key: str, value: Optional[str]) -> Optional[str]:
        """Lock key max 500 karakter"""
        if value and len(value) > 500:
            raise ValueError(f"lock_key too long: {len(value)} > 500")
        return value

    @validates("metrics")
    def validate_metrics(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        """Metrics validation"""
        if not isinstance(value, dict):
            raise ValueError("metrics must be a dict")
        return value

    # ========================================================================
    # HYBRID PROPERTIES
    # ========================================================================

    @hybrid_property
    def is_running(self) -> bool:
        """Job alyor mu?"""
        return self.status == ScheduledJobStatus.RUNNING

    @hybrid_property
    def is_completed(self) -> bool:
        """Job tamamland m?"""
        return self.status in [
            ScheduledJobStatus.SUCCESS,
            ScheduledJobStatus.FAILED,
            ScheduledJobStatus.SKIPPED
        ]

    @hybrid_property
    def is_successful(self) -> bool:
        """Job baarl m?"""
        return self.status == ScheduledJobStatus.SUCCESS

    @hybrid_property
    def is_failed(self) -> bool:
        """Job baarsz m?"""
        return self.status == ScheduledJobStatus.FAILED

    @hybrid_property
    def was_skipped(self) -> bool:
        """Job atland m?"""
        return self.status == ScheduledJobStatus.SKIPPED

    # ========================================================================
    # METHODS
    # ========================================================================

    def start(self, lock_acquired: bool = True, lock_key: Optional[str] = None) -> None:
        """
        Job' balat

        Args:
            lock_acquired: Distributed lock alnd m?
            lock_key: Redis lock key

        Raises:
            ValueError: Eer status PENDING deilse
        """
        if self.status != ScheduledJobStatus.PENDING:
            raise ValueError(
                f"Cannot start job in status '{self.status.value}'. "
                f"Only PENDING jobs can be started."
            )

        self.status = ScheduledJobStatus.RUNNING
        self.lock_acquired = lock_acquired

        if lock_key:
            self.lock_key = lock_key

        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Scheduled job started: {self.id} (type={self.job_type.value}, "
            f"tenant={self.tenant_id}, lock_acquired={lock_acquired})"
        )

    def complete(self, success: bool, metrics: Optional[Dict[str, Any]] = None,
                 error_message: Optional[str] = None, error_type: Optional[str] = None,
                 next_run_at: Optional[datetime] = None) -> None:
        """
        Job' tamamla

        Args:
            success: Baarl m?
            metrics: Job metrikleri (documents_processed, cost_usd, etc.)
            error_message: Hata mesaj (baarszsa)
            error_type: Hata tipi (baarszsa)
            next_run_at: Bir sonraki alma zaman (recurring jobs iin)

        Raises:
            ValueError: Eer status RUNNING deilse
        """
        if self.status != ScheduledJobStatus.RUNNING:
            raise ValueError(
                f"Cannot complete job in status '{self.status.value}'. "
                f"Only RUNNING jobs can be completed."
            )

        self.status = ScheduledJobStatus.SUCCESS if success else ScheduledJobStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)

        # Calculate duration
        delta = self.completed_at - self.triggered_at
        self.duration_ms = int(delta.total_seconds() * 1000)

        # Set metrics
        if metrics:
            self.metrics = metrics

        # Set error info (if failed)
        if not success:
            self.error_message = error_message
            self.error_type = error_type

        # Check SLA violation
        sla_ms = self.sla_minutes * 60 * 1000
        if self.duration_ms > sla_ms:
            self.sla_violated = True
            logger.warning(
                f" SLA violated: scheduled job {self.id} ({self.job_type.value}) "
                f"took {self.duration_ms}ms (SLA: {sla_ms}ms, tenant={self.tenant_id})"
            )

        # Set next run time
        if next_run_at:
            self.next_run_at = next_run_at

        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Scheduled job completed: {self.id} (type={self.job_type.value}, "
            f"status={self.status.value}, duration={self.duration_ms}ms, "
            f"sla_violated={self.sla_violated}, tenant={self.tenant_id})"
        )

    def skip(self, reason: str, rate_limited: bool = False,
             feature_flag_enabled: bool = False, next_run_at: Optional[datetime] = None) -> None:
        """
        Job' atla (skip)

        Args:
            reason: Atlanma nedeni
            rate_limited: Rate limit nedeniyle mi atland?
            feature_flag_enabled: Feature flag kapal myd?
            next_run_at: Bir sonraki alma zaman

        Raises:
            ValueError: Eer status PENDING veya RUNNING deilse
        """
        if self.status not in [ScheduledJobStatus.PENDING, ScheduledJobStatus.RUNNING]:
            raise ValueError(
                f"Cannot skip job in status '{self.status.value}'. "
                f"Only PENDING/RUNNING jobs can be skipped."
            )

        self.status = ScheduledJobStatus.SKIPPED
        self.completed_at = datetime.now(timezone.utc)

        # Calculate duration (if started)
        if self.status == ScheduledJobStatus.RUNNING:
            delta = self.completed_at - self.triggered_at
            self.duration_ms = int(delta.total_seconds() * 1000)

        # Set skip reason
        self.metadata["skip_reason"] = reason
        self.rate_limited = rate_limited
        self.feature_flag_enabled = feature_flag_enabled

        # Set next run time
        if next_run_at:
            self.next_run_at = next_run_at

        # Mark as modified for SQLAlchemy
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "metadata")

        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Scheduled job skipped: {self.id} (type={self.job_type.value}, "
            f"reason={reason}, rate_limited={rate_limited}, tenant={self.tenant_id})"
        )

    def increment_retry(self) -> None:
        """Retry saysn artr"""
        self.retry_count += 1
        self.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Scheduled job retry incremented: {self.id} (type={self.job_type.value}, "
            f"retry_count={self.retry_count}, tenant={self.tenant_id})"
        )

    def calculate_next_run(self, interval_minutes: int) -> datetime:
        """
        Bir sonraki alma zamann hesapla

        Args:
            interval_minutes: Interval (dakika)

        Returns:
            Bir sonraki alma zaman
        """
        if self.completed_at:
            next_run = self.completed_at + timedelta(minutes=interval_minutes)
        else:
            next_run = self.triggered_at + timedelta(minutes=interval_minutes)

        self.next_run_at = next_run
        self.updated_at = datetime.now(timezone.utc)

        logger.debug(
            f"Next run calculated: {self.id} (type={self.job_type.value}, "
            f"next_run_at={next_run.isoformat()})"
        )

        return next_run

    def to_dict(self) -> Dict[str, Any]:
        """
        Job' dictionary'ye dntr

        Returns:
            Dict representation
        """
        return {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "sla_minutes": self.sla_minutes,
            "sla_violated": self.sla_violated,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "lock_acquired": self.lock_acquired,
            "lock_key": self.lock_key,
            "feature_flag_enabled": self.feature_flag_enabled,
            "rate_limited": self.rate_limited,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def get_sla_for_job_type(cls, job_type: ScheduledJobType) -> int:
        """
        Job type iin SLA'y dndr (dakika)

        Args:
            job_type: Job tipi

        Returns:
            SLA (dakika)
        """
        sla_map = {
            ScheduledJobType.DAILY_INDEX_HEALTH_CHECK: 5,
            ScheduledJobType.NIGHTLY_BULK_INGESTION: 120,
            ScheduledJobType.WEEKLY_COMPLIANCE_REPORT: 30,
            ScheduledJobType.MONTHLY_COST_OPTIMIZATION: 60,
            ScheduledJobType.ORPHAN_WORKFLOW_CLEANUP: 15,
            ScheduledJobType.HOURLY_SLA_CHECK: 10,
            ScheduledJobType.DAILY_METRICS_AGGREGATION: 20,
            ScheduledJobType.NIGHTLY_EMBEDDING_REFRESH: 180,
            ScheduledJobType.WEEKLY_INDEX_OPTIMIZATION: 60,
            ScheduledJobType.DAILY_AUDIT_LOG_ROTATION: 30,
        }

        return sla_map.get(job_type, 60)  # Default: 60 minutes

    def __repr__(self) -> str:
        return (
            f"<ScheduledJob(id={self.id}, type={self.job_type.value}, "
            f"status={self.status.value}, tenant={self.tenant_id}, "
            f"duration={self.duration_ms}ms, sla_violated={self.sla_violated})>"
        )


# ============================================================================
# EVENT LISTENERS
# ============================================================================


@event.listens_for(ScheduledJob, "before_insert")
def scheduled_job_before_insert(mapper, connection, target: ScheduledJob) -> None:
    """
    Insert ncesi validation ve default deer ayarlar
    """
    # Ensure UUID
    if target.id is None:
        target.id = uuid4()

    # Ensure triggered_at
    if target.triggered_at is None:
        target.triggered_at = datetime.now(timezone.utc)

    # Auto-set SLA if not provided
    if target.sla_minutes is None or target.sla_minutes <= 0:
        target.sla_minutes = ScheduledJob.get_sla_for_job_type(target.job_type)

    # Ensure timestamps
    if target.created_at is None:
        target.created_at = datetime.now(timezone.utc)

    if target.updated_at is None:
        target.updated_at = datetime.now(timezone.utc)

    logger.debug(
        f"Scheduled job before_insert: {target.job_type.value} (tenant={target.tenant_id})"
    )


@event.listens_for(ScheduledJob, "before_update")
def scheduled_job_before_update(mapper, connection, target: ScheduledJob) -> None:
    """
    Update ncesi timestamp gncelleme
    """
    target.updated_at = datetime.now(timezone.utc)

    logger.debug(
        f"Scheduled job before_update: {target.id} (type={target.job_type.value}, "
        f"status={target.status.value})"
    )


@event.listens_for(ScheduledJob, "after_insert")
def scheduled_job_after_insert(mapper, connection, target: ScheduledJob) -> None:
    """
    Insert sonras audit log
    """
    logger.info(
        f" Scheduled job created: {target.id} (type={target.job_type.value}, "
        f"tenant={target.tenant_id}, sla={target.sla_minutes}m)"
    )


@event.listens_for(ScheduledJob, "after_delete")
def scheduled_job_after_delete(mapper, connection, target: ScheduledJob) -> None:
    """
    Delete sonras audit log (soft delete)
    """
    logger.warning(
        f" Scheduled job deleted: {target.id} (type={target.job_type.value}, "
        f"tenant={target.tenant_id})"
    )
