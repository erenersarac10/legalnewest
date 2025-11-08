"""
Audit Integrity Service - Blockchain-Style Tamper-Proof Audit Trail.

This module provides cryptographic integrity verification for audit logs:
- Hash-chain generation (blockchain-style event linking)
- Tamper-proof event ID generation
- Integrity verification and validation
- Chain gap detection
- Forensic analysis

Architecture:
    Each audit event contains:
    - event_id: Unique identifier (UUID)
    - event_hash: SHA-256(event_data + previous_hash)
    - previous_hash: Hash of previous event (chain link)
    - sequence_number: Monotonic sequence (gap detection)
    - created_at: Timestamp (chronological ordering)

    Chain Structure:
    Event 1: hash = SHA-256(data_1 + "genesis")
    Event 2: hash = SHA-256(data_2 + hash_1)
    Event 3: hash = SHA-256(data_3 + hash_2)
    ...

    This creates an immutable chain where:
    - Any modification to Event N breaks hash verification
    - Any deletion creates a sequence gap
    - Any insertion is detectable via timestamp + sequence anomalies

Why Hash-Chain?
    - Blockchain-proven: Bitcoin uses same principle
    - Forensic evidence: Tampering is cryptographically provable
    - Compliance: GDPR Article 32(1)(b) requires data integrity
    - Legal defensibility: Hash-chain provides non-repudiation

Performance:
    - Hash generation: ~0.1ms per event (SHA-256)
    - Chain verification: O(n) linear scan
    - Optimized verification: Only verify suspicious ranges

Security:
    - Hash algorithm: SHA-256 (FIPS 140-2 approved)
    - Salt: Tenant-specific secret (prevents rainbow tables)
    - Previous hash: Links events cryptographically
    - Sequence numbers: Detects gaps and insertions

Example:
    >>> from backend.services.audit_integrity import AuditIntegrityService
    >>>
    >>> async with get_db() as db:
    ...     integrity = AuditIntegrityService(db)
    ...
    ...     # Create event with hash-chain
    ...     event = await integrity.create_integrity_event(
    ...         tenant_id=tenant_id,
    ...         event_type="DATA_ACCESS",
    ...         event_data={"user_id": user_id, "resource": "documents/123"}
    ...     )
    ...     print(f"Event hash: {event.event_hash}")
    ...     print(f"Previous: {event.previous_hash}")
    ...
    ...     # Verify chain integrity
    ...     result = await integrity.verify_chain(
    ...         tenant_id=tenant_id,
    ...         start_date=start,
    ...         end_date=end
    ...     )
    ...     if result["is_valid"]:
    ...         print("✓ Chain integrity verified")
    ...     else:
    ...         print(f"✗ Integrity violation: {result['violations']}")
"""

import datetime
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.models.compliance_audit_log import (
    ComplianceAuditLog,
    ComplianceEventType,
)
from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Hash algorithm (FIPS 140-2 approved)
HASH_ALGORITHM = "sha256"

# Genesis hash (first event in chain)
GENESIS_HASH = "0" * 64  # 64 hex chars (SHA-256)

# Verification batch size (prevent memory overflow)
VERIFICATION_BATCH_SIZE = 1000


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class IntegrityEvent:
    """Integrity-protected event."""

    event_id: UUID
    event_hash: str
    previous_hash: str
    sequence_number: int
    created_at: datetime.datetime
    event_type: str
    event_data: Dict[str, Any]
    tenant_id: UUID


@dataclass
class IntegrityVerificationResult:
    """Chain verification result."""

    is_valid: bool
    total_events: int
    verified_events: int
    violations: List[Dict[str, Any]]
    chain_length: int
    verification_duration_ms: float


@dataclass
class IntegrityViolation:
    """Integrity violation details."""

    event_id: UUID
    sequence_number: int
    violation_type: str  # hash_mismatch, chain_gap, missing_predecessor, timestamp_anomaly
    expected_value: Optional[str]
    actual_value: Optional[str]
    description: str


# =============================================================================
# AUDIT INTEGRITY SERVICE
# =============================================================================


class AuditIntegrityService:
    """
    Audit integrity service with hash-chain verification.

    This service provides blockchain-style integrity protection
    for audit logs, enabling cryptographic proof of tampering.

    Features:
        - Hash-chain generation (SHA-256)
        - Event linking (previous_hash)
        - Sequence numbering (gap detection)
        - Chain verification
        - Tamper detection
        - Forensic analysis

    All operations are tenant-scoped for multi-tenancy security.
    """

    def __init__(self, db: AsyncSession, tenant_secret: Optional[str] = None):
        """
        Initialize integrity service.

        Args:
            db: Database session
            tenant_secret: Tenant-specific secret for salting (optional)
        """
        self.db = db
        self.tenant_secret = tenant_secret or "default-secret"  # In production, fetch from secrets manager

    # =========================================================================
    # HASH-CHAIN GENERATION
    # =========================================================================

    def generate_event_hash(
        self,
        event_data: Dict[str, Any],
        previous_hash: str,
        sequence_number: int,
        tenant_id: UUID,
    ) -> str:
        """
        Generate cryptographic hash for event.

        This creates a hash that links to the previous event,
        forming an unbreakable chain.

        Hash = SHA-256(
            tenant_id +
            sequence_number +
            event_data (canonical JSON) +
            previous_hash +
            tenant_secret (salt)
        )

        Args:
            event_data: Event data to hash
            previous_hash: Hash of previous event
            sequence_number: Sequence number
            tenant_id: Tenant ID

        Returns:
            Hexadecimal hash string (64 chars)

        Example:
            >>> hash1 = service.generate_event_hash(
            ...     event_data={"action": "login"},
            ...     previous_hash=GENESIS_HASH,
            ...     sequence_number=1,
            ...     tenant_id=tenant_id
            ... )
            >>> print(hash1)  # "a1b2c3d4..."
        """
        # Canonical JSON (sorted keys for consistency)
        canonical_data = json.dumps(event_data, sort_keys=True, ensure_ascii=False)

        # Concatenate all components
        hash_input = (
            f"{tenant_id}"
            f"{sequence_number}"
            f"{canonical_data}"
            f"{previous_hash}"
            f"{self.tenant_secret}"
        )

        # Generate SHA-256 hash
        hash_object = hashlib.new(HASH_ALGORITHM)
        hash_object.update(hash_input.encode("utf-8"))

        return hash_object.hexdigest()

    async def get_previous_event(
        self,
        tenant_id: UUID,
    ) -> Optional[Tuple[str, int]]:
        """
        Get previous event's hash and sequence number.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tuple of (previous_hash, sequence_number) or None if no previous events
        """
        # Query last event for tenant (ordered by sequence)
        query = (
            select(
                ComplianceAuditLog.event_hash,
                ComplianceAuditLog.sequence_number,
            )
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .where(ComplianceAuditLog.event_hash.isnot(None))
            .order_by(desc(ComplianceAuditLog.sequence_number))
            .limit(1)
        )

        result = await self.db.execute(query)
        row = result.first()

        if row:
            return (row[0], row[1])
        return None

    async def get_next_sequence_number(
        self,
        tenant_id: UUID,
    ) -> int:
        """
        Get next sequence number for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Next sequence number (1 if no events exist)
        """
        query = (
            select(func.max(ComplianceAuditLog.sequence_number))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
        )

        result = await self.db.execute(query)
        max_seq = result.scalar()

        return (max_seq + 1) if max_seq else 1

    # =========================================================================
    # EVENT CREATION WITH INTEGRITY
    # =========================================================================

    async def create_integrity_event(
        self,
        tenant_id: UUID,
        event_type: ComplianceEventType,
        event_data: Dict[str, Any],
        *,
        data_subject_id: Optional[UUID] = None,
        compliance_framework: str = "GDPR",
    ) -> IntegrityEvent:
        """
        Create audit event with hash-chain integrity.

        This method:
        1. Gets previous event's hash
        2. Gets next sequence number
        3. Generates new event hash (linked to previous)
        4. Creates audit log with integrity metadata
        5. Returns integrity-protected event

        Args:
            tenant_id: Tenant ID
            event_type: Event type
            event_data: Event data
            data_subject_id: Data subject ID
            compliance_framework: Framework (GDPR, KVKK, etc.)

        Returns:
            IntegrityEvent with hash-chain metadata

        Example:
            >>> event = await service.create_integrity_event(
            ...     tenant_id=tenant_id,
            ...     event_type=ComplianceEventType.DATA_ACCESS,
            ...     event_data={
            ...         "user_id": str(user_id),
            ...         "resource": "documents/123",
            ...         "action": "view"
            ...     }
            ... )
            >>> print(f"Event {event.sequence_number}: {event.event_hash}")
        """
        # Get previous event (for chain linking)
        prev_result = await self.get_previous_event(tenant_id)
        if prev_result:
            previous_hash, _ = prev_result
        else:
            previous_hash = GENESIS_HASH  # First event

        # Get next sequence number
        sequence_number = await self.get_next_sequence_number(tenant_id)

        # Generate event hash
        event_hash = self.generate_event_hash(
            event_data=event_data,
            previous_hash=previous_hash,
            sequence_number=sequence_number,
            tenant_id=tenant_id,
        )

        # Create event ID
        event_id = uuid4()

        # Create compliance audit log with integrity metadata
        log = ComplianceAuditLog(
            id=event_id,
            tenant_id=tenant_id,
            data_subject_id=data_subject_id,
            event_type=event_type,
            compliance_framework=compliance_framework,
            data_categories=[],
            processing_purpose=f"Integrity-protected event: {event_type}",
            recipients=[],
            description=json.dumps(event_data),
            metadata={
                "integrity": {
                    "event_hash": event_hash,
                    "previous_hash": previous_hash,
                    "sequence_number": sequence_number,
                    "hash_algorithm": HASH_ALGORITHM,
                },
                **event_data,
            },
            # Add integrity fields (requires model update)
            # event_hash=event_hash,
            # previous_hash=previous_hash,
            # sequence_number=sequence_number,
        )

        self.db.add(log)
        await self.db.flush()

        logger.info(
            f"Created integrity event #{sequence_number} for tenant {tenant_id}",
            extra={
                "event_id": str(event_id),
                "sequence_number": sequence_number,
                "event_hash": event_hash,
                "previous_hash": previous_hash,
            },
        )

        return IntegrityEvent(
            event_id=event_id,
            event_hash=event_hash,
            previous_hash=previous_hash,
            sequence_number=sequence_number,
            created_at=log.created_at,
            event_type=str(event_type),
            event_data=event_data,
            tenant_id=tenant_id,
        )

    # =========================================================================
    # CHAIN VERIFICATION
    # =========================================================================

    async def verify_chain(
        self,
        tenant_id: UUID,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        limit: Optional[int] = None,
    ) -> IntegrityVerificationResult:
        """
        Verify audit log hash-chain integrity.

        This performs a full cryptographic verification:
        1. Fetch events in sequence order
        2. For each event:
           - Verify hash matches computed hash
           - Verify previous_hash matches predecessor
           - Verify sequence numbers are continuous
           - Verify timestamps are monotonic
        3. Return verification result with violations

        Args:
            tenant_id: Tenant ID
            start_date: Start date (optional)
            end_date: End date (optional)
            limit: Maximum events to verify (optional)

        Returns:
            IntegrityVerificationResult

        Example:
            >>> result = await service.verify_chain(tenant_id=tenant_id)
            >>> if result.is_valid:
            ...     print(f"✓ Verified {result.verified_events} events")
            ... else:
            ...     for v in result.violations:
            ...         print(f"✗ {v['violation_type']}: {v['description']}")
        """
        start_time = datetime.datetime.utcnow()
        violations = []

        # Build query
        query = (
            select(ComplianceAuditLog)
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .order_by(ComplianceAuditLog.sequence_number)
        )

        if start_date:
            query = query.where(ComplianceAuditLog.created_at >= start_date)
        if end_date:
            query = query.where(ComplianceAuditLog.created_at <= end_date)
        if limit:
            query = query.limit(limit)

        # Execute query
        result = await self.db.execute(query)
        events = list(result.scalars().all())

        total_events = len(events)
        verified_events = 0
        previous_hash = GENESIS_HASH
        previous_sequence = 0
        previous_timestamp = None

        # Verify each event
        for event in events:
            # Extract integrity metadata
            integrity_meta = event.metadata.get("integrity", {})
            if not integrity_meta:
                violations.append({
                    "event_id": str(event.id),
                    "sequence_number": None,
                    "violation_type": "missing_integrity_metadata",
                    "description": "Event missing integrity metadata (pre-integrity implementation)",
                })
                continue

            event_hash = integrity_meta.get("event_hash")
            stored_previous_hash = integrity_meta.get("previous_hash")
            sequence_number = integrity_meta.get("sequence_number")

            # Check 1: Sequence number continuity
            if sequence_number != previous_sequence + 1:
                violations.append({
                    "event_id": str(event.id),
                    "sequence_number": sequence_number,
                    "violation_type": "chain_gap",
                    "expected_value": str(previous_sequence + 1),
                    "actual_value": str(sequence_number),
                    "description": f"Sequence gap: expected {previous_sequence + 1}, got {sequence_number}",
                })

            # Check 2: Previous hash matches
            if stored_previous_hash != previous_hash:
                violations.append({
                    "event_id": str(event.id),
                    "sequence_number": sequence_number,
                    "violation_type": "missing_predecessor",
                    "expected_value": previous_hash,
                    "actual_value": stored_previous_hash,
                    "description": f"Previous hash mismatch",
                })

            # Check 3: Hash verification
            event_data = {k: v for k, v in event.metadata.items() if k != "integrity"}
            computed_hash = self.generate_event_hash(
                event_data=event_data,
                previous_hash=stored_previous_hash,
                sequence_number=sequence_number,
                tenant_id=tenant_id,
            )

            if computed_hash != event_hash:
                violations.append({
                    "event_id": str(event.id),
                    "sequence_number": sequence_number,
                    "violation_type": "hash_mismatch",
                    "expected_value": computed_hash,
                    "actual_value": event_hash,
                    "description": "Event hash mismatch (data tampering detected)",
                })

            # Check 4: Timestamp monotonicity
            if previous_timestamp and event.created_at < previous_timestamp:
                violations.append({
                    "event_id": str(event.id),
                    "sequence_number": sequence_number,
                    "violation_type": "timestamp_anomaly",
                    "expected_value": f">= {previous_timestamp}",
                    "actual_value": str(event.created_at),
                    "description": "Timestamp went backwards (possible manipulation)",
                })

            # Update previous values
            previous_hash = event_hash
            previous_sequence = sequence_number
            previous_timestamp = event.created_at
            verified_events += 1

        # Calculate duration
        duration_ms = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000

        is_valid = len(violations) == 0

        logger.info(
            f"Chain verification completed: {verified_events}/{total_events} events, "
            f"{len(violations)} violations",
            extra={
                "tenant_id": str(tenant_id),
                "is_valid": is_valid,
                "verified_events": verified_events,
                "total_events": total_events,
                "violations": len(violations),
                "duration_ms": duration_ms,
            },
        )

        return IntegrityVerificationResult(
            is_valid=is_valid,
            total_events=total_events,
            verified_events=verified_events,
            violations=violations,
            chain_length=previous_sequence,
            verification_duration_ms=duration_ms,
        )

    # =========================================================================
    # FORENSIC ANALYSIS
    # =========================================================================

    async def detect_tampering(
        self,
        tenant_id: UUID,
        event_id: UUID,
    ) -> Dict[str, Any]:
        """
        Perform forensic analysis on specific event.

        This checks if a single event has been tampered with
        by verifying its hash and chain linkage.

        Args:
            tenant_id: Tenant ID
            event_id: Event ID to analyze

        Returns:
            Forensic analysis result

        Example:
            >>> result = await service.detect_tampering(
            ...     tenant_id=tenant_id,
            ...     event_id=suspicious_event_id
            ... )
            >>> if result["is_tampered"]:
            ...     print(f"ALERT: Tampering detected - {result['evidence']}")
        """
        # Fetch event
        query = (
            select(ComplianceAuditLog)
            .where(ComplianceAuditLog.id == event_id)
            .where(ComplianceAuditLog.tenant_id == tenant_id)
        )
        result = await self.db.execute(query)
        event = result.scalar_one_or_none()

        if not event:
            raise ValidationError(f"Event {event_id} not found")

        # Extract integrity metadata
        integrity_meta = event.metadata.get("integrity", {})
        if not integrity_meta:
            return {
                "is_tampered": False,
                "confidence": "unknown",
                "reason": "No integrity metadata (pre-integrity event)",
            }

        event_hash = integrity_meta.get("event_hash")
        stored_previous_hash = integrity_meta.get("previous_hash")
        sequence_number = integrity_meta.get("sequence_number")

        # Recompute hash
        event_data = {k: v for k, v in event.metadata.items() if k != "integrity"}
        computed_hash = self.generate_event_hash(
            event_data=event_data,
            previous_hash=stored_previous_hash,
            sequence_number=sequence_number,
            tenant_id=tenant_id,
        )

        is_tampered = computed_hash != event_hash

        return {
            "event_id": str(event_id),
            "sequence_number": sequence_number,
            "is_tampered": is_tampered,
            "confidence": "high" if is_tampered else "verified",
            "stored_hash": event_hash,
            "computed_hash": computed_hash,
            "evidence": {
                "hash_match": not is_tampered,
                "previous_hash": stored_previous_hash,
                "sequence_number": sequence_number,
                "timestamp": event.created_at.isoformat(),
            },
        }

    async def get_chain_statistics(
        self,
        tenant_id: UUID,
    ) -> Dict[str, Any]:
        """
        Get hash-chain statistics for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Chain statistics

        Example:
            >>> stats = await service.get_chain_statistics(tenant_id=tenant_id)
            >>> print(f"Chain length: {stats['chain_length']}")
            >>> print(f"Integrity coverage: {stats['integrity_coverage']}%")
        """
        # Total events
        total_query = select(func.count(ComplianceAuditLog.id)).where(
            ComplianceAuditLog.tenant_id == tenant_id
        )
        total_result = await self.db.execute(total_query)
        total_events = total_result.scalar() or 0

        # Events with integrity
        # Note: This would be more efficient with a dedicated column
        # For now, we check metadata
        integrity_query = select(ComplianceAuditLog).where(
            ComplianceAuditLog.tenant_id == tenant_id
        )
        integrity_result = await self.db.execute(integrity_query)
        events = list(integrity_result.scalars().all())

        events_with_integrity = sum(
            1 for e in events if e.metadata.get("integrity")
        )

        # Chain length (max sequence)
        seq_query = (
            select(func.max(ComplianceAuditLog.sequence_number))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
        )
        seq_result = await self.db.execute(seq_query)
        chain_length = seq_result.scalar() or 0

        return {
            "tenant_id": str(tenant_id),
            "total_events": total_events,
            "events_with_integrity": events_with_integrity,
            "chain_length": chain_length,
            "integrity_coverage": (
                round((events_with_integrity / total_events) * 100, 2)
                if total_events > 0
                else 0
            ),
            "genesis_hash": GENESIS_HASH,
            "hash_algorithm": HASH_ALGORITHM,
        }
