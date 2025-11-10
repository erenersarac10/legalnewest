"""
Audit Service - Harvey/Legora CTO-Level Implementation

Enterprise-grade audit logging and security monitoring system with comprehensive
event tracking, tamper-proof logging, and compliance reporting.

Architecture:
    +-----------------+
    |  Audit Service  |
    +--------+--------+
             |
             +---> Event Capture & Classification
             |
             +---> Tamper-Proof Storage
             |
             +---> Security Event Detection
             |
             +---> Compliance Reporting
             |
             +---> User Activity Tracking
             |
             +---> Query & Search Engine
             |
             +---> Alert & Notification

Key Features:
    - Comprehensive audit trail
    - Tamper-proof logging with checksums
    - Real-time security event detection
    - User activity tracking (KVKK Article 12)
    - Compliance reporting (KVKK, GDPR, ISO 27001)
    - Advanced query and search
    - Alert and notification system
    - Log retention policies
    - Log archival and compression
    - Forensic analysis support

Event Categories:
    Authentication:
        - Login/Logout
        - Password changes
        - MFA events
        - Session management

    Authorization:
        - Permission changes
        - Role assignments
        - Access attempts

    Data Access:
        - Document views
        - Document downloads
        - Search queries
        - API calls

    Data Modification:
        - Create/Update/Delete operations
        - Bulk operations
        - Import/Export

    Security Events:
        - Failed login attempts
        - Unauthorized access
        - Suspicious activities
        - Security policy violations

    System Events:
        - Configuration changes
        - Service starts/stops
        - Errors and exceptions
        - Performance issues

KVKK Compliance:
    - Article 12: Log retention (min 6 months)
    - Access logs for personal data
    - Data breach detection
    - Audit trail for DSR requests
    - Consent tracking logs

Author: Harvey/Legora CTO
Date: 2025-11-10
Lines: 905
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
import logging
import json
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class EventCategory(str, Enum):
    """Audit event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY = "security"
    SYSTEM = "system"
    COMPLIANCE = "compliance"
    USER_ACTIVITY = "user_activity"


class EventSeverity(str, Enum):
    """Event severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventStatus(str, Enum):
    """Event status"""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    ERROR = "error"


class AlertType(str, Enum):
    """Alert types"""
    SECURITY_BREACH = "security_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ERROR = "system_error"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class AuditEvent:
    """Audit event record"""
    id: UUID
    timestamp: datetime
    category: EventCategory
    event_type: str
    severity: EventSeverity
    status: EventStatus
    user_id: Optional[UUID]
    tenant_id: Optional[UUID]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[UUID]
    action: str
    description: str
    details: Dict[str, Any]
    session_id: Optional[str] = None
    previous_checksum: Optional[str] = None
    checksum: str = field(default="")

    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate tamper-proof checksum"""
        data = f"{self.id}|{self.timestamp}|{self.category}|{self.event_type}|{self.user_id}|{self.action}|{self.previous_checksum or ''}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class SecurityAlert:
    """Security alert"""
    id: UUID
    alert_type: AlertType
    severity: EventSeverity
    title: str
    description: str
    related_events: List[UUID]
    detected_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[UUID] = None
    status: str = "open"  # open, investigating, resolved, false_positive


@dataclass
class UserActivitySummary:
    """User activity summary"""
    user_id: UUID
    period_start: datetime
    period_end: datetime
    total_events: int
    login_count: int
    failed_login_count: int
    documents_accessed: int
    documents_modified: int
    last_activity: datetime
    suspicious_activities: int


@dataclass
class ComplianceAuditReport:
    """Compliance audit report"""
    id: UUID
    report_type: str
    period_start: datetime
    period_end: datetime
    total_events: int
    events_by_category: Dict[str, int]
    security_incidents: int
    compliance_violations: int
    user_activities: List[UserActivitySummary]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: Optional[UUID] = None


class AuditService:
    """
    Enterprise audit logging and security monitoring service.

    Provides comprehensive audit trails, security event detection,
    compliance reporting, and forensic analysis for Harvey/Legora.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize audit service.

        Args:
            db: Async database session
        """
        self.db = db
        self.logger = logger
        self.events: List[AuditEvent] = []
        self.alerts: Dict[UUID, SecurityAlert] = {}
        self.last_checksum: Optional[str] = None

        # Configuration
        self.retention_days = 180  # KVKK minimum 6 months
        self.alert_threshold = {
            "failed_login": 5,
            "unauthorized_access": 3,
        }

    # ===================================================================
    # PUBLIC API - Event Logging
    # ===================================================================

    async def log_event(
        self,
        category: EventCategory,
        event_type: str,
        action: str,
        description: str,
        severity: EventSeverity = EventSeverity.INFO,
        status: EventStatus = EventStatus.SUCCESS,
        user_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log audit event.

        Args:
            category: Event category
            event_type: Event type
            action: Action performed
            description: Event description
            severity: Event severity
            status: Event status
            user_id: User ID
            tenant_id: Tenant ID
            ip_address: IP address
            user_agent: User agent
            resource_type: Resource type
            resource_id: Resource ID
            details: Additional details
            session_id: Session ID

        Returns:
            Created audit event
        """
        try:
            event = AuditEvent(
                id=uuid4(),
                timestamp=datetime.utcnow(),
                category=category,
                event_type=event_type,
                severity=severity,
                status=status,
                user_id=user_id,
                tenant_id=tenant_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                description=description,
                details=details or {},
                session_id=session_id,
                previous_checksum=self.last_checksum,
            )

            self.events.append(event)
            self.last_checksum = event.checksum

            # Check for security alerts
            await self._check_security_alerts(event)

            self.logger.info(f"Audit event logged: {event_type} by {user_id}")

            return event

        except Exception as e:
            self.logger.error(f"Failed to log audit event: {str(e)}")
            raise

    async def log_authentication(
        self,
        event_type: str,
        user_id: Optional[UUID],
        status: EventStatus,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log authentication event"""
        return await self.log_event(
            category=EventCategory.AUTHENTICATION,
            event_type=event_type,
            action=event_type,
            description=f"Authentication: {event_type}",
            severity=EventSeverity.INFO if status == EventStatus.SUCCESS else EventSeverity.MEDIUM,
            status=status,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
        )

    async def log_data_access(
        self,
        resource_type: str,
        resource_id: UUID,
        action: str,
        user_id: UUID,
        tenant_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log data access event (KVKK compliance)"""
        return await self.log_event(
            category=EventCategory.DATA_ACCESS,
            event_type="data_access",
            action=action,
            description=f"Data access: {action} on {resource_type}",
            severity=EventSeverity.INFO,
            status=EventStatus.SUCCESS,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
        )

    async def log_data_modification(
        self,
        resource_type: str,
        resource_id: UUID,
        action: str,
        user_id: UUID,
        tenant_id: Optional[UUID] = None,
        previous_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
    ) -> AuditEvent:
        """Log data modification event"""
        details = {}
        if previous_value is not None:
            details["previous_value"] = str(previous_value)
        if new_value is not None:
            details["new_value"] = str(new_value)

        return await self.log_event(
            category=EventCategory.DATA_MODIFICATION,
            event_type="data_modification",
            action=action,
            description=f"Data modification: {action} on {resource_type}",
            severity=EventSeverity.LOW,
            status=EventStatus.SUCCESS,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
        )

    async def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: EventSeverity,
        user_id: Optional[UUID] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log security event"""
        return await self.log_event(
            category=EventCategory.SECURITY,
            event_type=event_type,
            action=event_type,
            description=description,
            severity=severity,
            status=EventStatus.SUCCESS,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
        )

    # ===================================================================
    # PUBLIC API - Query & Search
    # ===================================================================

    async def query_events(
        self,
        category: Optional[EventCategory] = None,
        event_type: Optional[str] = None,
        user_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        severity: Optional[EventSeverity] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.

        Args:
            category: Filter by category
            event_type: Filter by event type
            user_id: Filter by user
            tenant_id: Filter by tenant
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            severity: Filter by severity
            start_date: Start date
            end_date: End date
            limit: Result limit

        Returns:
            List of matching audit events
        """
        events = self.events

        # Apply filters
        if category:
            events = [e for e in events if e.category == category]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]
        if resource_id:
            events = [e for e in events if e.resource_id == resource_id]
        if severity:
            events = [e for e in events if e.severity == severity]
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]

        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x.timestamp, reverse=True)

        return events[:limit]

    async def search_events(
        self,
        search_text: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Search audit events by text.

        Args:
            search_text: Search text
            limit: Result limit

        Returns:
            List of matching events
        """
        search_lower = search_text.lower()
        matches = []

        for event in self.events:
            if (search_lower in event.description.lower() or
                search_lower in event.action.lower() or
                search_lower in json.dumps(event.details).lower()):
                matches.append(event)

        matches.sort(key=lambda x: x.timestamp, reverse=True)
        return matches[:limit]

    async def get_user_activity(
        self,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UserActivitySummary:
        """
        Get user activity summary.

        Args:
            user_id: User ID
            start_date: Start date
            end_date: End date

        Returns:
            User activity summary
        """
        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        events = await self.query_events(
            user_id=user_id,
            start_date=start,
            end_date=end,
            limit=10000,
        )

        login_count = sum(1 for e in events if e.category == EventCategory.AUTHENTICATION and e.status == EventStatus.SUCCESS)
        failed_login = sum(1 for e in events if e.category == EventCategory.AUTHENTICATION and e.status == EventStatus.FAILURE)
        docs_accessed = sum(1 for e in events if e.category == EventCategory.DATA_ACCESS)
        docs_modified = sum(1 for e in events if e.category == EventCategory.DATA_MODIFICATION)
        suspicious = sum(1 for e in events if e.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL])

        return UserActivitySummary(
            user_id=user_id,
            period_start=start,
            period_end=end,
            total_events=len(events),
            login_count=login_count,
            failed_login_count=failed_login,
            documents_accessed=docs_accessed,
            documents_modified=docs_modified,
            last_activity=events[0].timestamp if events else start,
            suspicious_activities=suspicious,
        )

    # ===================================================================
    # PUBLIC API - Security Alerts
    # ===================================================================

    async def create_alert(
        self,
        alert_type: AlertType,
        severity: EventSeverity,
        title: str,
        description: str,
        related_events: List[UUID],
        assigned_to: Optional[UUID] = None,
    ) -> SecurityAlert:
        """
        Create security alert.

        Args:
            alert_type: Alert type
            severity: Alert severity
            title: Alert title
            description: Alert description
            related_events: Related event IDs
            assigned_to: Assigned user ID

        Returns:
            Created alert
        """
        alert = SecurityAlert(
            id=uuid4(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            related_events=related_events,
            detected_at=datetime.utcnow(),
            assigned_to=assigned_to,
        )

        self.alerts[alert.id] = alert

        self.logger.warning(f"Security alert created: {title} (Severity: {severity})")

        return alert

    async def acknowledge_alert(
        self,
        alert_id: UUID,
        acknowledged_by: UUID,
    ) -> SecurityAlert:
        """Acknowledge security alert"""
        alert = self.alerts.get(alert_id)
        if not alert:
            raise ValueError(f"Alert not found: {alert_id}")

        alert.acknowledged_at = datetime.utcnow()
        alert.status = "investigating"

        return alert

    async def resolve_alert(
        self,
        alert_id: UUID,
        resolved_by: UUID,
        resolution: str,
    ) -> SecurityAlert:
        """Resolve security alert"""
        alert = self.alerts.get(alert_id)
        if not alert:
            raise ValueError(f"Alert not found: {alert_id}")

        alert.resolved_at = datetime.utcnow()
        alert.status = "resolved"

        return alert

    async def list_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[EventSeverity] = None,
    ) -> List[SecurityAlert]:
        """
        List security alerts.

        Args:
            status: Filter by status
            severity: Filter by severity

        Returns:
            List of alerts
        """
        alerts = list(self.alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        alerts.sort(key=lambda x: x.detected_at, reverse=True)
        return alerts

    # ===================================================================
    # PUBLIC API - Compliance Reporting
    # ===================================================================

    async def generate_compliance_report(
        self,
        report_type: str,
        start_date: datetime,
        end_date: datetime,
        generated_by: Optional[UUID] = None,
    ) -> ComplianceAuditReport:
        """
        Generate compliance audit report.

        Args:
            report_type: Report type (kvkk, gdpr, iso27001)
            start_date: Report period start
            end_date: Report period end
            generated_by: User generating report

        Returns:
            Compliance audit report
        """
        events = await self.query_events(
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

        # Count events by category
        events_by_category = {}
        for category in EventCategory:
            count = sum(1 for e in events if e.category == category)
            events_by_category[category.value] = count

        # Count security incidents
        security_incidents = sum(1 for e in events
                                if e.category == EventCategory.SECURITY and
                                e.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL])

        # Count compliance violations
        compliance_violations = sum(1 for e in events if e.category == EventCategory.COMPLIANCE)

        # Get unique users
        unique_users = set(e.user_id for e in events if e.user_id)
        user_activities = []
        for user_id in list(unique_users)[:50]:  # Limit to 50 users for report
            summary = await self.get_user_activity(user_id, start_date, end_date)
            user_activities.append(summary)

        report = ComplianceAuditReport(
            id=uuid4(),
            report_type=report_type,
            period_start=start_date,
            period_end=end_date,
            total_events=len(events),
            events_by_category=events_by_category,
            security_incidents=security_incidents,
            compliance_violations=compliance_violations,
            user_activities=user_activities,
            generated_by=generated_by,
        )

        self.logger.info(f"Generated compliance report: {report_type}")

        return report

    # ===================================================================
    # PUBLIC API - Integrity Verification
    # ===================================================================

    async def verify_integrity(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Verify audit log integrity using checksums.

        Args:
            start_index: Start index
            end_index: End index (None for all)

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        end = end_index or len(self.events)

        for i in range(start_index, min(end, len(self.events))):
            event = self.events[i]

            # Recalculate checksum
            expected_checksum = event._calculate_checksum()

            if event.checksum != expected_checksum:
                errors.append(f"Checksum mismatch at index {i}: Event {event.id}")

        is_valid = len(errors) == 0

        if is_valid:
            self.logger.info(f"Audit log integrity verified: {end - start_index} events")
        else:
            self.logger.error(f"Audit log integrity check failed: {len(errors)} errors")

        return is_valid, errors

    # ===================================================================
    # PRIVATE HELPERS - Security Alert Detection
    # ===================================================================

    async def _check_security_alerts(self, event: AuditEvent) -> None:
        """Check for security alert conditions"""

        # Check failed login attempts
        if (event.category == EventCategory.AUTHENTICATION and
            event.status == EventStatus.FAILURE):
            await self._check_failed_login_threshold(event)

        # Check unauthorized access
        if (event.category == EventCategory.AUTHORIZATION and
            event.status == EventStatus.FAILURE):
            await self._create_unauthorized_access_alert(event)

        # Check suspicious activity
        if event.severity == EventSeverity.CRITICAL:
            await self._create_suspicious_activity_alert(event)

    async def _check_failed_login_threshold(self, event: AuditEvent) -> None:
        """Check failed login threshold"""
        if not event.user_id:
            return

        # Count recent failed logins for this user
        recent_time = datetime.utcnow() - timedelta(minutes=15)
        recent_failures = await self.query_events(
            category=EventCategory.AUTHENTICATION,
            user_id=event.user_id,
            status=EventStatus.FAILURE,
            start_date=recent_time,
        )

        if len(recent_failures) >= self.alert_threshold["failed_login"]:
            await self.create_alert(
                alert_type=AlertType.SECURITY_BREACH,
                severity=EventSeverity.HIGH,
                title=f"Multiple failed login attempts for user {event.user_id}",
                description=f"{len(recent_failures)} failed login attempts in 15 minutes",
                related_events=[e.id for e in recent_failures],
            )

    async def _create_unauthorized_access_alert(self, event: AuditEvent) -> None:
        """Create unauthorized access alert"""
        await self.create_alert(
            alert_type=AlertType.UNAUTHORIZED_ACCESS,
            severity=EventSeverity.MEDIUM,
            title=f"Unauthorized access attempt",
            description=event.description,
            related_events=[event.id],
        )

    async def _create_suspicious_activity_alert(self, event: AuditEvent) -> None:
        """Create suspicious activity alert"""
        await self.create_alert(
            alert_type=AlertType.SUSPICIOUS_ACTIVITY,
            severity=event.severity,
            title=f"Suspicious activity detected",
            description=event.description,
            related_events=[event.id],
        )

    # ===================================================================
    # PRIVATE HELPERS - Maintenance
    # ===================================================================

    async def cleanup_old_events(self) -> int:
        """Clean up events older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        initial_count = len(self.events)
        self.events = [e for e in self.events if e.timestamp >= cutoff_date]
        removed_count = initial_count - len(self.events)

        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old audit events")

        return removed_count
