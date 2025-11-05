"""
Notification model for multi-channel notifications in Turkish Legal AI.

This module provides the Notification model for managing user notifications:
- In-app notifications
- Email notifications
- SMS notifications
- Push notifications
- Webhook notifications
- Multi-channel delivery
- Priority-based routing
- Read/unread tracking
- Notification preferences
- Delivery status tracking

Notification Types:
    - SYSTEM: System announcements
    - DOCUMENT: Document-related (uploaded, processed, analyzed)
    - TASK: Task assignments and updates
    - DEADLINE: Deadline reminders
    - APPROVAL: Approval requests
    - DATA_REQUEST: KVKK data request updates
    - CHAT: Chat mentions and replies
    - ALERT: Critical alerts
    - MARKETING: Marketing communications (opt-in)

Notification Channels:
    - IN_APP: Application notifications
    - EMAIL: Email notifications
    - SMS: SMS text messages
    - PUSH: Mobile/web push notifications
    - WEBHOOK: Webhook delivery

Priority Levels:
    - LOW: Low priority, batch delivery
    - NORMAL: Standard priority
    - HIGH: High priority, immediate
    - URGENT: Critical, multi-channel

Example:
    >>> # Send document processing notification
    >>> notification = Notification.create(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     notification_type=NotificationType.DOCUMENT,
    ...     title="Belge İşleme Tamamlandı",
    ...     message="Sözleşme.pdf başarıyla işlendi",
    ...     channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
    ...     priority=NotificationPriority.NORMAL,
    ...     action_url="/documents/123"
    ... )
    >>> 
    >>> # Mark as read
    >>> notification.mark_read()
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
    ForeignKey,
    Integer,
    String,
    Text,
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


class NotificationType(str, enum.Enum):
    """
    Notification type classification.
    
    Types:
    - SYSTEM: System announcements and updates
    - DOCUMENT: Document-related notifications
    - TASK: Task assignments and updates
    - DEADLINE: Deadline and reminder notifications
    - APPROVAL: Approval requests and decisions
    - DATA_REQUEST: KVKK data request updates
    - CHAT: Chat mentions and replies
    - ALERT: Critical alerts and warnings
    - MARKETING: Marketing communications (opt-in)
    - SECURITY: Security alerts (login, password, etc.)
    """
    
    SYSTEM = "system"
    DOCUMENT = "document"
    TASK = "task"
    DEADLINE = "deadline"
    APPROVAL = "approval"
    DATA_REQUEST = "data_request"
    CHAT = "chat"
    ALERT = "alert"
    MARKETING = "marketing"
    SECURITY = "security"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.SYSTEM: "Sistem Bildirimi",
            self.DOCUMENT: "Belge Bildirimi",
            self.TASK: "Görev Bildirimi",
            self.DEADLINE: "Son Tarih Hatırlatması",
            self.APPROVAL: "Onay İsteği",
            self.DATA_REQUEST: "Veri Talebi",
            self.CHAT: "Sohbet Bildirimi",
            self.ALERT: "Uyarı",
            self.MARKETING: "Pazarlama",
            self.SECURITY: "Güvenlik Bildirimi",
        }
        return names.get(self, self.value)


class NotificationChannel(str, enum.Enum):
    """Notification delivery channel."""
    
    IN_APP = "in_app"      # Application notification center
    EMAIL = "email"        # Email notification
    SMS = "sms"            # SMS text message
    PUSH = "push"          # Mobile/web push notification
    WEBHOOK = "webhook"    # Webhook delivery
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.IN_APP: "Uygulama İçi",
            self.EMAIL: "E-posta",
            self.SMS: "SMS",
            self.PUSH: "Push Bildirimi",
            self.WEBHOOK: "Webhook",
        }
        return names.get(self, self.value)


class NotificationPriority(str, enum.Enum):
    """Notification priority level."""
    
    LOW = "low"            # Low priority, batch delivery
    NORMAL = "normal"      # Standard priority
    HIGH = "high"          # High priority, immediate delivery
    URGENT = "urgent"      # Critical, multi-channel delivery
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.LOW: "Düşük",
            self.NORMAL: "Normal",
            self.HIGH: "Yüksek",
            self.URGENT: "Acil",
        }
        return names.get(self, self.value)


class NotificationStatus(str, enum.Enum):
    """Notification delivery status."""
    
    PENDING = "pending"        # Queued for delivery
    SENT = "sent"              # Successfully sent
    DELIVERED = "delivered"    # Confirmed delivery
    FAILED = "failed"          # Delivery failed
    CANCELLED = "cancelled"    # Cancelled before delivery
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# NOTIFICATION MODEL
# =============================================================================


class Notification(Base, BaseModelMixin, TenantMixin, SoftDeleteMixin):
    """
    Notification model for multi-channel user notifications.
    
    Manages notifications across multiple channels:
    - In-app notification center
    - Email notifications
    - SMS notifications
    - Push notifications
    - Webhook delivery
    
    Notification Lifecycle:
    1. Create notification with channels
    2. Queue for delivery
    3. Deliver to each channel
    4. Track delivery status
    5. User reads (in-app)
    
    Channel Priority:
        URGENT: All channels immediately
        HIGH: In-app + Email immediately, SMS if enabled
        NORMAL: In-app immediately, Email batched
        LOW: In-app only, batched
    
    Attributes:
        user_id: Recipient user
        user: User relationship
        
        notification_type: Type of notification
        priority: Priority level
        
        title: Notification title (short)
        message: Notification message (detailed)
        
        channels: Delivery channels (array)
        
        is_read: Read status (for in-app)
        read_at: When notification was read
        
        action_url: URL to navigate to (optional)
        action_label: Action button label (optional)
        
        related_entity_type: Related entity type
        related_entity_id: Related entity UUID
        
        delivery_status: Per-channel delivery status (JSON)
        
        sent_at: When notification was sent
        delivered_at: When delivery confirmed
        
        expires_at: Notification expiration (optional)
        
        metadata: Additional context (icons, images, etc.)
        
        requires_action: User action required flag
        action_taken_at: When action was taken
        
    Relationships:
        tenant: Parent tenant
        user: Recipient user
    """
    
    __tablename__ = "notifications"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Recipient user",
    )
    
    user = relationship(
        "User",
        back_populates="notifications",
    )
    
    # =========================================================================
    # NOTIFICATION CLASSIFICATION
    # =========================================================================
    
    notification_type = Column(
        Enum(NotificationType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of notification",
    )
    
    priority = Column(
        Enum(NotificationPriority, native_enum=False, length=50),
        nullable=False,
        default=NotificationPriority.NORMAL,
        index=True,
        comment="Priority level",
    )
    
    # =========================================================================
    # CONTENT
    # =========================================================================
    
    title = Column(
        String(500),
        nullable=False,
        comment="Notification title (short summary)",
    )
    
    message = Column(
        Text,
        nullable=False,
        comment="Notification message (detailed content)",
    )
    
    # =========================================================================
    # CHANNELS
    # =========================================================================
    
    channels = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Delivery channels (in_app, email, sms, push, webhook)",
    )
    
    # =========================================================================
    # READ STATUS (for in-app)
    # =========================================================================
    
    is_read = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Read status (for in-app notifications)",
    )
    
    read_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When notification was read",
    )
    
    # =========================================================================
    # ACTION
    # =========================================================================
    
    action_url = Column(
        String(2000),
        nullable=True,
        comment="URL to navigate to when clicked",
    )
    
    action_label = Column(
        String(100),
        nullable=True,
        comment="Action button label (e.g., 'Görüntüle', 'Onayla')",
    )
    
    # =========================================================================
    # RELATED ENTITY
    # =========================================================================
    
    related_entity_type = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Related entity type (document, task, data_request, etc.)",
    )
    
    related_entity_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Related entity UUID",
    )
    
    # =========================================================================
    # DELIVERY STATUS
    # =========================================================================
    
    delivery_status = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Per-channel delivery status (channel -> status mapping)",
    )
    
    sent_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When notification was sent to channels",
    )
    
    delivered_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When delivery was confirmed",
    )
    
    # =========================================================================
    # EXPIRATION
    # =========================================================================
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Notification expiration timestamp (optional)",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (icon, image_url, color, sound, etc.)",
    )
    
    # =========================================================================
    # ACTION TRACKING
    # =========================================================================
    
    requires_action = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Notification requires user action (approval, confirmation)",
    )
    
    action_taken_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When required action was taken",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user's unread notifications
        Index(
            "ix_notifications_user_unread",
            "user_id",
            "is_read",
            postgresql_where="is_read = false AND deleted_at IS NULL",
        ),
        
        # Index for user's notifications by type
        Index(
            "ix_notifications_user_type",
            "user_id",
            "notification_type",
            "created_at",
        ),
        
        # Index for priority notifications
        Index(
            "ix_notifications_priority",
            "priority",
            "sent_at",
            postgresql_where="priority IN ('high', 'urgent')",
        ),
        
        # Index for entity notifications
        Index(
            "ix_notifications_entity",
            "related_entity_type",
            "related_entity_id",
        ),
        
        # Index for pending delivery
        Index(
            "ix_notifications_pending",
            "sent_at",
            postgresql_where="sent_at IS NULL AND deleted_at IS NULL",
        ),
        
        # Index for expired notifications
        Index(
            "ix_notifications_expired",
            "expires_at",
            postgresql_where="expires_at IS NOT NULL",
        ),
    )
    
    # =========================================================================
    # NOTIFICATION CREATION
    # =========================================================================
    
    @classmethod
    def create(
        cls,
        user_id: UUIDType,
        tenant_id: UUIDType,
        notification_type: NotificationType,
        title: str,
        message: str,
        channels: list[NotificationChannel] | None = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        action_url: str | None = None,
        action_label: str | None = None,
        related_entity_type: str | None = None,
        related_entity_id: UUIDType | None = None,
        requires_action: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "Notification":
        """
        Create a new notification.
        
        Args:
            user_id: Recipient user UUID
            tenant_id: Tenant UUID
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            channels: Delivery channels
            priority: Priority level
            action_url: URL for action
            action_label: Action button label
            related_entity_type: Related entity type
            related_entity_id: Related entity UUID
            requires_action: Action required flag
            metadata: Additional context
            
        Returns:
            Notification: New notification instance
            
        Example:
            >>> notification = Notification.create(
            ...     user_id=user.id,
            ...     tenant_id=tenant.id,
            ...     notification_type=NotificationType.DOCUMENT,
            ...     title="Belge İşleme Tamamlandı",
            ...     message="Sözleşme.pdf başarıyla işlendi ve analiz edildi",
            ...     channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            ...     priority=NotificationPriority.NORMAL,
            ...     action_url="/documents/123",
            ...     action_label="Belgeyi Görüntüle",
            ...     related_entity_type="document",
            ...     related_entity_id=document_id,
            ...     metadata={"icon": "document-check", "color": "success"}
            ... )
        """
        # Default to in-app only
        if channels is None:
            channels = [NotificationChannel.IN_APP]
        
        # Convert enum to string
        channels_str = [ch.value if isinstance(ch, NotificationChannel) else ch for ch in channels]
        
        notification = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            notification_type=notification_type,
            title=title,
            message=message,
            channels=channels_str,
            priority=priority,
            action_url=action_url,
            action_label=action_label,
            related_entity_type=related_entity_type,
            related_entity_id=related_entity_id,
            requires_action=requires_action,
            metadata=metadata or {},
            delivery_status={},
        )
        
        logger.info(
            "Notification created",
            notification_id=str(notification.id),
            user_id=str(user_id),
            type=notification_type.value,
            channels=channels_str,
            priority=priority.value,
        )
        
        return notification
    
    # =========================================================================
    # READ MANAGEMENT
    # =========================================================================
    
    def mark_read(self) -> None:
        """Mark notification as read."""
        if not self.is_read:
            self.is_read = True
            self.read_at = datetime.now(timezone.utc)
            
            logger.debug(
                "Notification marked as read",
                notification_id=str(self.id),
                user_id=str(self.user_id),
            )
    
    def mark_unread(self) -> None:
        """Mark notification as unread."""
        if self.is_read:
            self.is_read = False
            self.read_at = None
            
            logger.debug(
                "Notification marked as unread",
                notification_id=str(self.id),
            )
    
    # =========================================================================
    # DELIVERY MANAGEMENT
    # =========================================================================
    
    def mark_sent(self, channel: NotificationChannel | str) -> None:
        """
        Mark notification as sent to a channel.
        
        Args:
            channel: Delivery channel
        """
        channel_str = channel.value if isinstance(channel, NotificationChannel) else channel
        
        if not self.delivery_status:
            self.delivery_status = {}
        
        self.delivery_status[channel_str] = {
            "status": NotificationStatus.SENT.value,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "delivery_status")
        
        # Update sent_at if first channel
        if not self.sent_at:
            self.sent_at = datetime.now(timezone.utc)
        
        logger.debug(
            "Notification sent",
            notification_id=str(self.id),
            channel=channel_str,
        )
    
    def mark_delivered(self, channel: NotificationChannel | str) -> None:
        """
        Mark notification as delivered to a channel.
        
        Args:
            channel: Delivery channel
        """
        channel_str = channel.value if isinstance(channel, NotificationChannel) else channel
        
        if not self.delivery_status:
            self.delivery_status = {}
        
        self.delivery_status[channel_str] = {
            "status": NotificationStatus.DELIVERED.value,
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "delivery_status")
        
        # Update delivered_at if all channels delivered
        if not self.delivered_at and self.is_fully_delivered():
            self.delivered_at = datetime.now(timezone.utc)
        
        logger.debug(
            "Notification delivered",
            notification_id=str(self.id),
            channel=channel_str,
        )
    
    def mark_failed(self, channel: NotificationChannel | str, error: str) -> None:
        """
        Mark notification delivery as failed for a channel.
        
        Args:
            channel: Delivery channel
            error: Error message
        """
        channel_str = channel.value if isinstance(channel, NotificationChannel) else channel
        
        if not self.delivery_status:
            self.delivery_status = {}
        
        self.delivery_status[channel_str] = {
            "status": NotificationStatus.FAILED.value,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "delivery_status")
        
        logger.warning(
            "Notification delivery failed",
            notification_id=str(self.id),
            channel=channel_str,
            error=error,
        )
    
    def is_fully_delivered(self) -> bool:
        """
        Check if notification is delivered to all channels.
        
        Returns:
            bool: True if delivered to all channels
        """
        if not self.channels or not self.delivery_status:
            return False
        
        for channel in self.channels:
            status = self.delivery_status.get(channel, {}).get("status")
            if status != NotificationStatus.DELIVERED.value:
                return False
        
        return True
    
    def get_failed_channels(self) -> list[str]:
        """
        Get list of channels that failed delivery.
        
        Returns:
            list: Failed channel names
        """
        if not self.delivery_status:
            return []
        
        failed = []
        for channel, status_info in self.delivery_status.items():
            if status_info.get("status") == NotificationStatus.FAILED.value:
                failed.append(channel)
        
        return failed
    
    # =========================================================================
    # ACTION MANAGEMENT
    # =========================================================================
    
    def mark_action_taken(self) -> None:
        """Mark required action as taken."""
        if self.requires_action:
            self.action_taken_at = datetime.now(timezone.utc)
            
            logger.info(
                "Notification action taken",
                notification_id=str(self.id),
            )
    
    def is_action_pending(self) -> bool:
        """
        Check if action is still pending.
        
        Returns:
            bool: True if action required and not taken
        """
        return self.requires_action and not self.action_taken_at
    
    # =========================================================================
    # EXPIRATION
    # =========================================================================
    
    def is_expired(self) -> bool:
        """
        Check if notification is expired.
        
        Returns:
            bool: True if expired
        """
        if not self.expires_at:
            return False
        
        return datetime.now(timezone.utc) > self.expires_at
    
    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================
    
    @classmethod
    def mark_all_read_for_user(cls, user_id: UUIDType) -> int:
        """
        Mark all notifications as read for a user.
        
        This is a placeholder. In production, this would be:
        session.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.is_read == False
        ).update({"is_read": True, "read_at": datetime.now(timezone.utc)})
        
        Args:
            user_id: User UUID
            
        Returns:
            int: Number of notifications marked as read
        """
        # Placeholder
        logger.info(
            "Marked all notifications as read",
            user_id=str(user_id),
        )
        return 0
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("title")
    def validate_title(self, key: str, title: str) -> str:
        """Validate title."""
        if not title or not title.strip():
            raise ValidationError(
                message="Title cannot be empty",
                field="title",
            )
        
        return title.strip()
    
    @validates("message")
    def validate_message(self, key: str, message: str) -> str:
        """Validate message."""
        if not message or not message.strip():
            raise ValidationError(
                message="Message cannot be empty",
                field="message",
            )
        
        return message.strip()
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Notification("
            f"id={self.id}, "
            f"type={self.notification_type.value}, "
            f"is_read={self.is_read}"
            f")>"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        
        # Add display names
        data["notification_type_display"] = self.notification_type.display_name_tr
        data["priority_display"] = self.priority.display_name_tr
        
        # Add computed fields
        data["is_expired"] = self.is_expired()
        data["is_action_pending"] = self.is_action_pending()
        data["is_fully_delivered"] = self.is_fully_delivered()
        data["failed_channels"] = self.get_failed_channels()
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Notification",
    "NotificationType",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationStatus",
]

