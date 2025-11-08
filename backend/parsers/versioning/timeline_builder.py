"""Timeline Builder - Harvey/Legora CTO-Level Production-Grade
Builds timelines of Turkish legal document changes and events

Production Features:
- Timeline creation and management
- Temporal ordering of events
- Turkish legal event types (publication, amendment, repeal)
- Timeline visualization data generation
- Event filtering and querying
- Timeline export (JSON, visualization-ready)
- Multi-document timeline support
- Statistics tracking
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of timeline events"""
    PUBLICATION = "PUBLICATION"  # Yayım (publication)
    EFFECTIVITY = "EFFECTIVITY"  # Yürürlük (coming into force)
    AMENDMENT = "AMENDMENT"  # Değişiklik (amendment)
    REPEAL = "REPEAL"  # Mülga (repeal)
    CANCELLATION = "CANCELLATION"  # İptal (cancellation)
    SUSPENSION = "SUSPENSION"  # Askıya Alma (suspension)
    REVISION = "REVISION"  # Revizyon (revision)
    SNAPSHOT = "SNAPSHOT"  # Snapshot taken
    VALIDATION = "VALIDATION"  # Validation performed
    CUSTOM = "CUSTOM"  # Custom event


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = "CRITICAL"  # Critical event
    HIGH = "HIGH"  # High priority
    MEDIUM = "MEDIUM"  # Medium priority
    LOW = "LOW"  # Low priority
    INFO = "INFO"  # Informational


@dataclass
class TimelineEvent:
    """Represents a single event in timeline"""
    event_id: str
    event_type: EventType
    timestamp: str  # ISO format datetime

    # Event details
    title: str
    description: Optional[str] = None

    # Related entities
    document_id: Optional[str] = None
    version_id: Optional[str] = None
    snapshot_id: Optional[str] = None

    # Turkish legal references
    law_reference: Optional[str] = None  # e.g., "6698 sayılı Kanun"
    gazette_reference: Optional[str] = None  # Resmi Gazete referansı
    article_reference: Optional[str] = None  # Madde referansı

    # Event metadata
    priority: EventPriority = EventPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal relationships
    related_events: List[str] = field(default_factory=list)  # Related event IDs

    def __lt__(self, other: 'TimelineEvent') -> bool:
        """Compare events by timestamp for sorting"""
        return self.timestamp < other.timestamp

    def summary(self) -> str:
        """Get human-readable summary"""
        parts = [f"[{self.event_type.value}]"]
        parts.append(f"{self.title}")

        if self.law_reference:
            parts.append(f"({self.law_reference})")

        if self.timestamp:
            try:
                dt = datetime.fromisoformat(self.timestamp)
                date_str = dt.strftime('%Y-%m-%d')
                parts.append(f"- {date_str}")
            except:
                parts.append(f"- {self.timestamp[:10]}")

        return ' '.join(parts)


@dataclass
class Timeline:
    """Complete timeline with events"""
    timeline_id: str
    name: str

    # Events
    events: List[TimelineEvent] = field(default_factory=list)

    # Timeline metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None

    # Document associations
    document_ids: Set[str] = field(default_factory=set)

    # Time range
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: TimelineEvent) -> None:
        """Add event to timeline"""
        self.events.append(event)
        self.events.sort()  # Keep sorted by timestamp

        # Update document associations
        if event.document_id:
            self.document_ids.add(event.document_id)

        # Update time range
        if not self.start_date or event.timestamp < self.start_date:
            self.start_date = event.timestamp

        if not self.end_date or event.timestamp > self.end_date:
            self.end_date = event.timestamp

        self.updated_at = datetime.now().isoformat()

    def get_events_by_type(self, event_type: EventType) -> List[TimelineEvent]:
        """Get events of specific type"""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[TimelineEvent]:
        """Get events within date range"""
        return [
            e for e in self.events
            if start_date <= e.timestamp <= end_date
        ]

    def get_events_by_document(self, document_id: str) -> List[TimelineEvent]:
        """Get events for specific document"""
        return [e for e in self.events if e.document_id == document_id]

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Timeline: {self.name} ({self.timeline_id})")
        lines.append(f"Events: {len(self.events)}")
        lines.append(f"Documents: {len(self.document_ids)}")

        if self.start_date and self.end_date:
            lines.append(f"Period: {self.start_date[:10]} to {self.end_date[:10]}")

        # Event type distribution
        type_counts = defaultdict(int)
        for event in self.events:
            type_counts[event.event_type.value] += 1

        if type_counts:
            lines.append(f"\nEvent Distribution:")
            for event_type, count in sorted(type_counts.items()):
                lines.append(f"  - {event_type}: {count}")

        return '\n'.join(lines)


class TimelineBuilder:
    """Timeline Builder for Turkish Legal Documents

    Builds and manages timelines of document changes:
    - Create timelines from events
    - Add various event types
    - Query and filter events
    - Export timeline data
    - Generate visualization-ready data
    - Turkish legal event tracking

    Features:
    - Temporal ordering
    - Multi-document timelines
    - Event relationships
    - Turkish legal event types
    - Export to multiple formats
    - Statistics tracking
    """

    def __init__(self):
        """Initialize Timeline Builder"""
        # Timeline storage
        self.timelines: Dict[str, Timeline] = {}

        # Event storage (all events across timelines)
        self.events: Dict[str, TimelineEvent] = {}

        # Index by document
        self.document_index: Dict[str, Set[str]] = defaultdict(set)  # document_id -> event_ids

        # Index by event type
        self.type_index: Dict[EventType, Set[str]] = defaultdict(set)  # type -> event_ids

        # Statistics
        self.stats = {
            'total_timelines': 0,
            'total_events': 0,
            'event_types': defaultdict(int),
            'events_per_timeline': 0.0,
            'timelines_by_document': defaultdict(int),
        }

        logger.info("Initialized Timeline Builder")

    def create_timeline(
        self,
        name: str,
        **kwargs
    ) -> Timeline:
        """Create a new timeline

        Args:
            name: Timeline name
            **kwargs: Options
                - tags: Timeline tags
                - metadata: Timeline metadata

        Returns:
            Created Timeline
        """
        timeline_id = self._generate_timeline_id(name)

        timeline = Timeline(
            timeline_id=timeline_id,
            name=name,
            tags=kwargs.get('tags', []),
            metadata=kwargs.get('metadata', {})
        )

        self.timelines[timeline_id] = timeline

        # Update statistics
        self.stats['total_timelines'] += 1

        logger.info(f"Created timeline {timeline_id}: {name}")
        return timeline

    def add_event(
        self,
        timeline_id: str,
        event_type: EventType,
        timestamp: str,
        title: str,
        **kwargs
    ) -> TimelineEvent:
        """Add event to timeline

        Args:
            timeline_id: Timeline ID
            event_type: Type of event
            timestamp: Event timestamp (ISO format)
            title: Event title
            **kwargs: Options
                - description: Event description
                - document_id: Associated document
                - version_id: Associated version
                - snapshot_id: Associated snapshot
                - law_reference: Law reference
                - gazette_reference: Gazette reference
                - article_reference: Article reference
                - priority: Event priority
                - tags: Event tags
                - metadata: Event metadata

        Returns:
            Created TimelineEvent
        """
        timeline = self.timelines.get(timeline_id)
        if not timeline:
            logger.error(f"Timeline not found: {timeline_id}")
            raise ValueError(f"Timeline not found: {timeline_id}")

        # Generate event ID
        event_id = self._generate_event_id(event_type)

        # Create event
        event = TimelineEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            title=title,
            description=kwargs.get('description'),
            document_id=kwargs.get('document_id'),
            version_id=kwargs.get('version_id'),
            snapshot_id=kwargs.get('snapshot_id'),
            law_reference=kwargs.get('law_reference'),
            gazette_reference=kwargs.get('gazette_reference'),
            article_reference=kwargs.get('article_reference'),
            priority=kwargs.get('priority', EventPriority.MEDIUM),
            tags=kwargs.get('tags', []),
            metadata=kwargs.get('metadata', {})
        )

        # Add to timeline
        timeline.add_event(event)

        # Store event globally
        self.events[event_id] = event

        # Update indices
        if event.document_id:
            self.document_index[event.document_id].add(event_id)

        self.type_index[event_type].add(event_id)

        # Update statistics
        self._update_event_stats(event)

        logger.info(f"Added event {event_id} to timeline {timeline_id}")
        return event

    def add_publication_event(
        self,
        timeline_id: str,
        timestamp: str,
        document_id: str,
        **kwargs
    ) -> TimelineEvent:
        """Add publication event (Turkish: Yayım)

        Args:
            timeline_id: Timeline ID
            timestamp: Publication date
            document_id: Document ID
            **kwargs: Additional options

        Returns:
            Created event
        """
        title = kwargs.get('title', 'Document Published')

        return self.add_event(
            timeline_id=timeline_id,
            event_type=EventType.PUBLICATION,
            timestamp=timestamp,
            title=title,
            document_id=document_id,
            priority=EventPriority.HIGH,
            **kwargs
        )

    def add_amendment_event(
        self,
        timeline_id: str,
        timestamp: str,
        document_id: str,
        amending_law: str,
        **kwargs
    ) -> TimelineEvent:
        """Add amendment event (Turkish: Değişiklik)

        Args:
            timeline_id: Timeline ID
            timestamp: Amendment date
            document_id: Document ID
            amending_law: Amending law reference
            **kwargs: Additional options

        Returns:
            Created event
        """
        title = kwargs.get('title', f'Amended by {amending_law}')

        return self.add_event(
            timeline_id=timeline_id,
            event_type=EventType.AMENDMENT,
            timestamp=timestamp,
            title=title,
            document_id=document_id,
            law_reference=amending_law,
            priority=EventPriority.HIGH,
            **kwargs
        )

    def add_repeal_event(
        self,
        timeline_id: str,
        timestamp: str,
        document_id: str,
        **kwargs
    ) -> TimelineEvent:
        """Add repeal event (Turkish: Mülga)

        Args:
            timeline_id: Timeline ID
            timestamp: Repeal date
            document_id: Document ID
            **kwargs: Additional options

        Returns:
            Created event
        """
        title = kwargs.get('title', 'Document Repealed')

        return self.add_event(
            timeline_id=timeline_id,
            event_type=EventType.REPEAL,
            timestamp=timestamp,
            title=title,
            document_id=document_id,
            priority=EventPriority.CRITICAL,
            **kwargs
        )

    def build_from_versions(
        self,
        timeline_name: str,
        versions: List[Tuple[Any, Any]],  # List of (metadata, content)
        **kwargs
    ) -> Timeline:
        """Build timeline from version history

        Args:
            timeline_name: Name for timeline
            versions: List of (version_metadata, version_content) tuples
            **kwargs: Options

        Returns:
            Built Timeline
        """
        timeline = self.create_timeline(timeline_name, **kwargs)

        for metadata, content in versions:
            # Add publication event if available
            if hasattr(metadata, 'publication_date') and metadata.publication_date:
                self.add_event(
                    timeline_id=timeline.timeline_id,
                    event_type=EventType.PUBLICATION,
                    timestamp=metadata.publication_date,
                    title=f"Version {metadata.version_number} published",
                    version_id=metadata.version_id,
                    priority=EventPriority.HIGH
                )

            # Add effectivity event if available
            if hasattr(metadata, 'effectivity_date') and metadata.effectivity_date:
                self.add_event(
                    timeline_id=timeline.timeline_id,
                    event_type=EventType.EFFECTIVITY,
                    timestamp=metadata.effectivity_date,
                    title=f"Version {metadata.version_number} in effect",
                    version_id=metadata.version_id,
                    priority=EventPriority.MEDIUM
                )

            # Add amendment event if available
            if hasattr(metadata, 'amending_law') and metadata.amending_law:
                timestamp = metadata.publication_date or metadata.created_at
                self.add_event(
                    timeline_id=timeline.timeline_id,
                    event_type=EventType.AMENDMENT,
                    timestamp=timestamp,
                    title=f"Amended by {metadata.amending_law}",
                    version_id=metadata.version_id,
                    law_reference=metadata.amending_law,
                    priority=EventPriority.HIGH
                )

        logger.info(f"Built timeline from {len(versions)} versions")
        return timeline

    def build_from_snapshots(
        self,
        timeline_name: str,
        snapshots: List[Any],  # List of snapshot metadata
        **kwargs
    ) -> Timeline:
        """Build timeline from snapshots

        Args:
            timeline_name: Name for timeline
            snapshots: List of snapshot metadata
            **kwargs: Options

        Returns:
            Built Timeline
        """
        timeline = self.create_timeline(timeline_name, **kwargs)

        for snapshot in snapshots:
            # Add snapshot event
            self.add_event(
                timeline_id=timeline.timeline_id,
                event_type=EventType.SNAPSHOT,
                timestamp=snapshot.created_at,
                title=f"Snapshot created: {snapshot.snapshot_type.value}",
                snapshot_id=snapshot.snapshot_id,
                document_id=snapshot.document_id,
                priority=EventPriority.LOW
            )

        logger.info(f"Built timeline from {len(snapshots)} snapshots")
        return timeline

    def query_events(self, **filters) -> List[TimelineEvent]:
        """Query events with filters

        Args:
            **filters: Filter criteria
                - event_type: Event type
                - priority: Event priority
                - document_id: Document ID
                - start_date: Start date (inclusive)
                - end_date: End date (inclusive)
                - tags: List of tags (any match)

        Returns:
            List of matching events
        """
        results = list(self.events.values())

        # Apply filters
        if 'event_type' in filters:
            event_type = filters['event_type']
            if isinstance(event_type, str):
                event_type = EventType[event_type]
            results = [e for e in results if e.event_type == event_type]

        if 'priority' in filters:
            priority = filters['priority']
            if isinstance(priority, str):
                priority = EventPriority[priority]
            results = [e for e in results if e.priority == priority]

        if 'document_id' in filters:
            doc_id = filters['document_id']
            results = [e for e in results if e.document_id == doc_id]

        if 'start_date' in filters:
            start_date = filters['start_date']
            results = [e for e in results if e.timestamp >= start_date]

        if 'end_date' in filters:
            end_date = filters['end_date']
            results = [e for e in results if e.timestamp <= end_date]

        if 'tags' in filters:
            filter_tags = set(filters['tags'])
            results = [e for e in results if filter_tags.intersection(set(e.tags))]

        # Sort by timestamp
        results.sort()

        return results

    def get_timeline(self, timeline_id: str) -> Optional[Timeline]:
        """Get timeline by ID

        Args:
            timeline_id: Timeline ID

        Returns:
            Timeline or None
        """
        return self.timelines.get(timeline_id)

    def export_timeline(
        self,
        timeline_id: str,
        format: str = 'json'
    ) -> Optional[Any]:
        """Export timeline

        Args:
            timeline_id: Timeline ID
            format: Export format ('json', 'visualization')

        Returns:
            Exported data
        """
        timeline = self.get_timeline(timeline_id)
        if not timeline:
            return None

        if format == 'json':
            return self._export_json(timeline)
        elif format == 'visualization':
            return self._export_visualization(timeline)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None

    def _export_json(self, timeline: Timeline) -> Dict[str, Any]:
        """Export timeline as JSON

        Args:
            timeline: Timeline to export

        Returns:
            JSON-serializable dictionary
        """
        return {
            'timeline_id': timeline.timeline_id,
            'name': timeline.name,
            'created_at': timeline.created_at,
            'updated_at': timeline.updated_at,
            'start_date': timeline.start_date,
            'end_date': timeline.end_date,
            'tags': timeline.tags,
            'metadata': timeline.metadata,
            'events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp,
                    'title': event.title,
                    'description': event.description,
                    'document_id': event.document_id,
                    'version_id': event.version_id,
                    'snapshot_id': event.snapshot_id,
                    'law_reference': event.law_reference,
                    'gazette_reference': event.gazette_reference,
                    'article_reference': event.article_reference,
                    'priority': event.priority.value,
                    'tags': event.tags,
                    'metadata': event.metadata,
                    'related_events': event.related_events
                }
                for event in timeline.events
            ]
        }

    def _export_visualization(self, timeline: Timeline) -> Dict[str, Any]:
        """Export timeline in visualization-ready format

        Args:
            timeline: Timeline to export

        Returns:
            Visualization data structure
        """
        # Group events by year-month
        groups = defaultdict(list)

        for event in timeline.events:
            try:
                dt = datetime.fromisoformat(event.timestamp)
                period = dt.strftime('%Y-%m')
                groups[period].append(event)
            except:
                groups['unknown'].append(event)

        # Build visualization data
        vis_data = {
            'timeline': {
                'id': timeline.timeline_id,
                'name': timeline.name,
                'start': timeline.start_date,
                'end': timeline.end_date
            },
            'periods': [],
            'events': [],
            'statistics': {
                'total_events': len(timeline.events),
                'event_types': defaultdict(int),
                'documents': len(timeline.document_ids)
            }
        }

        # Add period data
        for period, events in sorted(groups.items()):
            if period != 'unknown':
                vis_data['periods'].append({
                    'period': period,
                    'count': len(events),
                    'event_ids': [e.event_id for e in events]
                })

        # Add event data
        for event in timeline.events:
            # Color coding by event type
            color = self._get_event_color(event.event_type)

            # Size by priority
            size = self._get_event_size(event.priority)

            event_data = {
                'id': event.event_id,
                'type': event.event_type.value,
                'timestamp': event.timestamp,
                'title': event.title,
                'description': event.description,
                'color': color,
                'size': size,
                'priority': event.priority.value,
                'document_id': event.document_id,
                'law_reference': event.law_reference,
                'tags': event.tags
            }

            vis_data['events'].append(event_data)

            # Update statistics
            vis_data['statistics']['event_types'][event.event_type.value] += 1

        return vis_data

    def _get_event_color(self, event_type: EventType) -> str:
        """Get color for event type (for visualization)"""
        colors = {
            EventType.PUBLICATION: '#4CAF50',  # Green
            EventType.EFFECTIVITY: '#2196F3',  # Blue
            EventType.AMENDMENT: '#FF9800',  # Orange
            EventType.REPEAL: '#F44336',  # Red
            EventType.CANCELLATION: '#9C27B0',  # Purple
            EventType.SUSPENSION: '#FFC107',  # Amber
            EventType.REVISION: '#00BCD4',  # Cyan
            EventType.SNAPSHOT: '#9E9E9E',  # Grey
            EventType.VALIDATION: '#8BC34A',  # Light Green
            EventType.CUSTOM: '#607D8B',  # Blue Grey
        }
        return colors.get(event_type, '#000000')

    def _get_event_size(self, priority: EventPriority) -> int:
        """Get size for priority (for visualization)"""
        sizes = {
            EventPriority.CRITICAL: 10,
            EventPriority.HIGH: 8,
            EventPriority.MEDIUM: 6,
            EventPriority.LOW: 4,
            EventPriority.INFO: 3,
        }
        return sizes.get(priority, 5)

    def merge_timelines(
        self,
        timeline_ids: List[str],
        merged_name: str,
        **kwargs
    ) -> Timeline:
        """Merge multiple timelines

        Args:
            timeline_ids: List of timeline IDs to merge
            merged_name: Name for merged timeline
            **kwargs: Options

        Returns:
            Merged Timeline
        """
        merged = self.create_timeline(merged_name, **kwargs)

        for timeline_id in timeline_ids:
            timeline = self.get_timeline(timeline_id)
            if timeline:
                for event in timeline.events:
                    merged.add_event(event)

        logger.info(f"Merged {len(timeline_ids)} timelines into {merged.timeline_id}")
        return merged

    def filter_timeline(
        self,
        timeline_id: str,
        filtered_name: str,
        **filters
    ) -> Optional[Timeline]:
        """Create filtered timeline

        Args:
            timeline_id: Source timeline ID
            filtered_name: Name for filtered timeline
            **filters: Filter criteria

        Returns:
            Filtered Timeline or None
        """
        source = self.get_timeline(timeline_id)
        if not source:
            return None

        filtered = self.create_timeline(filtered_name)

        # Get filtered events
        events = self.query_events(**filters)

        # Add events that are in source timeline
        source_event_ids = {e.event_id for e in source.events}
        for event in events:
            if event.event_id in source_event_ids:
                filtered.add_event(event)

        logger.info(f"Created filtered timeline with {len(filtered.events)} events")
        return filtered

    def _generate_timeline_id(self, name: str) -> str:
        """Generate unique timeline ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        # Create safe name
        safe_name = ''.join(c if c.isalnum() else '_' for c in name)[:20]
        return f"TL_{safe_name}_{timestamp}"

    def _generate_event_id(self, event_type: EventType) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        type_code = event_type.value[:4]
        return f"EVT_{type_code}_{timestamp}"

    def _update_event_stats(self, event: TimelineEvent) -> None:
        """Update statistics"""
        self.stats['total_events'] += 1
        self.stats['event_types'][event.event_type.value] += 1

        if event.document_id:
            self.stats['timelines_by_document'][event.document_id] += 1

        # Update average
        if self.stats['total_timelines'] > 0:
            self.stats['events_per_timeline'] = self.stats['total_events'] / self.stats['total_timelines']

    def get_stats(self) -> Dict[str, Any]:
        """Get builder statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_timelines': 0,
            'total_events': 0,
            'event_types': defaultdict(int),
            'events_per_timeline': 0.0,
            'timelines_by_document': defaultdict(int),
        }
        logger.info("Statistics reset")


__all__ = [
    'TimelineBuilder',
    'Timeline',
    'TimelineEvent',
    'EventType',
    'EventPriority'
]
