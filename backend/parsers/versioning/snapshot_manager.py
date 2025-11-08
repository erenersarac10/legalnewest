"""Snapshot Manager - Harvey/Legora CTO-Level Production-Grade
Manages snapshots of parsed Turkish legal documents

Production Features:
- Snapshot creation and storage
- Snapshot retrieval and querying
- Snapshot comparison
- Delta compression for storage efficiency
- Snapshot metadata and tagging
- Snapshot expiration and cleanup
- Export/import functionality
- Incremental snapshots
- Statistics tracking
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
import gzip
import base64
from datetime import datetime, timedelta
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class SnapshotType(Enum):
    """Types of snapshots"""
    FULL = "FULL"  # Full document snapshot
    INCREMENTAL = "INCREMENTAL"  # Only changes from previous snapshot
    DELTA = "DELTA"  # Delta-compressed snapshot
    METADATA_ONLY = "METADATA_ONLY"  # Only metadata, no content


class CompressionType(Enum):
    """Compression types for snapshots"""
    NONE = "NONE"  # No compression
    GZIP = "GZIP"  # Gzip compression
    DELTA = "DELTA"  # Delta compression
    HYBRID = "HYBRID"  # Delta + gzip


class SnapshotStatus(Enum):
    """Snapshot status"""
    ACTIVE = "ACTIVE"  # Active snapshot
    ARCHIVED = "ARCHIVED"  # Archived
    EXPIRED = "EXPIRED"  # Expired
    DELETED = "DELETED"  # Marked for deletion


@dataclass
class SnapshotMetadata:
    """Metadata for a snapshot"""
    snapshot_id: str
    snapshot_type: SnapshotType

    # Timing
    created_at: str
    expires_at: Optional[str] = None

    # Document info
    document_id: Optional[str] = None
    document_type: Optional[str] = None  # law, regulation, decision

    # Version info
    version_id: Optional[str] = None
    version_number: Optional[str] = None

    # Compression
    compression_type: CompressionType = CompressionType.NONE
    compressed_size: int = 0
    uncompressed_size: int = 0
    compression_ratio: float = 0.0

    # Parent snapshot (for incremental)
    parent_snapshot_id: Optional[str] = None

    # Status
    status: SnapshotStatus = SnapshotStatus.ACTIVE

    # Tags and labels
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    # Additional metadata
    notes: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotData:
    """Complete snapshot with metadata and content"""
    metadata: SnapshotMetadata
    content: Any  # Document content (may be compressed)

    def get_uncompressed_content(self) -> Any:
        """Get uncompressed content

        Returns:
            Uncompressed document content
        """
        if self.metadata.compression_type == CompressionType.NONE:
            return self.content

        elif self.metadata.compression_type == CompressionType.GZIP:
            # Decompress gzip
            compressed_bytes = base64.b64decode(self.content)
            decompressed_bytes = gzip.decompress(compressed_bytes)
            return json.loads(decompressed_bytes.decode('utf-8'))

        else:
            # For delta/hybrid, return as-is (needs parent to reconstruct)
            return self.content


@dataclass
class SnapshotComparison:
    """Result of comparing two snapshots"""
    snapshot_a_id: str
    snapshot_b_id: str

    # Differences
    differences: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    total_differences: int = 0
    fields_added: int = 0
    fields_removed: int = 0
    fields_modified: int = 0

    # Similarity
    similarity_score: float = 0.0

    # Metadata
    comparison_time: float = 0.0
    compared_at: Optional[str] = None

    def add_difference(self, field: str, value_a: Any, value_b: Any) -> None:
        """Add a difference"""
        self.differences.append({
            'field': field,
            'value_a': value_a,
            'value_b': value_b
        })
        self.total_differences += 1

        if value_a is None:
            self.fields_added += 1
        elif value_b is None:
            self.fields_removed += 1
        else:
            self.fields_modified += 1

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Snapshot Comparison: {self.snapshot_a_id} vs {self.snapshot_b_id}")
        lines.append(f"Total Differences: {self.total_differences}")
        lines.append(f"  Added: {self.fields_added}")
        lines.append(f"  Removed: {self.fields_removed}")
        lines.append(f"  Modified: {self.fields_modified}")
        lines.append(f"Similarity: {self.similarity_score:.3f}")
        lines.append(f"Comparison Time: {self.comparison_time:.3f}s")

        if self.differences:
            lines.append(f"\nTop Differences:")
            for diff in self.differences[:5]:
                lines.append(f"  - {diff['field']}: {diff['value_a']} → {diff['value_b']}")

        return '\n'.join(lines)


class SnapshotManager:
    """Snapshot Manager for Turkish Legal Documents

    Manages document snapshots:
    - Create full and incremental snapshots
    - Store snapshots with compression
    - Retrieve and query snapshots
    - Compare snapshots
    - Snapshot expiration and cleanup
    - Export/import functionality

    Features:
    - Multiple compression strategies
    - Delta compression for efficiency
    - Automatic expiration
    - Tag-based organization
    - Statistics tracking
    """

    # Default expiration (30 days)
    DEFAULT_EXPIRATION_DAYS = 30

    # Compression threshold (compress if larger than 1KB)
    COMPRESSION_THRESHOLD = 1024

    def __init__(
        self,
        auto_compress: bool = True,
        default_expiration_days: int = DEFAULT_EXPIRATION_DAYS
    ):
        """Initialize Snapshot Manager

        Args:
            auto_compress: Automatically compress large snapshots
            default_expiration_days: Default expiration in days
        """
        self.auto_compress = auto_compress
        self.default_expiration_days = default_expiration_days

        # Snapshot storage
        self.snapshots: Dict[str, SnapshotData] = {}

        # Index by document ID
        self.document_index: Dict[str, List[str]] = defaultdict(list)

        # Index by tags
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Statistics
        self.stats = {
            'total_snapshots': 0,
            'active_snapshots': 0,
            'total_size_bytes': 0,
            'compressed_size_bytes': 0,
            'snapshot_types': defaultdict(int),
            'compression_types': defaultdict(int),
            'expired_snapshots': 0,
            'deleted_snapshots': 0,
            'average_compression_ratio': 0.0,
        }

        logger.info(f"Initialized Snapshot Manager (auto_compress={auto_compress})")

    def create_snapshot(
        self,
        content: Any,
        snapshot_type: SnapshotType = SnapshotType.FULL,
        **kwargs
    ) -> SnapshotMetadata:
        """Create a new snapshot

        Args:
            content: Document content to snapshot
            snapshot_type: Type of snapshot
            **kwargs: Options
                - document_id: Document identifier
                - document_type: Document type
                - version_id: Version identifier
                - version_number: Version number
                - parent_snapshot_id: Parent snapshot for incremental
                - tags: List of tags
                - labels: Dictionary of labels
                - notes: Snapshot notes
                - expiration_days: Days until expiration
                - compression: Force compression type

        Returns:
            SnapshotMetadata for created snapshot
        """
        start_time = time.time()

        # Generate snapshot ID
        snapshot_id = self._generate_snapshot_id(snapshot_type)

        # Calculate expiration
        expiration_days = kwargs.get('expiration_days', self.default_expiration_days)
        expires_at = None
        if expiration_days:
            expires_at = (datetime.now() + timedelta(days=expiration_days)).isoformat()

        # Calculate sizes
        uncompressed_size = self._calculate_size(content)

        # Determine compression
        compression_type = kwargs.get('compression', None)
        if compression_type is None:
            if self.auto_compress and uncompressed_size > self.COMPRESSION_THRESHOLD:
                compression_type = CompressionType.GZIP
            else:
                compression_type = CompressionType.NONE

        # Compress content if needed
        stored_content, compressed_size = self._compress_content(content, compression_type)

        # Calculate compression ratio
        if uncompressed_size > 0:
            compression_ratio = compressed_size / uncompressed_size
        else:
            compression_ratio = 1.0

        # Create metadata
        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            snapshot_type=snapshot_type,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
            document_id=kwargs.get('document_id'),
            document_type=kwargs.get('document_type'),
            version_id=kwargs.get('version_id'),
            version_number=kwargs.get('version_number'),
            compression_type=compression_type,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            compression_ratio=compression_ratio,
            parent_snapshot_id=kwargs.get('parent_snapshot_id'),
            status=SnapshotStatus.ACTIVE,
            tags=kwargs.get('tags', []),
            labels=kwargs.get('labels', {}),
            notes=kwargs.get('notes')
        )

        # Create snapshot
        snapshot = SnapshotData(metadata=metadata, content=stored_content)

        # Store snapshot
        self.snapshots[snapshot_id] = snapshot

        # Update indices
        if metadata.document_id:
            self.document_index[metadata.document_id].append(snapshot_id)

        for tag in metadata.tags:
            self.tag_index[tag].add(snapshot_id)

        # Update statistics
        self._update_snapshot_stats(metadata)

        logger.info(f"Created snapshot {snapshot_id} (type: {snapshot_type.value}, "
                   f"size: {compressed_size} bytes, ratio: {compression_ratio:.2f})")

        return metadata

    def get_snapshot(self, snapshot_id: str) -> Optional[SnapshotData]:
        """Get snapshot by ID

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            SnapshotData or None
        """
        snapshot = self.snapshots.get(snapshot_id)

        if snapshot:
            # Check if expired
            if self._is_expired(snapshot.metadata):
                snapshot.metadata.status = SnapshotStatus.EXPIRED
                logger.warning(f"Snapshot {snapshot_id} has expired")

        return snapshot

    def get_snapshots_by_document(self, document_id: str) -> List[SnapshotData]:
        """Get all snapshots for a document

        Args:
            document_id: Document identifier

        Returns:
            List of snapshots sorted by creation time
        """
        snapshot_ids = self.document_index.get(document_id, [])
        snapshots = [self.snapshots[sid] for sid in snapshot_ids if sid in self.snapshots]

        # Sort by creation time
        snapshots.sort(key=lambda s: s.metadata.created_at, reverse=True)

        return snapshots

    def get_snapshots_by_tag(self, tag: str) -> List[SnapshotData]:
        """Get snapshots by tag

        Args:
            tag: Tag to search for

        Returns:
            List of snapshots with the tag
        """
        snapshot_ids = self.tag_index.get(tag, set())
        return [self.snapshots[sid] for sid in snapshot_ids if sid in self.snapshots]

    def query_snapshots(self, **filters) -> List[SnapshotData]:
        """Query snapshots with filters

        Args:
            **filters: Filter criteria
                - document_type: Document type
                - snapshot_type: Snapshot type
                - status: Snapshot status
                - tags: List of tags (all must match)
                - created_after: Created after date
                - created_before: Created before date

        Returns:
            List of matching snapshots
        """
        results = list(self.snapshots.values())

        # Apply filters
        if 'document_type' in filters:
            results = [s for s in results if s.metadata.document_type == filters['document_type']]

        if 'snapshot_type' in filters:
            snapshot_type = filters['snapshot_type']
            if isinstance(snapshot_type, str):
                snapshot_type = SnapshotType[snapshot_type]
            results = [s for s in results if s.metadata.snapshot_type == snapshot_type]

        if 'status' in filters:
            status = filters['status']
            if isinstance(status, str):
                status = SnapshotStatus[status]
            results = [s for s in results if s.metadata.status == status]

        if 'tags' in filters:
            required_tags = set(filters['tags'])
            results = [s for s in results if required_tags.issubset(set(s.metadata.tags))]

        if 'created_after' in filters:
            after = filters['created_after']
            results = [s for s in results if s.metadata.created_at >= after]

        if 'created_before' in filters:
            before = filters['created_before']
            results = [s for s in results if s.metadata.created_at <= before]

        return results

    def compare_snapshots(
        self,
        snapshot_a_id: str,
        snapshot_b_id: str
    ) -> SnapshotComparison:
        """Compare two snapshots

        Args:
            snapshot_a_id: First snapshot ID
            snapshot_b_id: Second snapshot ID

        Returns:
            SnapshotComparison result
        """
        start_time = time.time()

        snapshot_a = self.get_snapshot(snapshot_a_id)
        snapshot_b = self.get_snapshot(snapshot_b_id)

        comparison = SnapshotComparison(
            snapshot_a_id=snapshot_a_id,
            snapshot_b_id=snapshot_b_id,
            compared_at=datetime.now().isoformat()
        )

        if not snapshot_a or not snapshot_b:
            logger.error(f"Snapshot not found: {snapshot_a_id} or {snapshot_b_id}")
            comparison.comparison_time = time.time() - start_time
            return comparison

        # Get uncompressed content
        content_a = snapshot_a.get_uncompressed_content()
        content_b = snapshot_b.get_uncompressed_content()

        # Compare
        if isinstance(content_a, dict) and isinstance(content_b, dict):
            self._compare_dicts(content_a, content_b, comparison, '')
        else:
            # Simple comparison
            if content_a != content_b:
                comparison.add_difference('content', content_a, content_b)

        # Calculate similarity
        if comparison.total_differences == 0:
            comparison.similarity_score = 1.0
        else:
            # Rough similarity based on differences
            total_fields = len(self._flatten_dict(content_a)) if isinstance(content_a, dict) else 1
            comparison.similarity_score = max(0.0, 1.0 - (comparison.total_differences / total_fields))

        comparison.comparison_time = time.time() - start_time

        logger.info(f"Compared snapshots: {comparison.total_differences} differences found")

        return comparison

    def create_incremental_snapshot(
        self,
        content: Any,
        parent_snapshot_id: str,
        **kwargs
    ) -> SnapshotMetadata:
        """Create incremental snapshot (only changes from parent)

        Args:
            content: Current document content
            parent_snapshot_id: Parent snapshot ID
            **kwargs: Additional options

        Returns:
            SnapshotMetadata for incremental snapshot
        """
        parent_snapshot = self.get_snapshot(parent_snapshot_id)
        if not parent_snapshot:
            logger.error(f"Parent snapshot not found: {parent_snapshot_id}")
            raise ValueError(f"Parent snapshot not found: {parent_snapshot_id}")

        # Get parent content
        parent_content = parent_snapshot.get_uncompressed_content()

        # Calculate delta
        delta = self._calculate_delta(parent_content, content)

        # Create snapshot with delta
        kwargs['parent_snapshot_id'] = parent_snapshot_id
        return self.create_snapshot(
            delta,
            snapshot_type=SnapshotType.INCREMENTAL,
            **kwargs
        )

    def restore_snapshot(self, snapshot_id: str) -> Optional[Any]:
        """Restore content from snapshot

        Args:
            snapshot_id: Snapshot ID to restore

        Returns:
            Restored content or None
        """
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return None

        # If full snapshot, just return content
        if snapshot.metadata.snapshot_type == SnapshotType.FULL:
            return snapshot.get_uncompressed_content()

        # If incremental, need to reconstruct from parent
        elif snapshot.metadata.snapshot_type == SnapshotType.INCREMENTAL:
            if not snapshot.metadata.parent_snapshot_id:
                logger.error(f"Incremental snapshot {snapshot_id} has no parent")
                return None

            # Restore parent first
            parent_content = self.restore_snapshot(snapshot.metadata.parent_snapshot_id)
            if parent_content is None:
                logger.error(f"Failed to restore parent snapshot")
                return None

            # Apply delta
            delta = snapshot.get_uncompressed_content()
            return self._apply_delta(parent_content, delta)

        return None

    def delete_snapshot(self, snapshot_id: str, permanent: bool = False) -> bool:
        """Delete snapshot

        Args:
            snapshot_id: Snapshot ID
            permanent: Permanently delete (vs mark as deleted)

        Returns:
            True if deleted successfully
        """
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return False

        if permanent:
            # Remove from storage
            del self.snapshots[snapshot_id]

            # Remove from indices
            if snapshot.metadata.document_id:
                doc_snapshots = self.document_index[snapshot.metadata.document_id]
                if snapshot_id in doc_snapshots:
                    doc_snapshots.remove(snapshot_id)

            for tag in snapshot.metadata.tags:
                self.tag_index[tag].discard(snapshot_id)

            # Update stats
            self.stats['deleted_snapshots'] += 1
            self.stats['active_snapshots'] -= 1
            self.stats['total_size_bytes'] -= snapshot.metadata.compressed_size

            logger.info(f"Permanently deleted snapshot {snapshot_id}")
        else:
            # Mark as deleted
            snapshot.metadata.status = SnapshotStatus.DELETED
            logger.info(f"Marked snapshot {snapshot_id} as deleted")

        return True

    def cleanup_expired(self) -> int:
        """Clean up expired snapshots

        Returns:
            Number of snapshots cleaned up
        """
        cleaned = 0
        to_delete = []

        for snapshot_id, snapshot in self.snapshots.items():
            if self._is_expired(snapshot.metadata):
                to_delete.append(snapshot_id)

        for snapshot_id in to_delete:
            if self.delete_snapshot(snapshot_id, permanent=True):
                cleaned += 1

        logger.info(f"Cleaned up {cleaned} expired snapshots")
        return cleaned

    def export_snapshot(self, snapshot_id: str, include_metadata: bool = True) -> Optional[Dict[str, Any]]:
        """Export snapshot to dictionary

        Args:
            snapshot_id: Snapshot ID
            include_metadata: Include metadata in export

        Returns:
            Exported snapshot dictionary
        """
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return None

        export_data = {}

        if include_metadata:
            export_data['metadata'] = {
                'snapshot_id': snapshot.metadata.snapshot_id,
                'snapshot_type': snapshot.metadata.snapshot_type.value,
                'created_at': snapshot.metadata.created_at,
                'expires_at': snapshot.metadata.expires_at,
                'document_id': snapshot.metadata.document_id,
                'document_type': snapshot.metadata.document_type,
                'version_id': snapshot.metadata.version_id,
                'version_number': snapshot.metadata.version_number,
                'compression_type': snapshot.metadata.compression_type.value,
                'parent_snapshot_id': snapshot.metadata.parent_snapshot_id,
                'status': snapshot.metadata.status.value,
                'tags': snapshot.metadata.tags,
                'labels': snapshot.metadata.labels,
                'notes': snapshot.metadata.notes,
            }

        export_data['content'] = snapshot.get_uncompressed_content()

        return export_data

    def import_snapshot(self, import_data: Dict[str, Any]) -> Optional[str]:
        """Import snapshot from dictionary

        Args:
            import_data: Snapshot data to import

        Returns:
            Imported snapshot ID or None
        """
        try:
            content = import_data['content']
            metadata_dict = import_data.get('metadata', {})

            # Create snapshot
            metadata = self.create_snapshot(
                content,
                snapshot_type=SnapshotType[metadata_dict.get('snapshot_type', 'FULL')],
                document_id=metadata_dict.get('document_id'),
                document_type=metadata_dict.get('document_type'),
                version_id=metadata_dict.get('version_id'),
                version_number=metadata_dict.get('version_number'),
                tags=metadata_dict.get('tags', []),
                labels=metadata_dict.get('labels', {}),
                notes=metadata_dict.get('notes')
            )

            logger.info(f"Imported snapshot {metadata.snapshot_id}")
            return metadata.snapshot_id

        except Exception as e:
            logger.error(f"Failed to import snapshot: {e}")
            return None

    def _compress_content(self, content: Any, compression_type: CompressionType) -> Tuple[Any, int]:
        """Compress content

        Args:
            content: Content to compress
            compression_type: Type of compression

        Returns:
            Tuple of (compressed_content, compressed_size)
        """
        if compression_type == CompressionType.NONE:
            return content, self._calculate_size(content)

        elif compression_type == CompressionType.GZIP:
            # Convert to JSON and gzip
            json_str = json.dumps(content, ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')
            compressed = gzip.compress(json_bytes, compresslevel=9)
            # Encode as base64 for storage
            encoded = base64.b64encode(compressed).decode('ascii')
            return encoded, len(compressed)

        else:
            # For delta/hybrid, just return as-is
            return content, self._calculate_size(content)

    def _calculate_size(self, content: Any) -> int:
        """Calculate size of content in bytes"""
        if isinstance(content, (dict, list)):
            json_str = json.dumps(content, ensure_ascii=False)
            return len(json_str.encode('utf-8'))
        elif isinstance(content, str):
            return len(content.encode('utf-8'))
        else:
            return len(str(content).encode('utf-8'))

    def _calculate_delta(self, old_content: Any, new_content: Any) -> Dict[str, Any]:
        """Calculate delta between two contents

        Args:
            old_content: Old content
            new_content: New content

        Returns:
            Delta dictionary
        """
        delta = {
            'type': 'delta',
            'changes': []
        }

        if isinstance(old_content, dict) and isinstance(new_content, dict):
            # Find differences
            all_keys = set(old_content.keys()) | set(new_content.keys())

            for key in all_keys:
                if key not in old_content:
                    delta['changes'].append({
                        'op': 'add',
                        'path': [key],
                        'value': new_content[key]
                    })
                elif key not in new_content:
                    delta['changes'].append({
                        'op': 'remove',
                        'path': [key]
                    })
                elif old_content[key] != new_content[key]:
                    delta['changes'].append({
                        'op': 'replace',
                        'path': [key],
                        'old_value': old_content[key],
                        'new_value': new_content[key]
                    })
        else:
            # Simple replacement
            delta['changes'].append({
                'op': 'replace',
                'path': [],
                'old_value': old_content,
                'new_value': new_content
            })

        return delta

    def _apply_delta(self, base_content: Any, delta: Dict[str, Any]) -> Any:
        """Apply delta to base content

        Args:
            base_content: Base content
            delta: Delta to apply

        Returns:
            Modified content
        """
        result = copy.deepcopy(base_content)

        for change in delta.get('changes', []):
            op = change['op']
            path = change['path']

            if op == 'add':
                if isinstance(result, dict) and len(path) == 1:
                    result[path[0]] = change['value']

            elif op == 'remove':
                if isinstance(result, dict) and len(path) == 1:
                    result.pop(path[0], None)

            elif op == 'replace':
                if len(path) == 0:
                    result = change['new_value']
                elif isinstance(result, dict) and len(path) == 1:
                    result[path[0]] = change['new_value']

        return result

    def _compare_dicts(
        self,
        dict_a: Dict[str, Any],
        dict_b: Dict[str, Any],
        comparison: SnapshotComparison,
        prefix: str
    ) -> None:
        """Recursively compare dictionaries"""
        all_keys = set(dict_a.keys()) | set(dict_b.keys())

        for key in all_keys:
            field_name = f"{prefix}.{key}" if prefix else key

            if key not in dict_a:
                comparison.add_difference(field_name, None, dict_b[key])
            elif key not in dict_b:
                comparison.add_difference(field_name, dict_a[key], None)
            elif dict_a[key] != dict_b[key]:
                # Check if both are dicts for deeper comparison
                if isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
                    self._compare_dicts(dict_a[key], dict_b[key], comparison, field_name)
                else:
                    comparison.add_difference(field_name, dict_a[key], dict_b[key])

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        result = {}
        for key, value in d.items():
            field_name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten_dict(value, field_name))
            else:
                result[field_name] = value
        return result

    def _is_expired(self, metadata: SnapshotMetadata) -> bool:
        """Check if snapshot is expired"""
        if not metadata.expires_at:
            return False

        try:
            expires = datetime.fromisoformat(metadata.expires_at)
            return datetime.now() > expires
        except Exception:
            return False

    def _generate_snapshot_id(self, snapshot_type: SnapshotType) -> str:
        """Generate unique snapshot ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        type_code = snapshot_type.value[:4]
        return f"SNAP_{type_code}_{timestamp}"

    def _update_snapshot_stats(self, metadata: SnapshotMetadata) -> None:
        """Update statistics"""
        self.stats['total_snapshots'] += 1
        self.stats['active_snapshots'] += 1
        self.stats['total_size_bytes'] += metadata.uncompressed_size
        self.stats['compressed_size_bytes'] += metadata.compressed_size
        self.stats['snapshot_types'][metadata.snapshot_type.value] += 1
        self.stats['compression_types'][metadata.compression_type.value] += 1

        # Update average compression ratio
        if self.stats['total_snapshots'] > 0:
            total_ratio = sum(
                s.metadata.compression_ratio
                for s in self.snapshots.values()
            )
            self.stats['average_compression_ratio'] = total_ratio / self.stats['total_snapshots']

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        stats = self.stats.copy()

        # Add storage efficiency
        if stats['total_size_bytes'] > 0:
            stats['storage_efficiency'] = 1.0 - (stats['compressed_size_bytes'] / stats['total_size_bytes'])
        else:
            stats['storage_efficiency'] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_snapshots': 0,
            'active_snapshots': 0,
            'total_size_bytes': 0,
            'compressed_size_bytes': 0,
            'snapshot_types': defaultdict(int),
            'compression_types': defaultdict(int),
            'expired_snapshots': 0,
            'deleted_snapshots': 0,
            'average_compression_ratio': 0.0,
        }
        logger.info("Statistics reset")


__all__ = [
    'SnapshotManager',
    'SnapshotData',
    'SnapshotMetadata',
    'SnapshotComparison',
    'SnapshotType',
    'CompressionType',
    'SnapshotStatus'
]
