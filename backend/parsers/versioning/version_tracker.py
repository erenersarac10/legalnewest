"""Version Tracker - Harvey/Legora CTO-Level Production-Grade
Tracks versions of Turkish legal documents with full comparison

Production Features:
- Version tracking and management
- Version comparison (structural and semantic)
- Version history with parent-child relationships
- Turkish legal version naming (Değişik, Mülga, Ek Madde)
- Version metadata (publication, effectivity, amendments)
- Diff generation and visualization
- Version graph navigation
- SHA-256 checksums for integrity
- Statistics tracking
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import hashlib
import json
from datetime import datetime
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Types of versions"""
    ORIGINAL = "ORIGINAL"  # Original version
    AMENDED = "AMENDED"  # Değişik (amended)
    REPEALED = "REPEALED"  # Mülga (repealed)
    ADDITIONAL = "ADDITIONAL"  # Ek (additional articles)
    TEMPORARY = "TEMPORARY"  # Geçici (temporary provisions)
    CONSOLIDATED = "CONSOLIDATED"  # Consolidated version


class ChangeType(Enum):
    """Types of changes between versions"""
    ADDED = "ADDED"  # Content added
    REMOVED = "REMOVED"  # Content removed
    MODIFIED = "MODIFIED"  # Content modified
    MOVED = "MOVED"  # Content moved
    RENUMBERED = "RENUMBERED"  # Article renumbered
    NO_CHANGE = "NO_CHANGE"  # No change


class AmendmentType(Enum):
    """Turkish legal amendment types"""
    DEGISIK = "DEĞİŞİK"  # Changed/Amended
    MULGA = "MÜLGA"  # Repealed
    EK_MADDE = "EK MADDE"  # Additional article
    GECICI_MADDE = "GEÇİCİ MADDE"  # Temporary article
    IPTAL = "İPTAL"  # Cancelled
    YURURLUKTEN_KALDIRILDI = "YÜRÜRLÜKTEN KALDIRILDI"  # Removed from force


@dataclass
class VersionMetadata:
    """Metadata for a version"""
    version_id: str
    version_number: str  # e.g., "1.0", "2.3"
    version_type: VersionType

    # Dates
    created_at: str
    publication_date: Optional[str] = None
    effectivity_date: Optional[str] = None

    # Amendment info
    amending_law: Optional[str] = None  # e.g., "6698 sayılı Kanun"
    amendment_type: Optional[AmendmentType] = None

    # Relationships
    parent_version: Optional[str] = None  # Previous version ID
    child_versions: List[str] = field(default_factory=list)  # Next versions

    # Checksums
    checksum: Optional[str] = None  # SHA-256 of content

    # Statistics
    article_count: int = 0
    total_changes: int = 0

    # Additional metadata
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionChange:
    """Represents a change between versions"""
    change_type: ChangeType
    location: str  # e.g., "article_5", "article_10.paragraph_2"

    # Old and new values
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    # Turkish legal amendment info
    amendment_notation: Optional[str] = None  # e.g., "(Değişik: 01/01/2020-1234/1 md.)"

    # Change metadata
    change_score: float = 0.0  # 0-1, magnitude of change
    description: Optional[str] = None

    def summary(self) -> str:
        """Get human-readable summary"""
        if self.change_type == ChangeType.ADDED:
            return f"Added at {self.location}"
        elif self.change_type == ChangeType.REMOVED:
            return f"Removed from {self.location}"
        elif self.change_type == ChangeType.MODIFIED:
            return f"Modified at {self.location}"
        elif self.change_type == ChangeType.MOVED:
            return f"Moved from {self.old_value} to {self.new_value}"
        elif self.change_type == ChangeType.RENUMBERED:
            return f"Renumbered from {self.old_value} to {self.new_value}"
        else:
            return f"No change at {self.location}"


@dataclass
class VersionComparison:
    """Result of comparing two versions"""
    version_a: str  # Version A ID
    version_b: str  # Version B ID

    # Changes
    changes: List[VersionChange] = field(default_factory=list)

    # Change statistics
    additions: int = 0
    removals: int = 0
    modifications: int = 0
    moves: int = 0
    total_changes: int = 0

    # Similarity metrics
    similarity_score: float = 0.0  # 0-1, higher = more similar
    change_ratio: float = 0.0  # Proportion of content changed

    # Comparison metadata
    comparison_time: float = 0.0
    compared_at: Optional[str] = None

    def add_change(self, change: VersionChange) -> None:
        """Add a change to comparison"""
        self.changes.append(change)
        self.total_changes += 1

        if change.change_type == ChangeType.ADDED:
            self.additions += 1
        elif change.change_type == ChangeType.REMOVED:
            self.removals += 1
        elif change.change_type == ChangeType.MODIFIED:
            self.modifications += 1
        elif change.change_type == ChangeType.MOVED:
            self.moves += 1

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Version Comparison: {self.version_a} → {self.version_b}")
        lines.append(f"Total Changes: {self.total_changes}")
        lines.append(f"  Additions: {self.additions}")
        lines.append(f"  Removals: {self.removals}")
        lines.append(f"  Modifications: {self.modifications}")
        lines.append(f"  Moves: {self.moves}")
        lines.append(f"Similarity Score: {self.similarity_score:.3f}")
        lines.append(f"Change Ratio: {self.change_ratio:.3f}")
        lines.append(f"Comparison Time: {self.comparison_time:.3f}s")

        if self.changes:
            lines.append(f"\nTop Changes:")
            sorted_changes = sorted(
                self.changes,
                key=lambda c: c.change_score,
                reverse=True
            )
            for change in sorted_changes[:5]:
                lines.append(f"  - {change.summary()}")

        return '\n'.join(lines)


class VersionTracker:
    """Version Tracker for Turkish Legal Documents

    Tracks and manages versions of legal documents:
    - Version creation and storage
    - Version comparison (structural and content)
    - Version history management
    - Turkish legal amendment tracking
    - Version graph navigation
    - Checksum verification

    Features:
    - Parent-child version relationships
    - Multiple comparison methods
    - Turkish legal version naming
    - Amendment type detection
    - Diff generation
    - Statistics tracking
    """

    def __init__(self):
        """Initialize Version Tracker"""
        # Version storage
        self.versions: Dict[str, Tuple[VersionMetadata, Any]] = {}  # version_id -> (metadata, content)

        # Version graph (for navigation)
        self.version_graph: Dict[str, Set[str]] = defaultdict(set)  # parent -> children

        # Statistics
        self.stats = {
            'total_versions': 0,
            'total_comparisons': 0,
            'total_changes_tracked': 0,
            'version_types': defaultdict(int),
            'amendment_types': defaultdict(int),
            'average_changes_per_version': 0.0,
        }

        logger.info("Initialized Version Tracker")

    def create_version(
        self,
        content: Any,
        version_number: str,
        version_type: VersionType = VersionType.ORIGINAL,
        parent_version: Optional[str] = None,
        **kwargs
    ) -> VersionMetadata:
        """Create a new version

        Args:
            content: Document content
            version_number: Version number (e.g., "1.0", "2.1")
            version_type: Type of version
            parent_version: Parent version ID
            **kwargs: Additional metadata
                - publication_date: Publication date
                - effectivity_date: Effectivity date
                - amending_law: Amending law reference
                - amendment_type: Type of amendment
                - notes: Version notes
                - tags: Version tags

        Returns:
            VersionMetadata for created version
        """
        start_time = time.time()

        # Generate version ID
        version_id = self._generate_version_id(version_number, version_type)

        # Calculate checksum
        checksum = self._calculate_checksum(content)

        # Count articles
        article_count = self._count_articles(content)

        # Create metadata
        metadata = VersionMetadata(
            version_id=version_id,
            version_number=version_number,
            version_type=version_type,
            created_at=datetime.now().isoformat(),
            publication_date=kwargs.get('publication_date'),
            effectivity_date=kwargs.get('effectivity_date'),
            amending_law=kwargs.get('amending_law'),
            amendment_type=kwargs.get('amendment_type'),
            parent_version=parent_version,
            checksum=checksum,
            article_count=article_count,
            notes=kwargs.get('notes'),
            tags=kwargs.get('tags', [])
        )

        # Store version
        self.versions[version_id] = (metadata, content)

        # Update version graph
        if parent_version:
            self.version_graph[parent_version].add(version_id)
            if parent_version in self.versions:
                parent_metadata, _ = self.versions[parent_version]
                parent_metadata.child_versions.append(version_id)

        # Update statistics
        self._update_version_stats(metadata)

        logger.info(f"Created version {version_id} (type: {version_type.value})")
        return metadata

    def get_version(self, version_id: str) -> Optional[Tuple[VersionMetadata, Any]]:
        """Get version by ID

        Args:
            version_id: Version identifier

        Returns:
            Tuple of (metadata, content) or None
        """
        return self.versions.get(version_id)

    def compare_versions(
        self,
        version_a_id: str,
        version_b_id: str,
        **kwargs
    ) -> VersionComparison:
        """Compare two versions

        Args:
            version_a_id: First version ID
            version_b_id: Second version ID
            **kwargs: Options
                - detailed: Include detailed diff (default: True)
                - semantic: Use semantic comparison (default: True)

        Returns:
            VersionComparison result
        """
        start_time = time.time()

        # Get versions
        version_a = self.get_version(version_a_id)
        version_b = self.get_version(version_b_id)

        if not version_a or not version_b:
            logger.error(f"Version not found: {version_a_id} or {version_b_id}")
            return VersionComparison(
                version_a=version_a_id,
                version_b=version_b_id,
                comparison_time=time.time() - start_time
            )

        metadata_a, content_a = version_a
        metadata_b, content_b = version_b

        logger.info(f"Comparing versions: {version_a_id} → {version_b_id}")

        # Create comparison result
        comparison = VersionComparison(
            version_a=version_a_id,
            version_b=version_b_id,
            compared_at=datetime.now().isoformat()
        )

        # Perform comparison
        detailed = kwargs.get('detailed', True)
        semantic = kwargs.get('semantic', True)

        if semantic and isinstance(content_a, dict) and isinstance(content_b, dict):
            self._compare_semantic(content_a, content_b, comparison)
        else:
            self._compare_textual(content_a, content_b, comparison)

        # Calculate similarity
        comparison.similarity_score = self._calculate_similarity(content_a, content_b)

        # Calculate change ratio
        if metadata_a.article_count > 0:
            comparison.change_ratio = comparison.total_changes / metadata_a.article_count

        # Finalize
        comparison.comparison_time = time.time() - start_time

        # Update statistics
        self.stats['total_comparisons'] += 1
        self.stats['total_changes_tracked'] += comparison.total_changes

        logger.info(f"Comparison complete: {comparison.total_changes} changes found")
        return comparison

    def get_version_history(
        self,
        version_id: str,
        include_ancestors: bool = True,
        include_descendants: bool = False
    ) -> List[VersionMetadata]:
        """Get version history

        Args:
            version_id: Starting version ID
            include_ancestors: Include parent versions
            include_descendants: Include child versions

        Returns:
            List of version metadata in chronological order
        """
        history = []

        # Get ancestors
        if include_ancestors:
            current = version_id
            while current:
                version_data = self.get_version(current)
                if version_data:
                    metadata, _ = version_data
                    history.insert(0, metadata)
                    current = metadata.parent_version
                else:
                    break

        # Get descendants
        if include_descendants:
            self._get_descendants(version_id, history)

        # If not including ancestors, add current version
        if not include_ancestors:
            version_data = self.get_version(version_id)
            if version_data:
                metadata, _ = version_data
                history.append(metadata)

        return history

    def _get_descendants(self, version_id: str, history: List[VersionMetadata]) -> None:
        """Recursively get descendant versions"""
        children = self.version_graph.get(version_id, set())
        for child_id in sorted(children):
            version_data = self.get_version(child_id)
            if version_data:
                metadata, _ = version_data
                history.append(metadata)
                self._get_descendants(child_id, history)

    def detect_amendment_type(self, content: Any) -> Optional[AmendmentType]:
        """Detect Turkish legal amendment type from content

        Args:
            content: Document content

        Returns:
            Detected amendment type or None
        """
        if not isinstance(content, dict):
            return None

        # Check for amendment markers in text
        text = self._extract_text(content)
        text_lower = text.lower()

        if 'değişik' in text_lower:
            return AmendmentType.DEGISIK
        elif 'mülga' in text_lower:
            return AmendmentType.MULGA
        elif 'ek madde' in text_lower:
            return AmendmentType.EK_MADDE
        elif 'geçici madde' in text_lower:
            return AmendmentType.GECICI_MADDE
        elif 'iptal' in text_lower:
            return AmendmentType.IPTAL
        elif 'yürürlükten kaldırıldı' in text_lower or 'yürürlükten kaldırılmıştır' in text_lower:
            return AmendmentType.YURURLUKTEN_KALDIRILDI

        return None

    def verify_checksum(self, version_id: str) -> bool:
        """Verify version checksum

        Args:
            version_id: Version ID to verify

        Returns:
            True if checksum matches, False otherwise
        """
        version_data = self.get_version(version_id)
        if not version_data:
            return False

        metadata, content = version_data
        current_checksum = self._calculate_checksum(content)

        return current_checksum == metadata.checksum

    def _compare_semantic(
        self,
        content_a: Dict[str, Any],
        content_b: Dict[str, Any],
        comparison: VersionComparison
    ) -> None:
        """Semantic comparison of structured content"""

        # Compare articles
        if 'articles' in content_a and 'articles' in content_b:
            articles_a = content_a['articles']
            articles_b = content_b['articles']

            if isinstance(articles_a, list) and isinstance(articles_b, list):
                self._compare_articles(articles_a, articles_b, comparison)

        # Compare metadata
        if 'metadata' in content_a and 'metadata' in content_b:
            self._compare_metadata(content_a['metadata'], content_b['metadata'], comparison)

        # Compare top-level fields
        all_fields = set(content_a.keys()) | set(content_b.keys())
        for field in all_fields:
            if field in ['articles', 'metadata']:
                continue  # Already compared

            if field in content_a and field not in content_b:
                comparison.add_change(VersionChange(
                    change_type=ChangeType.REMOVED,
                    location=field,
                    old_value=content_a[field],
                    change_score=0.5
                ))
            elif field not in content_a and field in content_b:
                comparison.add_change(VersionChange(
                    change_type=ChangeType.ADDED,
                    location=field,
                    new_value=content_b[field],
                    change_score=0.5
                ))
            elif content_a[field] != content_b[field]:
                comparison.add_change(VersionChange(
                    change_type=ChangeType.MODIFIED,
                    location=field,
                    old_value=content_a[field],
                    new_value=content_b[field],
                    change_score=0.3
                ))

    def _compare_articles(
        self,
        articles_a: List[Dict[str, Any]],
        articles_b: List[Dict[str, Any]],
        comparison: VersionComparison
    ) -> None:
        """Compare article lists"""

        # Build article lookup by number
        articles_a_map = {}
        for article in articles_a:
            if isinstance(article, dict):
                num = article.get('number', article.get('article_number'))
                if num:
                    articles_a_map[str(num)] = article

        articles_b_map = {}
        for article in articles_b:
            if isinstance(article, dict):
                num = article.get('number', article.get('article_number'))
                if num:
                    articles_b_map[str(num)] = article

        # Find changes
        all_numbers = set(articles_a_map.keys()) | set(articles_b_map.keys())

        for num in sorted(all_numbers, key=lambda x: int(x) if x.isdigit() else 999):
            location = f"article_{num}"

            if num in articles_a_map and num not in articles_b_map:
                # Removed
                comparison.add_change(VersionChange(
                    change_type=ChangeType.REMOVED,
                    location=location,
                    old_value=articles_a_map[num].get('content'),
                    change_score=0.8
                ))
            elif num not in articles_a_map and num in articles_b_map:
                # Added
                comparison.add_change(VersionChange(
                    change_type=ChangeType.ADDED,
                    location=location,
                    new_value=articles_b_map[num].get('content'),
                    change_score=0.8
                ))
            else:
                # Compare content
                article_a = articles_a_map[num]
                article_b = articles_b_map[num]

                content_a = article_a.get('content', '')
                content_b = article_b.get('content', '')

                if content_a != content_b:
                    # Calculate change magnitude
                    if content_a and content_b:
                        similarity = difflib.SequenceMatcher(None, content_a, content_b).ratio()
                        change_score = 1.0 - similarity
                    else:
                        change_score = 1.0

                    # Check for Turkish amendment notation
                    amendment_notation = self._extract_amendment_notation(content_b)

                    comparison.add_change(VersionChange(
                        change_type=ChangeType.MODIFIED,
                        location=location,
                        old_value=content_a,
                        new_value=content_b,
                        amendment_notation=amendment_notation,
                        change_score=change_score
                    ))

    def _compare_metadata(
        self,
        metadata_a: Dict[str, Any],
        metadata_b: Dict[str, Any],
        comparison: VersionComparison
    ) -> None:
        """Compare metadata"""
        all_fields = set(metadata_a.keys()) | set(metadata_b.keys())

        for field in all_fields:
            location = f"metadata.{field}"

            if field in metadata_a and field not in metadata_b:
                comparison.add_change(VersionChange(
                    change_type=ChangeType.REMOVED,
                    location=location,
                    old_value=metadata_a[field],
                    change_score=0.2
                ))
            elif field not in metadata_a and field in metadata_b:
                comparison.add_change(VersionChange(
                    change_type=ChangeType.ADDED,
                    location=location,
                    new_value=metadata_b[field],
                    change_score=0.2
                ))
            elif metadata_a[field] != metadata_b[field]:
                comparison.add_change(VersionChange(
                    change_type=ChangeType.MODIFIED,
                    location=location,
                    old_value=metadata_a[field],
                    new_value=metadata_b[field],
                    change_score=0.2
                ))

    def _compare_textual(
        self,
        content_a: Any,
        content_b: Any,
        comparison: VersionComparison
    ) -> None:
        """Textual comparison using diff"""
        text_a = self._extract_text(content_a)
        text_b = self._extract_text(content_b)

        # Use difflib to find differences
        diff = difflib.unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            lineterm=''
        )

        line_num = 0
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                comparison.add_change(VersionChange(
                    change_type=ChangeType.ADDED,
                    location=f"line_{line_num}",
                    new_value=line[1:],
                    change_score=0.5
                ))
            elif line.startswith('-') and not line.startswith('---'):
                comparison.add_change(VersionChange(
                    change_type=ChangeType.REMOVED,
                    location=f"line_{line_num}",
                    old_value=line[1:],
                    change_score=0.5
                ))
            line_num += 1

    def _extract_amendment_notation(self, text: str) -> Optional[str]:
        """Extract Turkish legal amendment notation

        e.g., "(Değişik: 01/01/2020-1234/1 md.)"
        """
        import re

        patterns = [
            r'\(Değişik:.*?\)',
            r'\(Mülga:.*?\)',
            r'\(Ek:.*?\)',
            r'\(İptal:.*?\)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _calculate_similarity(self, content_a: Any, content_b: Any) -> float:
        """Calculate similarity between two versions"""
        text_a = self._extract_text(content_a)
        text_b = self._extract_text(content_b)

        if not text_a or not text_b:
            return 0.0

        return difflib.SequenceMatcher(None, text_a, text_b).ratio()

    def _calculate_checksum(self, content: Any) -> str:
        """Calculate SHA-256 checksum of content"""
        # Convert content to JSON string for consistent hashing
        if isinstance(content, (dict, list)):
            content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        else:
            content_str = str(content)

        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def _generate_version_id(self, version_number: str, version_type: VersionType) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        type_code = version_type.value[:3]
        return f"{type_code}_{version_number}_{timestamp}"

    def _count_articles(self, content: Any) -> int:
        """Count articles in content"""
        if isinstance(content, dict) and 'articles' in content:
            articles = content['articles']
            if isinstance(articles, list):
                return len(articles)
        return 0

    def _extract_text(self, content: Any) -> str:
        """Extract text from content"""
        text_parts = []

        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            for field in ['content', 'text', 'title', 'decision_text']:
                if field in content and isinstance(content[field], str):
                    text_parts.append(content[field])

            if 'articles' in content and isinstance(content['articles'], list):
                for article in content['articles']:
                    if isinstance(article, dict):
                        if 'content' in article:
                            text_parts.append(str(article['content']))

        return '\n'.join(text_parts)

    def _update_version_stats(self, metadata: VersionMetadata) -> None:
        """Update statistics"""
        self.stats['total_versions'] += 1
        self.stats['version_types'][metadata.version_type.value] += 1

        if metadata.amendment_type:
            self.stats['amendment_types'][metadata.amendment_type.value] += 1

        # Update rolling average
        if self.stats['total_versions'] > 0:
            total_changes = sum(
                m.total_changes for m, _ in self.versions.values()
            )
            self.stats['average_changes_per_version'] = total_changes / self.stats['total_versions']

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_versions': 0,
            'total_comparisons': 0,
            'total_changes_tracked': 0,
            'version_types': defaultdict(int),
            'amendment_types': defaultdict(int),
            'average_changes_per_version': 0.0,
        }
        logger.info("Statistics reset")


__all__ = [
    'VersionTracker',
    'VersionMetadata',
    'VersionComparison',
    'VersionChange',
    'VersionType',
    'ChangeType',
    'AmendmentType'
]
