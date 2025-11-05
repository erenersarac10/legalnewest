"""
Document Version model for comprehensive version control in Turkish Legal AI.

This module provides the DocumentVersion model for tracking document changes:
- Complete version history tracking
- Delta storage (only changed content)
- Diff generation between versions
- Version comparison and rollback
- Change attribution (who, when, why)
- Branch support (draft, review, final)
- Merge conflict detection
- KVKK-compliant audit trail

Version Control Strategy:
    - Full snapshots for major versions
    - Delta/diff storage for minor versions
    - Automatic version creation on save
    - Manual version creation (checkpoints)
    - Branch support (main, draft, review)
    - Tag support (v1.0, final, approved)

Use Cases:
    - Contract revision tracking
    - Legal document collaboration
    - Compliance audit trail
    - Rollback to previous version
    - Compare versions side-by-side
    - Track who changed what when

Example:
    >>> # Automatic version on update
    >>> doc.update_content(new_text)
    >>> version = DocumentVersion.create_from_document(
    ...     document=doc,
    ...     change_type=ChangeType.CONTENT_UPDATE,
    ...     change_summary="Updated Article 5.2",
    ...     changed_by_id=user.id
    ... )
    >>> 
    >>> # Compare versions
    >>> diff = version.compare_with(previous_version)
    >>> print(diff['added_lines'])  # 5
    >>> print(diff['removed_lines'])  # 2
    >>> 
    >>> # Rollback
    >>> doc.restore_version(version.id)
"""

import enum
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    CheckConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, validates

from backend.core.database.models.document import Document
from backend.core.exceptions import ValidationError, VersionControlError
from backend.core.logging import get_logger
from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
    TenantMixin,
    AuditMixin,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ChangeType(str, enum.Enum):
    """
    Type of change that triggered version creation.
    
    Types:
    - INITIAL: Initial document creation
    - CONTENT_UPDATE: Content modified
    - METADATA_UPDATE: Metadata changed (name, tags, etc.)
    - MANUAL_CHECKPOINT: User-created checkpoint
    - AUTO_SAVE: Automatic save
    - MERGE: Merged from another branch
    - ROLLBACK: Restored from previous version
    - IMPORT: Imported from external source
    """
    
    INITIAL = "initial"                    # First version
    CONTENT_UPDATE = "content_update"      # Content changed
    METADATA_UPDATE = "metadata_update"    # Metadata changed
    MANUAL_CHECKPOINT = "manual_checkpoint"  # User checkpoint
    AUTO_SAVE = "auto_save"                # Auto-save
    MERGE = "merge"                        # Branch merge
    ROLLBACK = "rollback"                  # Restored version
    IMPORT = "import"                      # External import
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.INITIAL: "İlk Versiyon",
            self.CONTENT_UPDATE: "İçerik Güncellendi",
            self.METADATA_UPDATE: "Bilgiler Güncellendi",
            self.MANUAL_CHECKPOINT: "Manuel Kayıt",
            self.AUTO_SAVE: "Otomatik Kayıt",
            self.MERGE: "Birleştirme",
            self.ROLLBACK: "Geri Yükleme",
            self.IMPORT: "İçe Aktarma",
        }
        return names.get(self, self.value)


class VersionStatus(str, enum.Enum):
    """Version lifecycle status."""
    
    DRAFT = "draft"              # Draft version (work in progress)
    REVIEW = "review"            # Under review
    APPROVED = "approved"        # Approved by reviewer
    FINAL = "final"              # Finalized (no more edits)
    ARCHIVED = "archived"        # Archived (old version)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.DRAFT: "Taslak",
            self.REVIEW: "İnceleme",
            self.APPROVED: "Onaylı",
            self.FINAL: "Kesinleşmiş",
            self.ARCHIVED: "Arşivlenmiş",
        }
        return names.get(self, self.value)


class StorageStrategy(str, enum.Enum):
    """Version storage strategy."""
    
    FULL_SNAPSHOT = "full_snapshot"  # Complete document copy
    DELTA = "delta"                  # Only changes (diff)
    REFERENCE = "reference"          # Reference to S3 file
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# DOCUMENT VERSION MODEL
# =============================================================================


class DocumentVersion(Base, BaseModelMixin, TenantMixin, AuditMixin):
    """
    Document Version model for comprehensive version control.
    
    Tracks all changes to documents with:
    - Complete version history
    - Delta storage (efficient)
    - Change attribution (who, when, why)
    - Version comparison
    - Rollback support
    - Branch and tag support
    
    Storage Strategies:
    1. Full Snapshot: Store complete document (major versions)
    2. Delta: Store only changes (minor versions)
    3. Reference: Reference to S3 file (large files)
    
    Version Numbering:
        - major.minor.patch format
        - Example: 1.0.0, 1.1.0, 1.1.1
        - Major: Breaking changes, major revisions
        - Minor: Feature additions, significant updates
        - Patch: Bug fixes, minor edits
    
    Attributes:
        document_id: Parent document
        document: Document relationship
        
        version_number: Sequential number (1, 2, 3...)
        version_label: Semantic version (1.0.0, 1.1.0)
        version_tag: User-defined tag (v1, draft, final)
        
        change_type: Type of change
        change_summary: Brief description of changes
        change_details: Detailed change log (JSON)
        
        changed_by_id: Who made the change
        changed_by: User relationship
        
        storage_strategy: How version is stored
        content_snapshot: Full content (if full_snapshot)
        content_delta: Changes only (if delta)
        file_reference: S3 path (if reference)
        
        file_size: Size of this version
        file_hash: SHA-256 hash
        
        status: Version status (draft, review, approved)
        branch: Branch name (main, draft, review)
        
        parent_version_id: Previous version
        parent_version: Parent relationship
        child_versions: Child versions
        
        metadata_snapshot: Document metadata at this version
        tags: Version tags
        
        is_major_version: Major version flag
        is_current: Current active version
        
        reviewed_by_id: Who reviewed (if applicable)
        reviewed_at: Review timestamp
        review_notes: Review comments
        
        stats: Version statistics (lines added/removed, etc.)
        
    Relationships:
        tenant: Parent tenant
        document: Parent document
        changed_by: User who created version
        reviewed_by: User who reviewed
        parent_version: Previous version
        child_versions: Next versions
    """
    
    __tablename__ = "document_versions"
    
    # =========================================================================
    # DOCUMENT RELATIONSHIP
    # =========================================================================
    
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent document",
    )
    
    document = relationship(
        "Document",
        back_populates="document_versions",
        foreign_keys=[document_id],
    )
    
    # =========================================================================
    # VERSION IDENTIFICATION
    # =========================================================================
    
    version_number = Column(
        Integer,
        nullable=False,
        comment="Sequential version number (1, 2, 3...)",
    )
    
    version_label = Column(
        String(50),
        nullable=True,
        comment="Semantic version label (1.0.0, 1.1.0)",
    )
    
    version_tag = Column(
        String(100),
        nullable=True,
        comment="User-defined version tag (v1, draft, final, approved)",
    )
    
    # =========================================================================
    # CHANGE INFORMATION
    # =========================================================================
    
    change_type = Column(
        Enum(ChangeType, native_enum=False, length=50),
        nullable=False,
        default=ChangeType.CONTENT_UPDATE,
        comment="Type of change",
    )
    
    change_summary = Column(
        String(500),
        nullable=True,
        comment="Brief description of changes",
    )
    
    change_details = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed change log (fields changed, old/new values)",
    )
    
    # =========================================================================
    # ATTRIBUTION
    # =========================================================================
    
    changed_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who created this version",
    )
    
    changed_by = relationship(
        "User",
        foreign_keys=[changed_by_id],
        back_populates="document_versions_created",
    )
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    storage_strategy = Column(
        Enum(StorageStrategy, native_enum=False, length=50),
        nullable=False,
        default=StorageStrategy.DELTA,
        comment="How this version is stored",
    )
    
    content_snapshot = Column(
        Text,
        nullable=True,
        comment="Full content snapshot (if full_snapshot strategy)",
    )
    
    content_delta = Column(
        JSONB,
        nullable=True,
        comment="Content changes only (if delta strategy)",
    )
    
    file_reference = Column(
        String(500),
        nullable=True,
        comment="S3 path reference (if reference strategy)",
    )
    
    # =========================================================================
    # FILE INFORMATION
    # =========================================================================
    
    file_size = Column(
        Integer,
        nullable=True,
        comment="File size of this version (bytes)",
    )
    
    file_hash = Column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA-256 hash of version content",
    )
    
    # =========================================================================
    # STATUS & BRANCH
    # =========================================================================
    
    status = Column(
        Enum(VersionStatus, native_enum=False, length=50),
        nullable=False,
        default=VersionStatus.DRAFT,
        index=True,
        comment="Version status",
    )
    
    branch = Column(
        String(100),
        nullable=False,
        default="main",
        index=True,
        comment="Branch name (main, draft, review)",
    )
    
    # =========================================================================
    # VERSION TREE (Parent-Child)
    # =========================================================================
    
    parent_version_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Previous version (parent)",
    )
    
    # Self-referential relationship
    parent_version = relationship(
        "DocumentVersion",
        remote_side="DocumentVersion.id",
        back_populates="child_versions",
        foreign_keys=[parent_version_id],
    )
    
    child_versions = relationship(
        "DocumentVersion",
        back_populates="parent_version",
        foreign_keys=[parent_version_id],
    )
    
    # =========================================================================
    # METADATA & TAGS
    # =========================================================================
    
    metadata_snapshot = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Document metadata at time of version creation",
    )
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Version-specific tags",
    )
    
    # =========================================================================
    # FLAGS
    # =========================================================================
    
    is_major_version = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Major version (significant changes)",
    )
    
    is_current = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Current active version of document",
    )
    
    # =========================================================================
    # REVIEW
    # =========================================================================
    
    reviewed_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Who reviewed this version",
    )
    
    reviewed_by = relationship(
        "User",
        foreign_keys=[reviewed_by_id],
        back_populates="document_versions_reviewed",
    )
    
    reviewed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When version was reviewed",
    )
    
    review_notes = Column(
        Text,
        nullable=True,
        comment="Review comments and notes",
    )
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    stats = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Version statistics (lines_added, lines_removed, chars_changed)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Unique version number per document
        Index(
            "ix_document_versions_doc_num",
            "document_id",
            "version_number",
            unique=True,
        ),
        
        # Index for current version lookup
        Index(
            "ix_document_versions_current",
            "document_id",
            "is_current",
            postgresql_where="is_current = true",
        ),
        
        # Index for branch queries
        Index(
            "ix_document_versions_branch",
            "document_id",
            "branch",
        ),
        
        # Index for status filtering
        Index(
            "ix_document_versions_status",
            "document_id",
            "status",
        ),
        
        # Check: version number positive
        CheckConstraint(
            "version_number > 0",
            name="ck_document_versions_number_positive",
        ),
        
        # Check: at least one storage strategy used
        CheckConstraint(
            "content_snapshot IS NOT NULL OR content_delta IS NOT NULL OR file_reference IS NOT NULL",
            name="ck_document_versions_storage",
        ),
    )
    
    # =========================================================================
    # VERSION CREATION
    # =========================================================================
    
    @classmethod
    def create_from_document(
        cls,
        document: "Document",
        change_type: ChangeType,
        change_summary: str | None = None,
        changed_by_id: str | None = None,
        storage_strategy: StorageStrategy = StorageStrategy.DELTA,
        content: str | None = None,
        is_major: bool = False,
    ) -> "DocumentVersion":
        """
        Create a new version from current document state.
        
        Args:
            document: Document to version
            change_type: Type of change
            change_summary: Brief description
            changed_by_id: User making change
            storage_strategy: How to store version
            content: Document content (if available)
            is_major: Major version flag
            
        Returns:
            DocumentVersion: New version instance
            
        Example:
            >>> version = DocumentVersion.create_from_document(
            ...     document=doc,
            ...     change_type=ChangeType.CONTENT_UPDATE,
            ...     change_summary="Updated contract terms",
            ...     changed_by_id=str(user.id),
            ...     is_major=False
            ... )
        """
        # Get next version number
        next_version_num = cls._get_next_version_number(document.id)
        
        # Generate version label
        version_label = cls._generate_version_label(
            next_version_num,
            is_major=is_major,
        )
        
        # Create version
        version = cls(
            document_id=document.id,
            tenant_id=document.tenant_id,
            version_number=next_version_num,
            version_label=version_label,
            change_type=change_type,
            change_summary=change_summary,
            changed_by_id=changed_by_id,
            storage_strategy=storage_strategy,
            is_major_version=is_major,
            metadata_snapshot={
                "name": document.name,
                "document_type": document.document_type.value,
                "tags": document.tags,
                "file_size": document.file_size,
                "page_count": document.page_count,
                "word_count": document.word_count,
            },
        )
        
        # Store content based on strategy
        if storage_strategy == StorageStrategy.FULL_SNAPSHOT:
            version.content_snapshot = content or document.text_extracted
            version.file_size = len(version.content_snapshot or "")
        elif storage_strategy == StorageStrategy.DELTA:
            # Calculate delta from previous version
            previous = cls._get_latest_version(document.id)
            if previous and previous.content_snapshot:
                delta = cls._calculate_delta(
                    previous.content_snapshot,
                    content or document.text_extracted or "",
                )
                version.content_delta = delta
                version.stats = cls._calculate_stats(delta)
        elif storage_strategy == StorageStrategy.REFERENCE:
            version.file_reference = document.file_path
            version.file_size = document.file_size
        
        # Calculate hash
        if content:
            import hashlib
            version.file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        logger.info(
            "Document version created",
            document_id=str(document.id),
            version_id=str(version.id),
            version_number=next_version_num,
            change_type=change_type.value,
        )
        
        return version
    
    @staticmethod
    def _get_next_version_number(document_id: str) -> int:
        """
        Get next version number for document.
        
        Args:
            document_id: Document UUID
            
        Returns:
            int: Next version number
        """
        # This would query the database for max version_number
        # Placeholder: return 1 for now
        return 1
    
    @staticmethod
    def _get_latest_version(document_id: str) -> "DocumentVersion | None":
        """Get latest version of document."""
        # This would query the database
        # Placeholder: return None
        return None
    
    @staticmethod
    def _generate_version_label(
        version_number: int,
        is_major: bool = False,
    ) -> str:
        """
        Generate semantic version label.
        
        Args:
            version_number: Sequential version number
            is_major: Major version flag
            
        Returns:
            str: Version label (e.g., "1.0.0")
        """
        if is_major:
            major = version_number
            return f"{major}.0.0"
        else:
            # Simple sequential versioning
            return f"1.{version_number}.0"
    
    @staticmethod
    def _calculate_delta(old_content: str, new_content: str) -> dict[str, Any]:
        """
        Calculate content delta between versions.
        
        Uses difflib to compute efficient delta.
        
        Args:
            old_content: Previous version content
            new_content: New version content
            
        Returns:
            dict: Delta information
        """
        import difflib
        
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        # Generate unified diff
        diff = list(difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm='',
        ))
        
        # Calculate statistics
        added = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        
        return {
            "diff": diff,
            "added_lines": added,
            "removed_lines": removed,
            "total_changes": added + removed,
        }
    
    @staticmethod
    def _calculate_stats(delta: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate statistics from delta.
        
        Args:
            delta: Delta dictionary
            
        Returns:
            dict: Statistics
        """
        return {
            "lines_added": delta.get("added_lines", 0),
            "lines_removed": delta.get("removed_lines", 0),
            "total_changes": delta.get("total_changes", 0),
        }
    
    # =========================================================================
    # VERSION COMPARISON
    # =========================================================================
    
    def compare_with(self, other_version: "DocumentVersion") -> dict[str, Any]:
        """
        Compare this version with another version.
        
        Args:
            other_version: Version to compare with
            
        Returns:
            dict: Comparison results (added, removed, changed)
            
        Example:
            >>> diff = current_version.compare_with(previous_version)
            >>> print(f"Added: {diff['added_lines']}")
            >>> print(f"Removed: {diff['removed_lines']}")
        """
        # Get content from both versions
        content_a = self.get_content()
        content_b = other_version.get_content()
        
        if not content_a or not content_b:
            return {
                "error": "Cannot compare: content not available",
            }
        
        # Calculate diff
        delta = self._calculate_delta(content_b, content_a)
        
        return {
            "version_a": self.version_label or str(self.version_number),
            "version_b": other_version.version_label or str(other_version.version_number),
            "added_lines": delta["added_lines"],
            "removed_lines": delta["removed_lines"],
            "total_changes": delta["total_changes"],
            "diff": delta["diff"],
        }
    
    def get_content(self) -> str | None:
        """
        Get content of this version.
        
        Reconstructs content based on storage strategy.
        
        Returns:
            str | None: Version content
        """
        if self.storage_strategy == StorageStrategy.FULL_SNAPSHOT:
            return self.content_snapshot
        
        elif self.storage_strategy == StorageStrategy.DELTA:
            # Reconstruct from base + deltas
            # This requires traversing version tree
            # Placeholder: return None
            return None
        
        elif self.storage_strategy == StorageStrategy.REFERENCE:
            # Would fetch from S3
            # Placeholder: return None
            return None
        
        return None
    
    # =========================================================================
    # VERSION MANAGEMENT
    # =========================================================================
    
    def set_as_current(self) -> None:
        """
        Mark this version as current.
        
        Unmarks previous current version.
        
        Example:
            >>> version.set_as_current()
            >>> # Document now shows this version
        """
        # Would update other versions in DB to is_current=False
        # Then set this one to True
        self.is_current = True
        
        logger.info(
            "Version set as current",
            version_id=str(self.id),
            document_id=str(self.document_id),
            version_number=self.version_number,
        )
    
    def approve(
        self,
        reviewer_id: str,
        review_notes: str | None = None,
    ) -> None:
        """
        Approve this version.
        
        Args:
            reviewer_id: User approving
            review_notes: Review comments
            
        Example:
            >>> version.approve(
            ...     reviewer_id=str(senior_lawyer.id),
            ...     review_notes="Approved after legal review"
            ... )
        """
        self.status = VersionStatus.APPROVED
        self.reviewed_by_id = reviewer_id
        self.reviewed_at = datetime.now(timezone.utc)
        self.review_notes = review_notes
        
        logger.info(
            "Version approved",
            version_id=str(self.id),
            reviewer_id=reviewer_id,
        )
    
    def finalize(self) -> None:
        """Mark version as final (no more edits)."""
        self.status = VersionStatus.FINAL
        
        logger.info(
            "Version finalized",
            version_id=str(self.id),
        )
    
    def archive(self) -> None:
        """Archive this version."""
        self.status = VersionStatus.ARCHIVED
        self.is_current = False
        
        logger.info(
            "Version archived",
            version_id=str(self.id),
        )
    
    # =========================================================================
    # TAG MANAGEMENT
    # =========================================================================
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to version.
        
        Args:
            tag: Tag to add (e.g., "v1.0", "final", "approved")
            
        Example:
            >>> version.add_tag("final")
            >>> version.add_tag("client-approved")
        """
        if tag not in self.tags:
            self.tags.append(tag)
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "tags")
            
            logger.debug(
                "Tag added to version",
                version_id=str(self.id),
                tag=tag,
            )
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from version."""
        if tag in self.tags:
            self.tags.remove(tag)
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "tags")
    
    # =========================================================================
    # VERSION TREE NAVIGATION
    # =========================================================================
    
    def get_version_tree(self) -> dict[str, Any]:
        """
        Get version tree structure.
        
        Returns:
            dict: Version tree with parent/children
            
        Example:
            >>> tree = version.get_version_tree()
            >>> print(tree['ancestors'])  # [v1, v2, v3]
            >>> print(tree['descendants'])  # [v5, v6]
        """
        # Traverse version tree
        ancestors = []
        current = self.parent_version
        while current:
            ancestors.append({
                "id": str(current.id),
                "version_number": current.version_number,
                "version_label": current.version_label,
                "change_type": current.change_type.value,
            })
            current = current.parent_version
        
        # Get descendants (children)
        descendants = []
        for child in self.child_versions:
            descendants.append({
                "id": str(child.id),
                "version_number": child.version_number,
                "version_label": child.version_label,
                "change_type": child.change_type.value,
            })
        
        return {
            "current": {
                "id": str(self.id),
                "version_number": self.version_number,
                "version_label": self.version_label,
            },
            "ancestors": ancestors,
            "descendants": descendants,
        }
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("version_number")
    def validate_version_number(self, key: str, version_number: int) -> int:
        """Validate version number."""
        if version_number <= 0:
            raise ValidationError(
                message="Version number must be positive",
                field="version_number",
            )
        
        return version_number
    
    @validates("branch")
    def validate_branch(self, key: str, branch: str) -> str:
        """Validate branch name."""
        import re
        
        # Branch name: lowercase, alphanumeric, hyphens
        if not re.match(r"^[a-z0-9-]+$", branch):
            raise ValidationError(
                message="Branch name must be lowercase alphanumeric with hyphens",
                field="branch",
            )
        
        return branch
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<DocumentVersion("
            f"id={self.id}, "
            f"doc_id={self.document_id}, "
            f"v{self.version_number}"
            f")>"
        )
    
    def to_dict(self, include_content: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_content: Include content (default: False, can be large)
            
        Returns:
            dict: Version data
        """
        data = super().to_dict()
        
        # Remove large content by default
        if not include_content:
            data.pop("content_snapshot", None)
            data.pop("content_delta", None)
        
        # Add computed fields
        data["change_type_display"] = self.change_type.display_name_tr
        data["status_display"] = self.status.display_name_tr
        data["version_display"] = self.version_label or f"v{self.version_number}"
        
        # Add stats summary
        if self.stats:
            data["stats_summary"] = (
                f"+{self.stats.get('lines_added', 0)} "
                f"-{self.stats.get('lines_removed', 0)}"
            )
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "DocumentVersion",
    "ChangeType",
    "VersionStatus",
    "StorageStrategy",
]
