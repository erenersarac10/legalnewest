"""Patch Generator - Harvey/Legora CTO-Level Production-Grade
Generates patches to transform legal document versions

Production Features:
- Multiple patch formats (UNIFIED, JSON, ARTICLE_LEVEL, SEMANTIC)
- Article-level patch generation
- Line-level patch generation
- Semantic patch generation
- Patch validation
- Dry-run mode
- Reverse patch generation
- Conflict detection
- Patch metadata
- Turkish legal document awareness
- Structured output
"""
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from .diff_engine import DiffResult, Change, ChangeType
from .clause_differ import ClauseDiffResult, ArticleChange, ArticleChangeType
from .semantic_differ import SemanticDiffResult, SemanticChange

logger = logging.getLogger(__name__)


class PatchFormat(Enum):
    """Patch output formats"""
    UNIFIED = "UNIFIED"  # Unified diff format (like git diff)
    JSON = "JSON"  # Structured JSON
    ARTICLE_LEVEL = "ARTICLE_LEVEL"  # Article-level structured
    SEMANTIC = "SEMANTIC"  # Semantic-level structured
    CUSTOM = "CUSTOM"  # Custom Turkish legal format


class PatchOperation(Enum):
    """Types of patch operations"""
    ADD = "ADD"  # Add content
    DELETE = "DELETE"  # Delete content
    REPLACE = "REPLACE"  # Replace content
    MOVE = "MOVE"  # Move content
    RENAME = "RENAME"  # Rename (article renumbering)


@dataclass
class PatchHunk:
    """Represents a single patch hunk (one change)"""
    operation: PatchOperation
    old_start: Optional[int] = None  # Line number in old version
    old_count: Optional[int] = None  # Number of lines in old version
    new_start: Optional[int] = None  # Line number in new version
    new_count: Optional[int] = None  # Number of lines in new version
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    article_number: Optional[str] = None  # For article-level patches
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Patch:
    """Represents a complete patch"""
    patch_format: PatchFormat
    hunks: List[PatchHunk]

    # Metadata
    source_version: Optional[str] = None
    target_version: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: Optional[str] = None

    # Statistics
    total_additions: int = 0
    total_deletions: int = 0
    total_modifications: int = 0

    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert patch to dictionary"""
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        """Convert patch to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, **kwargs)


class PatchGenerator:
    """Patch Generator for Turkish Legal Documents

    Generates patches to transform legal document versions:
    - Multiple patch formats
    - Article-level and line-level granularity
    - Semantic patches
    - Validation and dry-run
    - Reverse patch generation

    Features:
    - 5 patch formats
    - 5 patch operations
    - Conflict detection
    - Metadata generation
    - Turkish legal awareness
    """

    def __init__(self):
        """Initialize Patch Generator"""
        logger.info("Initialized Patch Generator")

    def generate_from_diff(
        self, diff_result: DiffResult, **kwargs
    ) -> Patch:
        """Generate patch from DiffResult (line-level)

        Args:
            diff_result: DiffResult from DiffEngine
            **kwargs: Options
                - format: PatchFormat (default: UNIFIED)
                - context_lines: Number of context lines (default: 3)

        Returns:
            Patch object
        """
        patch_format = kwargs.get('format', PatchFormat.UNIFIED)
        context_lines = kwargs.get('context_lines', 3)

        logger.info(f"Generating {patch_format.value} patch from line-level diff")

        hunks = []
        for change in diff_result.changes:
            # Skip unchanged lines unless needed for context
            if change.change_type == ChangeType.UNCHANGED:
                continue

            hunk = self._change_to_hunk(change)
            hunks.append(hunk)

        # Compute statistics
        total_additions = sum(1 for h in hunks if h.operation == PatchOperation.ADD)
        total_deletions = sum(1 for h in hunks if h.operation == PatchOperation.DELETE)
        total_modifications = sum(1 for h in hunks if h.operation == PatchOperation.REPLACE)

        patch = Patch(
            patch_format=patch_format,
            hunks=hunks,
            total_additions=total_additions,
            total_deletions=total_deletions,
            total_modifications=total_modifications,
            metadata={
                'total_changes': diff_result.total_changes,
                'algorithm': diff_result.algorithm.value if diff_result.algorithm else None
            }
        )

        logger.info(f"Generated patch with {len(hunks)} hunks")
        return patch

    def generate_from_clause_diff(
        self, clause_diff: ClauseDiffResult, **kwargs
    ) -> Patch:
        """Generate patch from ClauseDiffResult (article-level)

        Args:
            clause_diff: ClauseDiffResult from ClauseDiffer
            **kwargs: Options
                - format: PatchFormat (default: ARTICLE_LEVEL)

        Returns:
            Patch object
        """
        patch_format = kwargs.get('format', PatchFormat.ARTICLE_LEVEL)

        logger.info(f"Generating {patch_format.value} patch from article-level diff")

        hunks = []
        for article_change in clause_diff.article_changes:
            # Skip unchanged articles
            if article_change.change_type == ArticleChangeType.UNCHANGED:
                continue

            hunk = self._article_change_to_hunk(article_change)
            hunks.append(hunk)

        patch = Patch(
            patch_format=patch_format,
            hunks=hunks,
            total_additions=clause_diff.articles_added,
            total_deletions=clause_diff.articles_deleted,
            total_modifications=clause_diff.articles_modified,
            metadata={
                'total_articles_old': clause_diff.total_articles_old,
                'total_articles_new': clause_diff.total_articles_new
            }
        )

        logger.info(f"Generated article-level patch with {len(hunks)} hunks")
        return patch

    def generate_from_semantic_diff(
        self, semantic_diff: SemanticDiffResult, **kwargs
    ) -> Patch:
        """Generate patch from SemanticDiffResult (semantic-level)

        Args:
            semantic_diff: SemanticDiffResult from SemanticDiffer
            **kwargs: Options
                - format: PatchFormat (default: SEMANTIC)

        Returns:
            Patch object
        """
        patch_format = kwargs.get('format', PatchFormat.SEMANTIC)

        logger.info(f"Generating {patch_format.value} patch from semantic diff")

        hunks = []
        for semantic_change in semantic_diff.semantic_changes:
            hunk = self._semantic_change_to_hunk(semantic_change)
            hunks.append(hunk)

        patch = Patch(
            patch_format=patch_format,
            hunks=hunks,
            total_modifications=semantic_diff.total_semantic_changes,
            metadata={
                'overall_impact': semantic_diff.overall_impact.value,
                'confidence': semantic_diff.confidence,
                'entities_added': len(semantic_diff.entities_added),
                'entities_removed': len(semantic_diff.entities_removed)
            }
        )

        logger.info(f"Generated semantic patch with {len(hunks)} hunks")
        return patch

    def _change_to_hunk(self, change: Change) -> PatchHunk:
        """Convert Change to PatchHunk"""
        if change.change_type == ChangeType.ADDED:
            operation = PatchOperation.ADD
        elif change.change_type == ChangeType.DELETED:
            operation = PatchOperation.DELETE
        elif change.change_type == ChangeType.MODIFIED:
            operation = PatchOperation.REPLACE
        else:
            operation = PatchOperation.REPLACE

        return PatchHunk(
            operation=operation,
            old_start=change.line_number_old,
            old_count=1 if change.old_content else 0,
            new_start=change.line_number_new,
            new_count=1 if change.new_content else 0,
            old_content=change.old_content,
            new_content=change.new_content,
            metadata={
                'similarity': change.similarity,
                'change_type': change.change_type.value
            }
        )

    def _article_change_to_hunk(self, article_change: ArticleChange) -> PatchHunk:
        """Convert ArticleChange to PatchHunk"""
        if article_change.change_type == ArticleChangeType.ADDED:
            operation = PatchOperation.ADD
        elif article_change.change_type == ArticleChangeType.DELETED:
            operation = PatchOperation.DELETE
        elif article_change.change_type == ArticleChangeType.RENAMED:
            operation = PatchOperation.RENAME
        elif article_change.change_type == ArticleChangeType.MOVED:
            operation = PatchOperation.MOVE
        else:
            operation = PatchOperation.REPLACE

        # Get article numbers
        old_article_num = None
        new_article_num = None
        old_content = None
        new_content = None

        if article_change.old_article:
            old_article_num = article_change.old_article.article_number
            old_content = article_change.old_article.content

        if article_change.new_article:
            new_article_num = article_change.new_article.article_number
            new_content = article_change.new_article.content

        return PatchHunk(
            operation=operation,
            old_content=old_content,
            new_content=new_content,
            article_number=new_article_num or old_article_num,
            metadata={
                'old_article_number': old_article_num,
                'new_article_number': new_article_num,
                'similarity': article_change.similarity,
                'title_changed': article_change.title_changed,
                'paragraphs_added': article_change.paragraphs_added,
                'paragraphs_deleted': article_change.paragraphs_deleted,
                'paragraphs_modified': article_change.paragraphs_modified,
                'summary': article_change.summary
            }
        )

    def _semantic_change_to_hunk(self, semantic_change: SemanticChange) -> PatchHunk:
        """Convert SemanticChange to PatchHunk"""
        # Use underlying text change operation
        text_change = semantic_change.text_change
        if text_change.change_type == ChangeType.ADDED:
            operation = PatchOperation.ADD
        elif text_change.change_type == ChangeType.DELETED:
            operation = PatchOperation.DELETE
        else:
            operation = PatchOperation.REPLACE

        return PatchHunk(
            operation=operation,
            old_start=text_change.line_number_old,
            new_start=text_change.line_number_new,
            old_content=text_change.old_content,
            new_content=text_change.new_content,
            metadata={
                'semantic_type': semantic_change.semantic_type.value,
                'legal_impact': semantic_change.legal_impact.value,
                'confidence': semantic_change.confidence,
                'entities_added': semantic_change.entities_added,
                'entities_removed': semantic_change.entities_removed,
                'relationships_added': semantic_change.relationships_added,
                'relationships_removed': semantic_change.relationships_removed,
                'summary': semantic_change.summary
            }
        )

    def generate_reverse_patch(self, patch: Patch) -> Patch:
        """Generate reverse patch (undo)

        Args:
            patch: Original patch

        Returns:
            Reverse patch that undoes the original
        """
        logger.info("Generating reverse patch")

        reverse_hunks = []
        for hunk in patch.hunks:
            reverse_hunk = PatchHunk(
                operation=self._reverse_operation(hunk.operation),
                old_start=hunk.new_start,  # Swap
                old_count=hunk.new_count,
                new_start=hunk.old_start,
                new_count=hunk.old_count,
                old_content=hunk.new_content,  # Swap
                new_content=hunk.old_content,
                context_before=hunk.context_after,  # Swap
                context_after=hunk.context_before,
                article_number=hunk.article_number,
                metadata={
                    **hunk.metadata,
                    'reversed': True
                }
            )
            reverse_hunks.append(reverse_hunk)

        reverse_patch = Patch(
            patch_format=patch.patch_format,
            hunks=reverse_hunks,
            source_version=patch.target_version,  # Swap
            target_version=patch.source_version,
            description=f"Reverse of: {patch.description}" if patch.description else "Reverse patch",
            total_additions=patch.total_deletions,  # Swap
            total_deletions=patch.total_additions,
            total_modifications=patch.total_modifications,
            metadata={
                **patch.metadata,
                'is_reverse': True,
                'original_patch_created_at': patch.created_at
            }
        )

        logger.info(f"Generated reverse patch with {len(reverse_hunks)} hunks")
        return reverse_patch

    def _reverse_operation(self, operation: PatchOperation) -> PatchOperation:
        """Reverse a patch operation"""
        if operation == PatchOperation.ADD:
            return PatchOperation.DELETE
        elif operation == PatchOperation.DELETE:
            return PatchOperation.ADD
        else:
            return operation  # REPLACE, MOVE, RENAME are symmetric

    def apply_patch(
        self, text: str, patch: Patch, **kwargs
    ) -> Tuple[str, List[str]]:
        """Apply patch to text

        Args:
            text: Original text
            patch: Patch to apply
            **kwargs: Options
                - dry_run: Don't actually apply, just validate (default: False)
                - fail_on_conflict: Raise error on conflict (default: False)

        Returns:
            Tuple of (modified_text, conflicts)
        """
        dry_run = kwargs.get('dry_run', False)
        fail_on_conflict = kwargs.get('fail_on_conflict', False)

        logger.info(f"Applying patch (dry_run={dry_run})")

        conflicts = []
        modified_text = text

        # Apply hunks
        for i, hunk in enumerate(patch.hunks):
            try:
                modified_text = self._apply_hunk(modified_text, hunk)
            except ValueError as e:
                conflict = f"Conflict in hunk {i}: {str(e)}"
                conflicts.append(conflict)
                logger.warning(conflict)

                if fail_on_conflict:
                    raise ValueError(f"Failed to apply patch: {conflict}")

        if dry_run:
            logger.info("Dry run complete - no changes applied")
            return text, conflicts

        logger.info(f"Patch applied with {len(conflicts)} conflicts")
        return modified_text, conflicts

    def _apply_hunk(self, text: str, hunk: PatchHunk) -> str:
        """Apply a single hunk to text"""
        lines = text.split('\n')

        if hunk.operation == PatchOperation.ADD:
            # Insert new content
            if hunk.new_start is not None and hunk.new_content:
                insert_pos = hunk.new_start
                if 0 <= insert_pos <= len(lines):
                    lines.insert(insert_pos, hunk.new_content)
                else:
                    raise ValueError(f"Invalid insert position: {insert_pos}")

        elif hunk.operation == PatchOperation.DELETE:
            # Delete content
            if hunk.old_start is not None and hunk.old_count:
                start = hunk.old_start
                end = start + hunk.old_count
                if 0 <= start < len(lines):
                    del lines[start:end]
                else:
                    raise ValueError(f"Invalid delete position: {start}")

        elif hunk.operation == PatchOperation.REPLACE:
            # Replace content
            if hunk.old_start is not None and hunk.new_content:
                if 0 <= hunk.old_start < len(lines):
                    lines[hunk.old_start] = hunk.new_content
                else:
                    raise ValueError(f"Invalid replace position: {hunk.old_start}")

        return '\n'.join(lines)

    def validate_patch(self, text: str, patch: Patch) -> Tuple[bool, List[str]]:
        """Validate that patch can be applied to text

        Args:
            text: Text to validate against
            patch: Patch to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        logger.info("Validating patch")

        errors = []
        lines = text.split('\n')

        for i, hunk in enumerate(patch.hunks):
            # Validate hunk positions
            if hunk.old_start is not None:
                if hunk.old_start < 0 or hunk.old_start >= len(lines):
                    errors.append(f"Hunk {i}: Invalid old_start position {hunk.old_start}")

            if hunk.new_start is not None:
                if hunk.new_start < 0 or hunk.new_start > len(lines):
                    errors.append(f"Hunk {i}: Invalid new_start position {hunk.new_start}")

            # Validate content matches
            if hunk.operation == PatchOperation.DELETE or hunk.operation == PatchOperation.REPLACE:
                if hunk.old_start is not None and hunk.old_content:
                    if hunk.old_start < len(lines):
                        if lines[hunk.old_start].strip() != hunk.old_content.strip():
                            errors.append(f"Hunk {i}: Content mismatch at line {hunk.old_start}")
                    else:
                        errors.append(f"Hunk {i}: Line {hunk.old_start} does not exist")

        is_valid = len(errors) == 0
        logger.info(f"Validation {'passed' if is_valid else 'failed'} with {len(errors)} errors")

        return is_valid, errors

    def format_patch(self, patch: Patch, format: PatchFormat) -> str:
        """Format patch as string

        Args:
            patch: Patch to format
            format: Output format

        Returns:
            Formatted patch string
        """
        if format == PatchFormat.JSON:
            return patch.to_json()

        elif format == PatchFormat.UNIFIED:
            return self._format_unified_diff(patch)

        elif format == PatchFormat.ARTICLE_LEVEL:
            return self._format_article_level(patch)

        elif format == PatchFormat.SEMANTIC:
            return self._format_semantic(patch)

        else:
            # Default to JSON
            return patch.to_json()

    def _format_unified_diff(self, patch: Patch) -> str:
        """Format patch as unified diff"""
        lines = []
        lines.append(f"--- a/{patch.source_version or 'original'}")
        lines.append(f"+++ b/{patch.target_version or 'modified'}")

        for hunk in patch.hunks:
            old_start = hunk.old_start or 0
            old_count = hunk.old_count or 0
            new_start = hunk.new_start or 0
            new_count = hunk.new_count or 0

            lines.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@")

            if hunk.old_content:
                lines.append(f"-{hunk.old_content}")
            if hunk.new_content:
                lines.append(f"+{hunk.new_content}")

        return '\n'.join(lines)

    def _format_article_level(self, patch: Patch) -> str:
        """Format patch as article-level changes"""
        lines = []
        lines.append(f"Madde Seviyesi Değişiklikler")
        lines.append(f"=" * 50)

        for hunk in patch.hunks:
            article_num = hunk.article_number or "?"
            operation = hunk.operation.value

            lines.append(f"\n{operation}: Madde {article_num}")

            if hunk.metadata.get('summary'):
                lines.append(f"  Özet: {hunk.metadata['summary']}")

            if hunk.old_content and hunk.new_content:
                lines.append(f"  Eski içerik: {hunk.old_content[:100]}...")
                lines.append(f"  Yeni içerik: {hunk.new_content[:100]}...")

        return '\n'.join(lines)

    def _format_semantic(self, patch: Patch) -> str:
        """Format patch as semantic changes"""
        lines = []
        lines.append(f"Anlamsal Değişiklikler")
        lines.append(f"=" * 50)

        for hunk in patch.hunks:
            semantic_type = hunk.metadata.get('semantic_type', '?')
            legal_impact = hunk.metadata.get('legal_impact', '?')

            lines.append(f"\n{semantic_type} (Etki: {legal_impact})")

            if hunk.metadata.get('summary'):
                lines.append(f"  Özet: {hunk.metadata['summary']}")

            if hunk.metadata.get('entities_added'):
                lines.append(f"  Eklenen varlıklar: {hunk.metadata['entities_added']}")

            if hunk.metadata.get('relationships_added'):
                lines.append(f"  Eklenen ilişkiler: {hunk.metadata['relationships_added']}")

        return '\n'.join(lines)


__all__ = [
    'PatchGenerator',
    'Patch',
    'PatchHunk',
    'PatchFormat',
    'PatchOperation'
]
