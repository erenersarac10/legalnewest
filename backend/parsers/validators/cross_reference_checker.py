"""Cross Reference Checker - Harvey/Legora CTO-Level Production-Grade
Validates cross-references in Turkish legal documents

Production Features:
- Cross-reference validation between articles
- Reference existence verification
- Reference direction validation (forward/backward)
- Broken reference detection
- Reference chain validation
- Turkish legal reference pattern recognition
- Circular reference detection
- Reference scope validation (internal/external)
- Contextual reference resolution
- Production-grade error messages with suggestions
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import time
import re
from collections import defaultdict

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


class CrossReferenceChecker(BaseValidator):
    """Cross Reference Checker for Turkish Legal Documents

    Validates cross-references and citations:
    - Article reference existence
    - Reference direction validation
    - Broken reference detection
    - Reference chain integrity
    - Circular reference detection
    - Turkish legal reference patterns
    - Internal vs external references
    - Reference context validation

    Features:
    - Turkish legal reference patterns
    - Multi-level reference chains
    - Reference scope validation
    - Contextual reference resolution
    - Detailed error reporting
    """

    # Turkish legal reference patterns
    REFERENCE_PATTERNS = {
        # Direct article references: "madde 5", "5. madde", "5 inci madde"
        'article_direct': [
            r'(?:madde\s+)?(\d+)(?:\s*\.?\s*madde)',
            r'(\d+)(?:\s*(?:inci|nci|üncü|ncı|uncu|ncu)\s+madde)',
            r'(?:madde\s+)?(\d+)',
        ],

        # Relative references: "yukarıda belirtilen madde", "aşağıdaki fıkra"
        'article_relative': [
            r'yukarı(?:da|ki)\s+(?:belirtilen|anılan|gösterilen)\s+madde',
            r'aşağı(?:da|ki)\s+(?:belirtilen|anılan|gösterilen)\s+madde',
            r'bu\s+madde',
            r'önceki\s+madde',
            r'sonraki\s+madde',
        ],

        # Paragraph references: "fıkra 2", "birinci fıkra"
        'paragraph': [
            r'(?:fıkra\s+)?(\d+)(?:\s*\.?\s*fıkra)',
            r'(birinci|ikinci|üçüncü|dördüncü|beşinci|altıncı|yedinci|sekizinci|dokuzuncu|onuncu)\s+fıkra',
            r'(\d+)(?:\s*(?:inci|nci)\s+fıkra)',
        ],

        # Section references: "bölüm", "kısım"
        'section': [
            r'(?:bölüm\s+)?(\d+)(?:\s*\.?\s*bölüm)',
            r'(birinci|ikinci|üçüncü)\s+bölüm',
            r'(?:kısım\s+)?(\d+)(?:\s*\.?\s*kısım)',
        ],

        # External references: "5237 sayılı Kanun", "KVKK'nın 5. maddesi"
        'external_law': [
            r'(\d{4})\s+sayılı\s+(?:Kanun|kanun)',
            r'([\w\s]+?)\s+Kanunu(?:\'?n[uı]n)?\s+(\d+)(?:\.|\s+inci)?\s+madde',
            r'(KVKK|TCK|HMK|CMK|TTK|TBK|TMK)(?:\'?n[ıi]n)?\s+(\d+)(?:\.|\s+inci)?\s+madde',
        ],

        # Chained references: "5. maddenin 2. fıkrası"
        'chained': [
            r'(\d+)(?:\s*\.?\s*madde)(?:nin|nın|nun)\s+(\d+)(?:\s*\.?\s*fıkra)',
            r'(\d+)(?:\s*(?:inci|nci)\s+madde)(?:nin|nın)\s+(birinci|ikinci|üçüncü)\s+fıkra',
        ],
    }

    # Turkish ordinal numbers to integers
    ORDINAL_TO_NUMBER = {
        'birinci': 1, 'ikinci': 2, 'üçüncü': 3, 'dördüncü': 4, 'beşinci': 5,
        'altıncı': 6, 'yedinci': 7, 'sekizinci': 8, 'dokuzuncu': 9, 'onuncu': 10,
        'onbirinci': 11, 'onikinci': 12, 'onüçüncü': 13, 'ondördüncü': 14, 'onbeşinci': 15,
    }

    def __init__(self):
        """Initialize Cross Reference Checker"""
        super().__init__(name="Cross Reference Checker")

        # Compile patterns for performance
        self.compiled_patterns = {}
        for pattern_type, patterns in self.REFERENCE_PATTERNS.items():
            self.compiled_patterns[pattern_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def validate(self, data: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate cross-references in document

        Args:
            data: Document data dictionary
            **kwargs: Options
                - check_external: Check external references (default: False)
                - strict: Fail on warnings (default: False)
                - max_chain_depth: Maximum reference chain depth (default: 10)

        Returns:
            ValidationResult with cross-reference validation issues
        """
        start_time = time.time()
        result = self.create_result()

        # Extract options
        check_external = kwargs.get('check_external', False)
        max_chain_depth = kwargs.get('max_chain_depth', 10)

        logger.info("Validating cross-references")

        # Build article index
        article_index = self._build_article_index(data)

        if not article_index:
            self.add_warning(
                result,
                "NO_ARTICLES",
                "No articles found in document",
                suggestion="Ensure document has 'articles' field with article data"
            )
            return self.finalize_result(result, start_time)

        logger.debug(f"Built index with {len(article_index)} articles")

        # Extract all references
        references = self._extract_references(data, article_index)

        if not references:
            self.add_info(
                result,
                "NO_REFERENCES",
                "No cross-references found in document",
                metadata={'article_count': len(article_index)}
            )
            return self.finalize_result(result, start_time)

        logger.debug(f"Found {len(references)} references")

        # Validate reference existence
        self._validate_reference_existence(references, article_index, result)

        # Validate reference directions
        self._validate_reference_directions(references, article_index, result)

        # Validate reference chains
        self._validate_reference_chains(references, article_index, result, max_chain_depth)

        # Detect circular references
        self._detect_circular_references(references, result)

        # Validate reference scopes
        self._validate_reference_scopes(references, result)

        # Check external references if requested
        if check_external:
            self._validate_external_references(references, result)

        # Validate relative references
        self._validate_relative_references(references, article_index, result)

        return self.finalize_result(result, start_time)

    def _build_article_index(self, data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Build index of articles in document

        Args:
            data: Document data

        Returns:
            Dictionary mapping article number to article data
        """
        index = {}

        # Get articles from various possible locations
        articles = data.get('articles', [])

        # Handle both list and dict formats
        if isinstance(articles, dict):
            articles = list(articles.values())

        if not isinstance(articles, list):
            return index

        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                continue

            # Get article number
            article_num = article.get('number')

            # Try to convert to int
            if article_num is not None:
                try:
                    article_num = int(article_num)
                except (ValueError, TypeError):
                    # Try to extract number from string
                    if isinstance(article_num, str):
                        match = re.search(r'\d+', article_num)
                        if match:
                            article_num = int(match.group())
                        else:
                            continue
                    else:
                        continue
            else:
                # Use index as article number if not specified
                article_num = i + 1

            index[article_num] = {
                'number': article_num,
                'content': article.get('content', ''),
                'title': article.get('title'),
                'paragraphs': article.get('paragraphs', []),
                'index': i,
            }

        return index

    def _extract_references(
        self,
        data: Dict[str, Any],
        article_index: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract all references from document

        Args:
            data: Document data
            article_index: Article index

        Returns:
            List of reference dictionaries
        """
        references = []

        for article_num, article_data in article_index.items():
            content = article_data.get('content', '')

            if not content:
                continue

            # Extract direct article references
            for pattern in self.compiled_patterns['article_direct']:
                for match in pattern.finditer(content):
                    try:
                        target_article = int(match.group(1))
                        references.append({
                            'source_article': article_num,
                            'target_article': target_article,
                            'type': 'article_direct',
                            'text': match.group(0),
                            'position': match.start(),
                            'scope': 'internal',
                        })
                    except (ValueError, IndexError):
                        continue

            # Extract relative references
            for pattern in self.compiled_patterns['article_relative']:
                for match in pattern.finditer(content):
                    references.append({
                        'source_article': article_num,
                        'target_article': None,  # Will be resolved
                        'type': 'article_relative',
                        'text': match.group(0),
                        'position': match.start(),
                        'scope': 'internal',
                        'relative_type': self._classify_relative_reference(match.group(0)),
                    })

            # Extract chained references
            for pattern in self.compiled_patterns['chained']:
                for match in pattern.finditer(content):
                    try:
                        target_article = int(match.group(1))
                        paragraph_ref = match.group(2)

                        # Convert ordinal to number if needed
                        if paragraph_ref in self.ORDINAL_TO_NUMBER:
                            paragraph_num = self.ORDINAL_TO_NUMBER[paragraph_ref]
                        else:
                            paragraph_num = int(paragraph_ref)

                        references.append({
                            'source_article': article_num,
                            'target_article': target_article,
                            'target_paragraph': paragraph_num,
                            'type': 'chained',
                            'text': match.group(0),
                            'position': match.start(),
                            'scope': 'internal',
                        })
                    except (ValueError, IndexError):
                        continue

            # Extract external references
            for pattern in self.compiled_patterns['external_law']:
                for match in pattern.finditer(content):
                    references.append({
                        'source_article': article_num,
                        'type': 'external_law',
                        'text': match.group(0),
                        'position': match.start(),
                        'scope': 'external',
                        'external_data': match.groups(),
                    })

        return references

    def _classify_relative_reference(self, text: str) -> str:
        """Classify relative reference type

        Args:
            text: Reference text

        Returns:
            Reference type: 'above', 'below', 'this', 'previous', 'next'
        """
        text_lower = text.lower()

        if 'yukarı' in text_lower:
            return 'above'
        elif 'aşağı' in text_lower:
            return 'below'
        elif 'bu madde' in text_lower:
            return 'this'
        elif 'önceki' in text_lower:
            return 'previous'
        elif 'sonraki' in text_lower:
            return 'next'

        return 'unknown'

    def _validate_reference_existence(
        self,
        references: List[Dict[str, Any]],
        article_index: Dict[int, Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate that referenced articles exist"""

        for ref in references:
            # Skip external and relative references
            if ref['type'] in ['external_law', 'article_relative']:
                continue

            target_article = ref.get('target_article')

            if target_article is None:
                continue

            passed = target_article in article_index
            self.update_check_stats(result, passed)

            if not passed:
                source_article = ref['source_article']
                available_articles = sorted(article_index.keys())

                # Find closest article number
                closest = min(available_articles, key=lambda x: abs(x - target_article))

                self.add_error(
                    result,
                    "BROKEN_REFERENCE",
                    f"Article {source_article} references non-existent Article {target_article}",
                    location=f"Article {source_article}",
                    context=f"Reference text: '{ref['text']}'",
                    suggestion=f"Article {target_article} does not exist. Did you mean Article {closest}?",
                    metadata={
                        'source': source_article,
                        'target': target_article,
                        'available_articles': available_articles,
                    }
                )

    def _validate_reference_directions(
        self,
        references: List[Dict[str, Any]],
        article_index: Dict[int, Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate reference directions (forward/backward)"""

        for ref in references:
            if ref['type'] not in ['article_direct', 'chained']:
                continue

            source = ref['source_article']
            target = ref.get('target_article')

            if target is None or target not in article_index:
                continue

            # Determine direction
            if target > source:
                direction = 'forward'
            elif target < source:
                direction = 'backward'
            else:
                direction = 'self'

            ref['direction'] = direction

            # Check for self-references (usually suspicious)
            if direction == 'self':
                self.update_check_stats(result, False)
                self.add_warning(
                    result,
                    "SELF_REFERENCE",
                    f"Article {source} references itself",
                    location=f"Article {source}",
                    context=f"Reference text: '{ref['text']}'",
                    suggestion="Verify that self-reference is intentional"
                )
            else:
                self.update_check_stats(result, True)

    def _validate_reference_chains(
        self,
        references: List[Dict[str, Any]],
        article_index: Dict[int, Dict[str, Any]],
        result: ValidationResult,
        max_depth: int
    ) -> None:
        """Validate reference chains"""

        # Build adjacency list
        ref_graph = defaultdict(list)
        for ref in references:
            if ref['type'] in ['article_direct', 'chained']:
                source = ref['source_article']
                target = ref.get('target_article')
                if target and target in article_index:
                    ref_graph[source].append(target)

        # Check chain depths
        for start_article in ref_graph:
            depth = self._get_chain_depth(start_article, ref_graph, set())

            if depth > max_depth:
                self.update_check_stats(result, False)
                self.add_warning(
                    result,
                    "DEEP_REFERENCE_CHAIN",
                    f"Article {start_article} has reference chain depth of {depth} (max: {max_depth})",
                    location=f"Article {start_article}",
                    suggestion="Consider simplifying reference structure",
                    metadata={'depth': depth, 'max_depth': max_depth}
                )
            else:
                self.update_check_stats(result, True)

    def _get_chain_depth(
        self,
        article: int,
        graph: Dict[int, List[int]],
        visited: Set[int]
    ) -> int:
        """Get maximum reference chain depth from article"""

        if article in visited:
            return 0  # Avoid infinite recursion

        visited.add(article)

        if article not in graph or not graph[article]:
            return 1

        max_depth = 0
        for target in graph[article]:
            depth = self._get_chain_depth(target, graph, visited.copy())
            max_depth = max(max_depth, depth)

        return max_depth + 1

    def _detect_circular_references(
        self,
        references: List[Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Detect circular reference chains"""

        # Build adjacency list
        ref_graph = defaultdict(list)
        for ref in references:
            if ref['type'] in ['article_direct', 'chained']:
                source = ref['source_article']
                target = ref.get('target_article')
                if target:
                    ref_graph[source].append(target)

        # Find cycles using DFS
        visited = set()
        rec_stack = set()

        for article in ref_graph:
            if article not in visited:
                cycle = self._detect_cycle_dfs(article, ref_graph, visited, rec_stack, [])

                if cycle:
                    self.update_check_stats(result, False)
                    cycle_str = ' -> '.join(f"Article {a}" for a in cycle)
                    self.add_error(
                        result,
                        "CIRCULAR_REFERENCE",
                        f"Circular reference detected: {cycle_str}",
                        location=f"Article {cycle[0]}",
                        suggestion="Remove or restructure circular references",
                        metadata={'cycle': cycle}
                    )
                else:
                    self.update_check_stats(result, True)

    def _detect_cycle_dfs(
        self,
        node: int,
        graph: Dict[int, List[int]],
        visited: Set[int],
        rec_stack: Set[int],
        path: List[int]
    ) -> Optional[List[int]]:
        """DFS-based cycle detection

        Returns:
            Cycle path if found, None otherwise
        """
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                cycle = self._detect_cycle_dfs(neighbor, graph, visited, rec_stack, path)
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]

        rec_stack.remove(node)
        path.pop()
        return None

    def _validate_reference_scopes(
        self,
        references: List[Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate reference scopes (internal vs external)"""

        internal_count = sum(1 for ref in references if ref['scope'] == 'internal')
        external_count = sum(1 for ref in references if ref['scope'] == 'external')

        self.add_info(
            result,
            "REFERENCE_SCOPE_STATS",
            f"Found {internal_count} internal and {external_count} external references",
            metadata={
                'internal_count': internal_count,
                'external_count': external_count,
                'total_count': len(references)
            }
        )

        # Check for excessive external references
        if external_count > internal_count and internal_count > 0:
            self.add_warning(
                result,
                "EXCESSIVE_EXTERNAL_REFERENCES",
                f"Document has more external ({external_count}) than internal ({internal_count}) references",
                suggestion="Verify that external references are necessary"
            )

    def _validate_external_references(
        self,
        references: List[Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate external references (to other laws)"""

        external_refs = [ref for ref in references if ref['type'] == 'external_law']

        if not external_refs:
            return

        # Group by referenced law
        law_refs = defaultdict(list)
        for ref in external_refs:
            law_identifier = ref.get('external_data', [''])[0]
            law_refs[law_identifier].append(ref)

        # Report external reference summary
        for law_id, refs in law_refs.items():
            self.add_info(
                result,
                "EXTERNAL_LAW_REFERENCE",
                f"Found {len(refs)} references to: {law_id}",
                metadata={
                    'law': law_id,
                    'reference_count': len(refs),
                    'source_articles': [ref['source_article'] for ref in refs]
                }
            )

    def _validate_relative_references(
        self,
        references: List[Dict[str, Any]],
        article_index: Dict[int, Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate relative references (yukarıda, aşağıda, etc.)"""

        relative_refs = [ref for ref in references if ref['type'] == 'article_relative']

        for ref in relative_refs:
            source = ref['source_article']
            relative_type = ref.get('relative_type', 'unknown')

            # Try to resolve relative reference
            resolved = self._resolve_relative_reference(source, relative_type, article_index)

            if resolved:
                ref['target_article'] = resolved
                self.update_check_stats(result, True)
                self.add_info(
                    result,
                    "RELATIVE_REFERENCE_RESOLVED",
                    f"Resolved relative reference in Article {source} to Article {resolved}",
                    location=f"Article {source}",
                    context=f"Reference text: '{ref['text']}'",
                    metadata={'resolved_to': resolved, 'relative_type': relative_type}
                )
            else:
                self.update_check_stats(result, False)
                self.add_warning(
                    result,
                    "AMBIGUOUS_RELATIVE_REFERENCE",
                    f"Could not resolve relative reference in Article {source}",
                    location=f"Article {source}",
                    context=f"Reference text: '{ref['text']}'",
                    suggestion="Use explicit article numbers instead of relative references",
                    metadata={'relative_type': relative_type}
                )

    def _resolve_relative_reference(
        self,
        source_article: int,
        relative_type: str,
        article_index: Dict[int, Dict[str, Any]]
    ) -> Optional[int]:
        """Resolve relative reference to specific article

        Args:
            source_article: Source article number
            relative_type: Type of relative reference
            article_index: Article index

        Returns:
            Resolved article number or None
        """
        article_numbers = sorted(article_index.keys())

        try:
            source_idx = article_numbers.index(source_article)
        except ValueError:
            return None

        if relative_type == 'previous':
            if source_idx > 0:
                return article_numbers[source_idx - 1]
        elif relative_type == 'next':
            if source_idx < len(article_numbers) - 1:
                return article_numbers[source_idx + 1]
        elif relative_type == 'this':
            return source_article
        elif relative_type in ['above', 'below']:
            # These are too ambiguous to resolve automatically
            return None

        return None


__all__ = ['CrossReferenceChecker']
