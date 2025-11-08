"""Clause Hierarchy Builder - Harvey/Legora CTO-Level
Builds tree structure from MADDE → FIKRA → BENT → ALT BENT hierarchy
"""
from typing import Dict, List, Any, Optional
import re
from ..core import LegalClause
from ..presets import CLAUSE_PATTERNS

class ClauseNode:
    """Represents a node in the clause hierarchy tree"""
    def __init__(self, clause_type: str, number: Optional[str], content: str, level: int):
        self.clause_type = clause_type  # 'article', 'paragraph', 'clause', 'subclause'
        self.number = number
        self.content = content
        self.level = level
        self.children: List['ClauseNode'] = []
        self.parent: Optional['ClauseNode'] = None

    def add_child(self, child: 'ClauseNode'):
        """Add child node and set parent relationship"""
        child.parent = self
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation"""
        return {
            'type': self.clause_type,
            'number': self.number,
            'content': self.content,
            'level': self.level,
            'children': [child.to_dict() for child in self.children]
        }

    def get_path(self) -> str:
        """Get hierarchical path (e.g., 'M.5/F.2/B.a')"""
        if self.parent:
            parent_path = self.parent.get_path()
            return f"{parent_path}/{self._get_prefix()}.{self.number}" if parent_path else f"{self._get_prefix()}.{self.number}"
        return f"{self._get_prefix()}.{self.number}"

    def _get_prefix(self) -> str:
        """Get prefix for this clause type"""
        prefixes = {
            'article': 'M',      # Madde
            'paragraph': 'F',     # Fıkra
            'clause': 'B',        # Bent
            'subclause': 'AB'     # Alt Bent
        }
        return prefixes.get(self.clause_type, 'X')


class ClauseHierarchyBuilder:
    """
    Builds hierarchical tree structure from Turkish legal document clauses.

    Turkish Legal Hierarchy:
    - MADDE (Article): Numbered 1, 2, 3...
    - FIKRA (Paragraph): Numbered (1), (2), (3)... or implicit first paragraph
    - BENT (Clause): Lettered a), b), c)...
    - ALT BENT (Subclause): Roman numerals i), ii), iii)... or numbered 1), 2), 3)

    Example:
        MADDE 5 - Şirket yönetimi
        (1) Şirket, yönetim kurulu tarafından yönetilir.  [Fıkra 1]
        (2) Yönetim kurulunun görevleri şunlardır:      [Fıkra 2]
        a) Genel kurul kararlarını uygulamak,            [Bent a]
        b) Şirket işlerini yürütmek,                     [Bent b]
            i) Bütçe hazırlamak,                          [Alt bent i]
            ii) Personel istihdam etmek.                  [Alt bent ii]
    """

    def __init__(self):
        self.patterns = CLAUSE_PATTERNS

    def build_hierarchy(self, article_text: str, article_number: int) -> ClauseNode:
        """
        Build hierarchical tree from article text.

        Args:
            article_text: Full text of the article (including all fıkra/bent)
            article_number: Article number

        Returns:
            Root ClauseNode representing the article with full hierarchy
        """
        # Create root article node
        root = ClauseNode('article', str(article_number), '', level=0)

        # Split into paragraphs (fıkra)
        paragraphs = self._extract_paragraphs(article_text)

        for para_idx, para_text in enumerate(paragraphs):
            para_number = str(para_idx + 1) if len(paragraphs) > 1 else '1'
            para_node = ClauseNode('paragraph', para_number, '', level=1)
            root.add_child(para_node)

            # Extract clauses (bent) from this paragraph
            clauses = self._extract_clauses(para_text)

            if clauses:
                # Has explicit clauses
                for clause_letter, clause_text in clauses:
                    clause_node = ClauseNode('clause', clause_letter, clause_text, level=2)
                    para_node.add_child(clause_node)

                    # Extract subclauses (alt bent) from this clause
                    subclauses = self._extract_subclauses(clause_text)
                    for subclause_num, subclause_text in subclauses:
                        subclause_node = ClauseNode('subclause', subclause_num, subclause_text, level=3)
                        clause_node.add_child(subclause_node)
            else:
                # No explicit clauses, paragraph content is the text itself
                para_node.content = para_text.strip()

        return root

    def _extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs (fıkra) from article text.

        Turkish law paragraphs are marked with (1), (2), (3)...
        First paragraph often has no marker.
        """
        # Pattern for numbered paragraphs: (1), (2), etc.
        pattern = re.compile(r'\((\d+)\)\s*')

        matches = list(pattern.finditer(text))

        if not matches:
            # No explicit paragraph markers - entire text is one paragraph
            return [text.strip()]

        paragraphs = []

        # Check if there's text before first numbered paragraph (implicit first paragraph)
        if matches[0].start() > 0:
            first_para = text[:matches[0].start()].strip()
            if first_para:
                paragraphs.append(first_para)

        # Extract numbered paragraphs
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            para_text = text[start:end].strip()
            if para_text:
                paragraphs.append(para_text)

        return paragraphs

    def _extract_clauses(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract clauses (bent) marked with a), b), c)... ç), d)...

        Turkish alphabet for bents: a, b, c, ç, d, e, f, g, ğ, h, ı, i, j, k, l, m, n, o, ö, p, r, s, ş, t, u, ü, v, y, z
        """
        pattern = re.compile(r'^([a-zçğıİöşü])\)\s+(.+?)(?=\n[a-zçğıİöşü]\)|\Z)', re.MULTILINE | re.DOTALL)

        clauses = []
        for match in pattern.finditer(text):
            letter = match.group(1)
            clause_text = match.group(2).strip()
            clauses.append((letter, clause_text))

        return clauses

    def _extract_subclauses(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract subclauses (alt bent) marked with i), ii), iii)... or 1), 2), 3)...
        """
        # Try Roman numerals first
        roman_pattern = re.compile(r'^([ivxlcdm]+)\)\s+(.+?)(?=\n[ivxlcdm]+\)|\Z)', re.MULTILINE | re.DOTALL | re.IGNORECASE)

        matches = list(roman_pattern.finditer(text))
        if matches:
            return [(match.group(1), match.group(2).strip()) for match in matches]

        # Try numbered subclauses
        num_pattern = re.compile(r'^(\d+)\)\s+(.+?)(?=\n\d+\)|\Z)', re.MULTILINE | re.DOTALL)
        matches = list(num_pattern.finditer(text))
        if matches:
            return [(match.group(1), match.group(2).strip()) for match in matches]

        return []

    def build_from_articles(self, articles: List[Dict[str, Any]]) -> List[ClauseNode]:
        """
        Build hierarchy trees for multiple articles.

        Args:
            articles: List of article dicts with 'number' and 'content'/'full_text'

        Returns:
            List of root ClauseNodes (one per article)
        """
        hierarchies = []

        for article in articles:
            number = article.get('number')
            text = article.get('full_text') or article.get('content', '')

            if number and text:
                hierarchy = self.build_hierarchy(text, number)
                hierarchies.append(hierarchy)

        return hierarchies

    def to_legal_clauses(self, root: ClauseNode) -> List[LegalClause]:
        """
        Convert ClauseNode tree to list of LegalClause objects.

        Flattens the tree while preserving hierarchy information.
        """
        from ..core.canonical_schema import LegalClause

        clauses = []

        def traverse(node: ClauseNode, path: str = ''):
            current_path = node.get_path()

            clause = LegalClause(
                clause_type=node.clause_type,
                number=node.number,
                content=node.content,
                level=node.level,
                parent_path=path if path else None,
                full_path=current_path
            )
            clauses.append(clause)

            # Recursively process children
            for child in node.children:
                traverse(child, current_path)

        traverse(root)
        return clauses

__all__ = ['ClauseHierarchyBuilder', 'ClauseNode']
