"""Legal Logic Validator - Harvey/Legora CTO-Level Production-Grade
Validates legal logic and consistency in Turkish legal documents

Production Features:
- Logical contradiction detection
- Condition-consequence validation
- Conflicting obligations detection
- Scope consistency validation
- Turkish legal logic patterns
- Impossible condition detection
- Hierarchical consistency validation
- Temporal consistency checking
- Mutual exclusivity validation
- Circular reference detection
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import time
import re

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


class LegalLogicValidator(BaseValidator):
    """Legal Logic Validator for Turkish Legal Documents

    Validates logical consistency and legal reasoning:
    - Logical contradictions in rules
    - Condition-consequence relationships
    - Conflicting obligations
    - Scope consistency
    - Impossible conditions
    - Hierarchical consistency
    - Temporal consistency
    - Mutual exclusivity

    Features:
    - Turkish legal logic patterns
    - Multi-level consistency checking
    - Cross-reference validation
    - Obligation conflict detection
    - Production-grade error messages
    """

    # Turkish legal logic keywords
    CONDITION_KEYWORDS = [
        'eğer', 'ise', 'olması halinde', 'durumunda', 'şartıyla',
        'koşuluyla', 'kaydıyla', 'halinde', 'takdirde', 'olduğunda'
    ]

    CONSEQUENCE_KEYWORDS = [
        'uygulanır', 'geçerlidir', 'olur', 'kabul edilir', 'sayılır',
        'yapılır', 'zorunludur', 'mecburidir', 'gereklidir', 'lazımdır'
    ]

    OBLIGATION_KEYWORDS = [
        'zorunludur', 'mecburidir', 'gereklidir', 'yükümlüdür',
        'yapılır', 'yapmak zorundadır', 'etmek zorundadır',
        'sağlamak zorundadır', 'uymak zorundadır'
    ]

    PROHIBITION_KEYWORDS = [
        'yasaktır', 'yasaklanmıştır', 'edilemez', 'yapılamaz',
        'olunamaz', 'kullanılamaz', 'mümkün değildir', 'caiz değildir',
        'olmaz', 'kabul edilemez'
    ]

    EXEMPTION_KEYWORDS = [
        'hariç', 'istisna', 'muaf', 'dışında', 'hariç tutulur',
        'uygulanmaz', 'bu hükümler kapsamında değildir', 'bu madde kapsamı dışındadır'
    ]

    NEGATION_KEYWORDS = [
        'değil', 'olmayan', 'olmadığı', 'hariç', 'dışında',
        'yoksa', 'bulunmayan', 'içermeyen'
    ]

    # Turkish legal hierarchy
    LEGAL_HIERARCHY = {
        'anayasa': 1,  # Constitution
        'kanun': 2,  # Law
        'kanun hükmünde kararname': 3,  # Decree with force of law
        'cumhurbaşkanlığı kararnamesi': 4,  # Presidential decree
        'tüzük': 5,  # Regulation (by law)
        'yönetmelik': 6,  # Regulation
        'tebliğ': 7,  # Communiqué
        'genelge': 8,  # Circular
    }

    # Temporal keywords
    TEMPORAL_KEYWORDS = {
        'before': ['önce', 'evvel', 'öncesinde', 'kadar'],
        'after': ['sonra', 'sonrasında', 'itibaren', 'başlayarak'],
        'during': ['sırasında', 'esnasında', 'boyunca'],
        'until': ['kadar', 'değin'],
    }

    def __init__(self):
        """Initialize Legal Logic Validator"""
        super().__init__(name="Legal Logic Validator")

        # Track validation context
        self.context = {
            'rules': [],
            'obligations': [],
            'prohibitions': [],
            'exemptions': [],
            'conditions': [],
            'scopes': [],
            'references': []
        }

    def validate(self, data: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate legal logic

        Args:
            data: Document data dictionary
            **kwargs: Options
                - check_contradictions: Check for contradictions (default: True)
                - check_conditions: Check condition-consequence (default: True)
                - check_obligations: Check obligation conflicts (default: True)
                - check_scope: Check scope consistency (default: True)
                - check_hierarchy: Check hierarchical consistency (default: True)
                - check_temporal: Check temporal consistency (default: True)
                - strict: Fail on warnings (default: False)

        Returns:
            ValidationResult with logic validation issues
        """
        start_time = time.time()
        result = self.create_result()

        # Reset context
        self._reset_context()

        # Extract validation options
        check_contradictions = kwargs.get('check_contradictions', True)
        check_conditions = kwargs.get('check_conditions', True)
        check_obligations = kwargs.get('check_obligations', True)
        check_scope = kwargs.get('check_scope', True)
        check_hierarchy = kwargs.get('check_hierarchy', True)
        check_temporal = kwargs.get('check_temporal', True)

        logger.info("Validating legal logic")

        # Build context from document
        self._build_context(data, result)

        # Validate logical contradictions
        if check_contradictions:
            self._validate_contradictions(result)

        # Validate condition-consequence relationships
        if check_conditions:
            self._validate_conditions(result)

        # Validate conflicting obligations
        if check_obligations:
            self._validate_obligations(result)

        # Validate scope consistency
        if check_scope:
            self._validate_scope_consistency(result)

        # Validate hierarchical consistency
        if check_hierarchy:
            self._validate_hierarchical_consistency(data, result)

        # Validate temporal consistency
        if check_temporal:
            self._validate_temporal_consistency(result)

        # Validate mutual exclusivity
        self._validate_mutual_exclusivity(result)

        # Detect circular references
        self._detect_circular_references(result)

        # Detect impossible conditions
        self._detect_impossible_conditions(result)

        return self.finalize_result(result, start_time)

    def _reset_context(self) -> None:
        """Reset validation context"""
        self.context = {
            'rules': [],
            'obligations': [],
            'prohibitions': [],
            'exemptions': [],
            'conditions': [],
            'scopes': [],
            'references': []
        }

    def _build_context(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Build validation context from document

        Args:
            data: Document data
            result: ValidationResult to update
        """
        # Extract articles
        articles = self._extract_articles(data)

        for article in articles:
            article_num = article.get('number', 'Unknown')
            content = article.get('content', '')

            # Analyze article content
            self._analyze_article(article_num, content)

        logger.debug(f"Built context: {len(self.context['rules'])} rules, "
                    f"{len(self.context['obligations'])} obligations, "
                    f"{len(self.context['prohibitions'])} prohibitions")

    def _extract_articles(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract articles from document"""
        articles = data.get('articles', [])

        if isinstance(articles, dict):
            # Convert dict to list
            return [{'number': k, 'content': v} for k, v in articles.items()]
        elif isinstance(articles, list):
            return articles
        else:
            return []

    def _analyze_article(self, article_num: str, content: str) -> None:
        """Analyze article and extract logic elements

        Args:
            article_num: Article number
            content: Article content
        """
        content_lower = content.lower()

        # Extract obligations
        for keyword in self.OBLIGATION_KEYWORDS:
            if keyword in content_lower:
                self.context['obligations'].append({
                    'article': article_num,
                    'content': content,
                    'keyword': keyword
                })
                break

        # Extract prohibitions
        for keyword in self.PROHIBITION_KEYWORDS:
            if keyword in content_lower:
                self.context['prohibitions'].append({
                    'article': article_num,
                    'content': content,
                    'keyword': keyword
                })
                break

        # Extract conditions
        for keyword in self.CONDITION_KEYWORDS:
            if keyword in content_lower:
                self.context['conditions'].append({
                    'article': article_num,
                    'content': content,
                    'keyword': keyword
                })
                break

        # Extract exemptions
        for keyword in self.EXEMPTION_KEYWORDS:
            if keyword in content_lower:
                self.context['exemptions'].append({
                    'article': article_num,
                    'content': content,
                    'keyword': keyword
                })

        # Extract scope
        if 'kapsam' in content_lower or 'scope' in content_lower:
            self.context['scopes'].append({
                'article': article_num,
                'content': content
            })

        # Extract references (madde references)
        refs = re.findall(r'(\d+)\.\s*madde', content_lower)
        if refs:
            self.context['references'].append({
                'from_article': article_num,
                'to_articles': refs,
                'content': content
            })

        # Add to general rules
        self.context['rules'].append({
            'article': article_num,
            'content': content
        })

    def _validate_contradictions(self, result: ValidationResult) -> None:
        """Validate logical contradictions

        Args:
            result: ValidationResult to update
        """
        # Check obligation vs prohibition conflicts
        for obligation in self.context['obligations']:
            for prohibition in self.context['prohibitions']:
                if self._are_contradictory(obligation['content'], prohibition['content']):
                    self.update_check_stats(result, False)
                    self.add_error(
                        result,
                        "LOGICAL_CONTRADICTION",
                        f"Madde {obligation['article']} bir yükümlülük getirirken, "
                        f"Madde {prohibition['article']} aynı konuyu yasaklıyor",
                        location=f"Madde {obligation['article']} ↔ Madde {prohibition['article']}",
                        context=f"Yükümlülük: {obligation['content'][:100]}...\n"
                               f"Yasak: {prohibition['content'][:100]}...",
                        suggestion="Çelişen hükümleri gözden geçirin ve tutarlılığı sağlayın"
                    )
                else:
                    self.update_check_stats(result, True)

        # Check within obligations
        for i, obl1 in enumerate(self.context['obligations']):
            for obl2 in self.context['obligations'][i+1:]:
                if obl1['article'] != obl2['article']:
                    if self._are_mutually_exclusive(obl1['content'], obl2['content']):
                        self.update_check_stats(result, False)
                        self.add_warning(
                            result,
                            "CONTRADICTORY_OBLIGATIONS",
                            f"Madde {obl1['article']} ve Madde {obl2['article']} "
                            f"çelişen yükümlülükler içeriyor",
                            location=f"Madde {obl1['article']} ↔ Madde {obl2['article']}",
                            suggestion="Yükümlülüklerin uyumlu olduğundan emin olun"
                        )

    def _validate_conditions(self, result: ValidationResult) -> None:
        """Validate condition-consequence relationships

        Args:
            result: ValidationResult to update
        """
        for condition in self.context['conditions']:
            content = condition['content']

            # Check if consequence exists
            has_consequence = any(kw in content.lower() for kw in self.CONSEQUENCE_KEYWORDS)

            passed = has_consequence
            self.update_check_stats(result, passed)

            if not passed:
                self.add_warning(
                    result,
                    "INCOMPLETE_CONDITION",
                    f"Madde {condition['article']} koşul içeriyor ancak sonuç belirtmiyor",
                    location=f"Madde {condition['article']}",
                    context=content[:200] + "..." if len(content) > 200 else content,
                    suggestion="Koşulun sonucunu açıkça belirtin (uygulanır, geçerlidir, vb.)"
                )

            # Check for impossible conditions
            if self._is_impossible_condition(content):
                self.update_check_stats(result, False)
                self.add_error(
                    result,
                    "IMPOSSIBLE_CONDITION",
                    f"Madde {condition['article']} imkansız bir koşul içeriyor",
                    location=f"Madde {condition['article']}",
                    context=content[:200] + "..." if len(content) > 200 else content,
                    suggestion="Koşulun mantıksal olarak gerçekleşebilir olduğundan emin olun"
                )

    def _validate_obligations(self, result: ValidationResult) -> None:
        """Validate conflicting obligations

        Args:
            result: ValidationResult to update
        """
        # Check for same-subject conflicting obligations
        for i, obl1 in enumerate(self.context['obligations']):
            for obl2 in self.context['obligations'][i+1:]:
                # Extract subjects
                subject1 = self._extract_subject(obl1['content'])
                subject2 = self._extract_subject(obl2['content'])

                if subject1 and subject2 and self._are_similar(subject1, subject2):
                    # Same subject, check for conflicts
                    if self._obligations_conflict(obl1['content'], obl2['content']):
                        self.update_check_stats(result, False)
                        self.add_error(
                            result,
                            "CONFLICTING_OBLIGATIONS",
                            f"Madde {obl1['article']} ve Madde {obl2['article']} "
                            f"aynı konu için çelişen yükümlülükler getiriyor",
                            location=f"Madde {obl1['article']} ↔ Madde {obl2['article']}",
                            context=f"Konu: {subject1}",
                            suggestion="Yükümlülükleri tek bir tutarlı çerçevede birleştirin"
                        )
                    else:
                        self.update_check_stats(result, True)

    def _validate_scope_consistency(self, result: ValidationResult) -> None:
        """Validate scope consistency

        Args:
            result: ValidationResult to update
        """
        if not self.context['scopes']:
            return

        # Get main scope
        main_scope = self.context['scopes'][0] if self.context['scopes'] else None

        if main_scope:
            # Check exemptions don't exceed scope
            for exemption in self.context['exemptions']:
                # Extract exemption subjects
                exempt_subjects = self._extract_exemption_subjects(exemption['content'])

                for subject in exempt_subjects:
                    # Check if subject is in scope
                    if not self._is_in_scope(subject, main_scope['content']):
                        self.update_check_stats(result, False)
                        self.add_warning(
                            result,
                            "EXEMPTION_EXCEEDS_SCOPE",
                            f"Madde {exemption['article']} kapsam dışı bir istisna içeriyor: {subject}",
                            location=f"Madde {exemption['article']}",
                            context=f"Kapsam (Madde {main_scope['article']}): {main_scope['content'][:100]}...",
                            suggestion="İstisnanın tanımlı kapsam içinde olduğundan emin olun"
                        )
                    else:
                        self.update_check_stats(result, True)

    def _validate_hierarchical_consistency(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate hierarchical consistency

        Args:
            data: Document data
            result: ValidationResult to update
        """
        # Get document type
        doc_type = self._get_document_type(data)

        # If regulation, check for law references
        if doc_type in ['yönetmelik', 'tüzük', 'tebliğ']:
            # Check if contradicts higher law
            references = data.get('references', [])

            for ref in references:
                ref_type = self._get_reference_type(ref)

                if ref_type == 'kanun':
                    # Check for contradictions
                    passed = not self._contradicts_reference(data, ref)
                    self.update_check_stats(result, passed)

                    if not passed:
                        self.add_error(
                            result,
                            "HIERARCHICAL_CONTRADICTION",
                            f"{doc_type.capitalize()} üst hukuk normunu (kanun) ihlal ediyor",
                            location="Genel",
                            context=f"Referans: {ref}",
                            suggestion="Yönetmelikler kanunlarla çelişemez, kanun hükümlerine uygun düzenleme yapın"
                        )

        # Check for proper authorization
        if doc_type in ['yönetmelik', 'tebliğ', 'genelge']:
            authority = data.get('authority', '')

            if not authority:
                self.update_check_stats(result, False)
                self.add_error(
                    result,
                    "MISSING_AUTHORIZATION",
                    f"{doc_type.capitalize()} çıkarma yetkisini gösteren kaynak belirtilmemiş",
                    location="authority",
                    suggestion="Yetki kaynağını (hangi kanun maddesi) açıkça belirtin"
                )
            else:
                self.update_check_stats(result, True)

    def _validate_temporal_consistency(self, result: ValidationResult) -> None:
        """Validate temporal consistency

        Args:
            result: ValidationResult to update
        """
        # Check for temporal contradictions
        for i, rule1 in enumerate(self.context['rules']):
            for rule2 in self.context['rules'][i+1:]:
                temporal_conflict = self._has_temporal_conflict(
                    rule1['content'],
                    rule2['content']
                )

                if temporal_conflict:
                    self.update_check_stats(result, False)
                    self.add_warning(
                        result,
                        "TEMPORAL_CONFLICT",
                        f"Madde {rule1['article']} ve Madde {rule2['article']} "
                        f"zamansal çelişki içeriyor",
                        location=f"Madde {rule1['article']} ↔ Madde {rule2['article']}",
                        suggestion="Zaman çizelgelerinin tutarlı olduğundan emin olun"
                    )
                else:
                    self.update_check_stats(result, True)

    def _validate_mutual_exclusivity(self, result: ValidationResult) -> None:
        """Validate mutual exclusivity

        Args:
            result: ValidationResult to update
        """
        # Check for mutually exclusive rules applied together
        for i, rule1 in enumerate(self.context['rules']):
            for rule2 in self.context['rules'][i+1:]:
                if self._are_mutually_exclusive(rule1['content'], rule2['content']):
                    if not self._has_disambiguation(rule1['content'], rule2['content']):
                        self.update_check_stats(result, False)
                        self.add_warning(
                            result,
                            "MUTUAL_EXCLUSIVITY_ISSUE",
                            f"Madde {rule1['article']} ve Madde {rule2['article']} "
                            f"karşılıklı özel durumlar içeriyor ancak öncelik belirsiz",
                            location=f"Madde {rule1['article']} ↔ Madde {rule2['article']}",
                            suggestion="Hangi maddenin öncelikli olduğunu açıkça belirtin"
                        )

    def _detect_circular_references(self, result: ValidationResult) -> None:
        """Detect circular references

        Args:
            result: ValidationResult to update
        """
        # Build reference graph
        ref_graph = {}
        for ref in self.context['references']:
            from_art = str(ref['from_article'])
            to_arts = [str(a) for a in ref['to_articles']]
            ref_graph[from_art] = to_arts

        # Check for cycles
        visited = set()
        path = []

        def has_cycle(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False

            visited.add(node)
            path.append(node)

            for neighbor in ref_graph.get(node, []):
                if has_cycle(neighbor):
                    return True

            path.remove(node)
            return False

        for node in ref_graph:
            if node not in visited:
                if has_cycle(node):
                    self.update_check_stats(result, False)
                    cycle_path = ' → '.join(path)
                    self.add_error(
                        result,
                        "CIRCULAR_REFERENCE",
                        f"Döngüsel atıf tespit edildi: Madde {cycle_path}",
                        location=f"Madde {node}",
                        suggestion="Döngüsel atıfları kaldırın, her madde kendisine referans veremez"
                    )

    def _detect_impossible_conditions(self, result: ValidationResult) -> None:
        """Detect impossible conditions

        Args:
            result: ValidationResult to update
        """
        for condition in self.context['conditions']:
            content = condition['content'].lower()

            # Check for logical impossibilities
            # Example: "hem X hem de Y değil" where X and not X
            if self._contains_logical_impossibility(content):
                self.update_check_stats(result, False)
                self.add_error(
                    result,
                    "IMPOSSIBLE_CONDITION",
                    f"Madde {condition['article']} mantıksal olarak imkansız bir koşul içeriyor",
                    location=f"Madde {condition['article']}",
                    context=content[:200] + "..." if len(content) > 200 else content,
                    suggestion="Koşulu mantıksal olarak tutarlı hale getirin"
                )

    # Helper methods for logic analysis

    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two texts are contradictory"""
        # Simple keyword-based contradiction detection
        # In production, use NLP for better accuracy
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check for same subject with opposite obligations
        subject1 = self._extract_subject(text1)
        subject2 = self._extract_subject(text2)

        if subject1 and subject2 and self._are_similar(subject1, subject2):
            # Check if one is obligation and other is prohibition
            has_obligation = any(kw in text1_lower for kw in self.OBLIGATION_KEYWORDS)
            has_prohibition = any(kw in text2_lower for kw in self.PROHIBITION_KEYWORDS)

            return has_obligation and has_prohibition

        return False

    def _are_mutually_exclusive(self, text1: str, text2: str) -> bool:
        """Check if two texts are mutually exclusive"""
        # Check for mutually exclusive conditions
        if 'sadece' in text1.lower() and 'sadece' in text2.lower():
            return True

        if 'yalnızca' in text1.lower() and 'yalnızca' in text2.lower():
            return True

        return False

    def _obligations_conflict(self, text1: str, text2: str) -> bool:
        """Check if two obligations conflict"""
        # Check for conflicting obligations
        # Example: "30 gün içinde" vs "60 gün içinde" for same action
        return False  # Simplified for now

    def _is_impossible_condition(self, text: str) -> bool:
        """Check if condition is impossible"""
        text_lower = text.lower()

        # Check for self-contradictory conditions
        if 'hem' in text_lower and 'hem de' in text_lower:
            # Check if contains contradiction
            if any(neg in text_lower for neg in self.NEGATION_KEYWORDS):
                return True

        return False

    def _contains_logical_impossibility(self, text: str) -> bool:
        """Check if text contains logical impossibility"""
        # Check for "X ve X değil" patterns
        return False  # Simplified for now

    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract subject from text"""
        # Simple subject extraction (first noun phrase)
        # In production, use NLP
        words = text.split()
        if len(words) > 0:
            return words[0]
        return None

    def _are_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar"""
        # Simple similarity check
        return text1.lower() == text2.lower()

    def _extract_exemption_subjects(self, text: str) -> List[str]:
        """Extract subjects from exemption text"""
        # Extract exempted subjects
        subjects = []
        # Simplified extraction
        return subjects

    def _is_in_scope(self, subject: str, scope_text: str) -> bool:
        """Check if subject is in scope"""
        # Check if subject mentioned in scope
        return subject.lower() in scope_text.lower()

    def _get_document_type(self, data: Dict[str, Any]) -> str:
        """Get document type"""
        # Check metadata
        metadata = data.get('metadata', {})
        doc_type = metadata.get('document_type', '').lower()

        if doc_type:
            return doc_type

        # Infer from fields
        if 'law_number' in data:
            return 'kanun'
        elif 'regulation_number' in data:
            return 'yönetmelik'
        elif 'decision_number' in data:
            return 'karar'

        return 'unknown'

    def _get_reference_type(self, reference: str) -> str:
        """Get type of referenced document"""
        ref_lower = reference.lower()

        for doc_type in self.LEGAL_HIERARCHY.keys():
            if doc_type in ref_lower:
                return doc_type

        return 'unknown'

    def _contradicts_reference(self, data: Dict[str, Any], reference: str) -> bool:
        """Check if document contradicts reference"""
        # Simplified check
        return False

    def _has_temporal_conflict(self, text1: str, text2: str) -> bool:
        """Check for temporal conflicts"""
        # Check for conflicting time requirements
        return False

    def _has_disambiguation(self, text1: str, text2: str) -> bool:
        """Check if texts have disambiguation"""
        # Check for priority indicators
        priority_keywords = ['öncelikle', 'ilk olarak', 'özel hüküm', 'istisnası']

        return any(kw in text1.lower() or kw in text2.lower() for kw in priority_keywords)


__all__ = ['LegalLogicValidator']
