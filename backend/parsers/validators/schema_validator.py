"""Schema Validator - Harvey/Legora CTO-Level Production-Grade
Validates Turkish legal document schemas and structure

Production Features:
- Document structure validation (Law, Regulation, Decision)
- Required field validation
- Field type validation
- Turkish legal document schema
- Metadata validation
- Pydantic integration
- Custom validation rules
- Nested structure validation
- Collection validation
"""
from typing import Dict, List, Any, Optional, Set, Type
import logging
import time
from dataclasses import dataclass

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


class SchemaValidator(BaseValidator):
    """Schema Validator for Turkish Legal Documents

    Validates document structure and required fields:
    - Required fields presence
    - Field type correctness
    - Turkish legal document structure
    - Metadata completeness
    - Collection constraints

    Features:
    - Multiple document types (Law, Regulation, Decision)
    - Flexible schema definition
    - Type checking
    - Nested validation
    - Turkish legal conventions
    """

    # Law schema
    LAW_SCHEMA = {
        'required_fields': [
            'law_number',  # Kanun numarası (5237, 6698, etc.)
            'title',  # Başlık
            'publication_date',  # Yayım tarihi
            'articles',  # Maddeler
        ],
        'optional_fields': [
            'preamble',  # Önsöz
            'enactment_formula',  # Kanunlaşma formülü
            'temporary_articles',  # Geçici maddeler
            'additional_articles',  # Ek maddeler
            'effectivity_date',  # Yürürlük tarihi
            'signature',  # İmza
            'metadata',  # Metadata
        ],
        'field_types': {
            'law_number': (str, int),
            'title': str,
            'publication_date': str,
            'articles': (list, dict),
            'preamble': str,
            'effectivity_date': str,
            'metadata': dict,
        },
        'nested_schemas': {
            'articles': {
                'required_fields': ['number', 'content'],
                'optional_fields': ['title', 'paragraphs', 'amendments'],
                'field_types': {
                    'number': (str, int),
                    'content': str,
                    'title': str,
                    'paragraphs': list,
                }
            }
        }
    }

    # Regulation schema
    REGULATION_SCHEMA = {
        'required_fields': [
            'regulation_number',  # Yönetmelik numarası
            'title',  # Başlık
            'authority',  # Yayımlayan kurum
            'publication_date',  # Yayım tarihi
            'articles',  # Maddeler
        ],
        'optional_fields': [
            'purpose',  # Amaç
            'scope',  # Kapsam
            'definitions',  # Tanımlar
            'references',  # Atıflar
            'temporary_articles',  # Geçici maddeler
            'metadata',  # Metadata
        ],
        'field_types': {
            'regulation_number': (str, int),
            'title': str,
            'authority': str,
            'publication_date': str,
            'articles': (list, dict),
            'definitions': dict,
            'metadata': dict,
        }
    }

    # Decision schema
    DECISION_SCHEMA = {
        'required_fields': [
            'decision_number',  # Karar numarası
            'court',  # Mahkeme
            'date',  # Karar tarihi
            'subject',  # Konu
            'decision_text',  # Karar metni
        ],
        'optional_fields': [
            'case_number',  # Dava numarası
            'parties',  # Taraflar
            'judge',  # Hakim
            'verdict',  # Hüküm
            'reasoning',  # Gerekçe
            'references',  # Atıflar
            'metadata',  # Metadata
        ],
        'field_types': {
            'decision_number': (str, int),
            'court': str,
            'date': str,
            'subject': str,
            'decision_text': str,
            'parties': list,
            'references': list,
            'metadata': dict,
        }
    }

    def __init__(self):
        """Initialize Schema Validator"""
        super().__init__(name="Schema Validator")

        # Schema registry
        self.schemas = {
            'law': self.LAW_SCHEMA,
            'regulation': self.REGULATION_SCHEMA,
            'decision': self.DECISION_SCHEMA,
        }

    def validate(self, data: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate document schema

        Args:
            data: Document data dictionary
            **kwargs: Options
                - schema_type: 'law', 'regulation', or 'decision'
                - strict: Fail on warnings (default: False)

        Returns:
            ValidationResult with schema validation issues
        """
        start_time = time.time()
        result = self.create_result()

        # Determine schema type
        schema_type = kwargs.get('schema_type')
        if not schema_type:
            schema_type = self._detect_schema_type(data)

        if not schema_type:
            self.add_error(
                result,
                "UNKNOWN_SCHEMA",
                "Cannot determine document schema type",
                suggestion="Specify schema_type parameter or include type indicators in data"
            )
            return self.finalize_result(result, start_time)

        logger.info(f"Validating {schema_type} schema")

        # Get schema
        schema = self.schemas.get(schema_type)
        if not schema:
            self.add_error(
                result,
                "INVALID_SCHEMA_TYPE",
                f"Unknown schema type: {schema_type}",
                suggestion=f"Valid types: {list(self.schemas.keys())}"
            )
            return self.finalize_result(result, start_time)

        # Validate required fields
        self._validate_required_fields(data, schema, result)

        # Validate field types
        self._validate_field_types(data, schema, result)

        # Validate nested structures
        if 'nested_schemas' in schema:
            self._validate_nested_structures(data, schema['nested_schemas'], result)

        # Validate Turkish legal conventions
        self._validate_turkish_conventions(data, schema_type, result)

        return self.finalize_result(result, start_time)

    def _detect_schema_type(self, data: Dict[str, Any]) -> Optional[str]:
        """Detect schema type from data

        Args:
            data: Document data

        Returns:
            Schema type or None
        """
        # Check for law indicators
        if 'law_number' in data or 'kanun_numarası' in data:
            return 'law'

        # Check for regulation indicators
        if 'regulation_number' in data or 'yönetmelik_numarası' in data or 'authority' in data:
            return 'regulation'

        # Check for decision indicators
        if 'decision_number' in data or 'karar_numarası' in data or 'court' in data:
            return 'decision'

        # Check metadata
        if 'metadata' in data:
            doc_type = data['metadata'].get('document_type', '').lower()
            if 'kanun' in doc_type or 'law' in doc_type:
                return 'law'
            elif 'yönetmelik' in doc_type or 'regulation' in doc_type:
                return 'regulation'
            elif 'karar' in doc_type or 'decision' in doc_type:
                return 'decision'

        return None

    def _validate_required_fields(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate required fields presence"""
        required_fields = schema.get('required_fields', [])

        for field in required_fields:
            passed = field in data and data[field] is not None

            self.update_check_stats(result, passed)

            if not passed:
                self.add_error(
                    result,
                    "MISSING_REQUIRED_FIELD",
                    f"Required field '{field}' is missing or None",
                    suggestion=f"Add '{field}' to document data"
                )

                # Check for Turkish field name variants
                turkish_variant = self._get_turkish_variant(field)
                if turkish_variant and turkish_variant in data:
                    self.add_info(
                        result,
                        "TURKISH_VARIANT_FOUND",
                        f"Found Turkish variant '{turkish_variant}' for '{field}'",
                        suggestion="Consider standardizing field names"
                    )

    def _validate_field_types(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate field types"""
        field_types = schema.get('field_types', {})

        for field, expected_type in field_types.items():
            if field not in data:
                continue  # Skip missing fields (handled by required check)

            value = data[field]
            if value is None:
                continue  # Skip None values

            # Handle multiple allowed types
            if isinstance(expected_type, tuple):
                passed = isinstance(value, expected_type)
                type_name = ' or '.join(t.__name__ for t in expected_type)
            else:
                passed = isinstance(value, expected_type)
                type_name = expected_type.__name__

            self.update_check_stats(result, passed)

            if not passed:
                actual_type = type(value).__name__
                self.add_error(
                    result,
                    "INVALID_FIELD_TYPE",
                    f"Field '{field}' has type {actual_type}, expected {type_name}",
                    location=field,
                    suggestion=f"Convert '{field}' to {type_name}"
                )

    def _validate_nested_structures(
        self,
        data: Dict[str, Any],
        nested_schemas: Dict[str, Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate nested structures (e.g., articles)"""
        for field, nested_schema in nested_schemas.items():
            if field not in data:
                continue

            value = data[field]

            # Handle list of items
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if not isinstance(item, dict):
                        self.add_warning(
                            result,
                            "INVALID_NESTED_ITEM",
                            f"{field}[{i}] is not a dictionary",
                            location=f"{field}[{i}]"
                        )
                        continue

                    # Validate nested item
                    self._validate_nested_item(item, nested_schema, result, f"{field}[{i}]")

            # Handle dictionary
            elif isinstance(value, dict):
                self._validate_nested_item(value, nested_schema, result, field)

    def _validate_nested_item(
        self,
        item: Dict[str, Any],
        schema: Dict[str, Any],
        result: ValidationResult,
        location: str
    ) -> None:
        """Validate a single nested item"""
        # Check required fields
        required_fields = schema.get('required_fields', [])
        for field in required_fields:
            passed = field in item and item[field] is not None

            self.update_check_stats(result, passed)

            if not passed:
                self.add_error(
                    result,
                    "MISSING_NESTED_FIELD",
                    f"Required field '{field}' missing in {location}",
                    location=location,
                    suggestion=f"Add '{field}' to {location}"
                )

        # Check field types
        field_types = schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field not in item or item[field] is None:
                continue

            value = item[field]

            if isinstance(expected_type, tuple):
                passed = isinstance(value, expected_type)
            else:
                passed = isinstance(value, expected_type)

            self.update_check_stats(result, passed)

            if not passed:
                actual_type = type(value).__name__
                expected_name = expected_type.__name__ if not isinstance(expected_type, tuple) else str(expected_type)
                self.add_warning(
                    result,
                    "INVALID_NESTED_FIELD_TYPE",
                    f"Field '{field}' in {location} has type {actual_type}, expected {expected_name}",
                    location=f"{location}.{field}"
                )

    def _validate_turkish_conventions(
        self,
        data: Dict[str, Any],
        schema_type: str,
        result: ValidationResult
    ) -> None:
        """Validate Turkish legal document conventions"""

        # Law-specific conventions
        if schema_type == 'law':
            self._validate_law_conventions(data, result)

        # Regulation-specific conventions
        elif schema_type == 'regulation':
            self._validate_regulation_conventions(data, result)

        # Decision-specific conventions
        elif schema_type == 'decision':
            self._validate_decision_conventions(data, result)

    def _validate_law_conventions(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate law-specific conventions"""

        # Check law number format
        if 'law_number' in data:
            law_num = str(data['law_number'])
            # Turkish law numbers are typically 4 digits
            if not law_num.isdigit() or len(law_num) != 4:
                self.add_warning(
                    result,
                    "UNUSUAL_LAW_NUMBER",
                    f"Law number '{law_num}' doesn't follow typical 4-digit format",
                    location="law_number",
                    suggestion="Verify law number is correct"
                )

        # Check for articles
        if 'articles' in data:
            articles = data['articles']
            if isinstance(articles, list) and len(articles) == 0:
                self.add_error(
                    result,
                    "NO_ARTICLES",
                    "Law has no articles",
                    location="articles",
                    suggestion="Add at least one article"
                )

    def _validate_regulation_conventions(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate regulation-specific conventions"""

        # Check authority
        if 'authority' in data:
            authority = data['authority']
            # Common Turkish authorities
            known_authorities = [
                'Cumhurbaşkanlığı', 'Bakanlık', 'Kurul', 'Başkanlık',
                'BDDK', 'SPK', 'EPDK', 'BTK', 'RTÜK', 'KVKK', 'REKABET'
            ]

            if not any(auth in authority for auth in known_authorities):
                self.add_info(
                    result,
                    "UNKNOWN_AUTHORITY",
                    f"Authority '{authority}' is not a known Turkish regulatory body",
                    location="authority"
                )

    def _validate_decision_conventions(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate decision-specific conventions"""

        # Check court name
        if 'court' in data:
            court = data['court']
            # Common Turkish courts
            known_courts = [
                'Yargıtay', 'Danıştay', 'Anayasa Mahkemesi', 'AYM',
                'Bölge Adliye Mahkemesi', 'Asliye', 'Sulh', 'İcra'
            ]

            if not any(c in court for c in known_courts):
                self.add_info(
                    result,
                    "UNKNOWN_COURT",
                    f"Court '{court}' is not a known Turkish court",
                    location="court"
                )

    def _get_turkish_variant(self, english_field: str) -> Optional[str]:
        """Get Turkish variant of English field name"""
        variants = {
            'law_number': 'kanun_numarası',
            'title': 'başlık',
            'publication_date': 'yayım_tarihi',
            'articles': 'maddeler',
            'regulation_number': 'yönetmelik_numarası',
            'authority': 'kurum',
            'decision_number': 'karar_numarası',
            'court': 'mahkeme',
            'date': 'tarih',
        }
        return variants.get(english_field)

    def register_schema(
        self,
        schema_type: str,
        schema: Dict[str, Any]
    ) -> None:
        """Register a custom schema

        Args:
            schema_type: Schema type identifier
            schema: Schema definition
        """
        self.schemas[schema_type] = schema
        logger.info(f"Registered custom schema: {schema_type}")

    def get_schema(self, schema_type: str) -> Optional[Dict[str, Any]]:
        """Get schema by type

        Args:
            schema_type: Schema type

        Returns:
            Schema definition or None
        """
        return self.schemas.get(schema_type)


__all__ = ['SchemaValidator']
