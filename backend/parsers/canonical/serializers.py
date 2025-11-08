"""Canonical Serializers - Harvey/Legora CTO-Level Production-Grade
Multi-format serialization for canonical Turkish legal documents

Production Features:
- JSON serialization with custom encoders
- XML serialization with Turkish legal structure
- YAML serialization for human-readable output
- CSV serialization for tabular data export
- Custom field filtering and exclusion
- Pretty printing and formatting options
- Enum value handling (use_enum_values)
- Date/datetime serialization
- Null/None field handling
- Nested structure preservation
- Character encoding (UTF-8 for Turkish)
- Validation before serialization
- Error handling and recovery
"""
from typing import Dict, List, Any, Optional, Set
from datetime import date, datetime
from decimal import Decimal
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import logging

from .models import (
    CanonicalLegalDocument, Article, Clause, Section,
    Citation, DocumentRelationship, Authority, Publication,
    EnforcementInfo, ProcessingMetadata, DocumentCollection
)
from .enums import DocumentType, DocumentStatus, ClauseType

logger = logging.getLogger(__name__)


# ============================================================================
# JSON SERIALIZER
# ============================================================================

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for canonical models"""

    def default(self, obj):
        """Custom encoding for special types"""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, 'model_dump'):  # Pydantic models
            return obj.model_dump(mode='json', exclude_none=True)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__

        return super().default(obj)


class JSONSerializer:
    """Serializes canonical documents to JSON"""

    def __init__(
        self,
        pretty: bool = True,
        exclude_none: bool = True,
        use_enum_values: bool = True,
        indent: int = 2
    ):
        """Initialize JSON serializer

        Args:
            pretty: Pretty print JSON
            exclude_none: Exclude None/null fields
            use_enum_values: Use enum values instead of names
            indent: Indentation spaces
        """
        self.pretty = pretty
        self.exclude_none = exclude_none
        self.use_enum_values = use_enum_values
        self.indent = indent if pretty else None

        logger.debug(f"Initialized JSONSerializer (pretty={pretty}, exclude_none={exclude_none})")

    def serialize(
        self,
        document: CanonicalLegalDocument,
        fields: Optional[List[str]] = None
    ) -> str:
        """Serialize document to JSON

        Args:
            document: Document to serialize
            fields: Optional list of fields to include (None = all)

        Returns:
            JSON string
        """
        try:
            # Get dict representation
            data = document.model_dump(
                mode='json',
                exclude_none=self.exclude_none
            )

            # Filter fields if specified
            if fields:
                data = {k: v for k, v in data.items() if k in fields}

            # Serialize to JSON
            json_str = json.dumps(
                data,
                cls=CustomJSONEncoder,
                ensure_ascii=False,  # Preserve Turkish characters
                indent=self.indent
            )

            logger.debug(f"Serialized document {document.document_id} to JSON ({len(json_str)} bytes)")

            return json_str

        except Exception as e:
            logger.error(f"JSON serialization failed for {document.document_id}: {e}")
            raise

    def serialize_batch(
        self,
        documents: List[CanonicalLegalDocument],
        fields: Optional[List[str]] = None
    ) -> str:
        """Serialize multiple documents to JSON array

        Args:
            documents: Documents to serialize
            fields: Optional fields filter

        Returns:
            JSON array string
        """
        try:
            docs_data = []

            for doc in documents:
                data = doc.model_dump(
                    mode='json',
                    exclude_none=self.exclude_none
                )

                if fields:
                    data = {k: v for k, v in data.items() if k in fields}

                docs_data.append(data)

            json_str = json.dumps(
                docs_data,
                cls=CustomJSONEncoder,
                ensure_ascii=False,
                indent=self.indent
            )

            logger.info(f"Serialized {len(documents)} documents to JSON")

            return json_str

        except Exception as e:
            logger.error(f"Batch JSON serialization failed: {e}")
            raise

    def serialize_to_file(
        self,
        document: CanonicalLegalDocument,
        file_path: str,
        fields: Optional[List[str]] = None
    ) -> None:
        """Serialize document to JSON file

        Args:
            document: Document to serialize
            file_path: Output file path
            fields: Optional fields filter
        """
        json_str = self.serialize(document, fields)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        logger.info(f"Serialized {document.document_id} to {file_path}")


# ============================================================================
# XML SERIALIZER
# ============================================================================

class XMLSerializer:
    """Serializes canonical documents to XML"""

    def __init__(
        self,
        pretty: bool = True,
        encoding: str = 'utf-8',
        exclude_none: bool = True
    ):
        """Initialize XML serializer

        Args:
            pretty: Pretty print XML
            encoding: Character encoding
            exclude_none: Exclude None/null fields
        """
        self.pretty = pretty
        self.encoding = encoding
        self.exclude_none = exclude_none

        logger.debug(f"Initialized XMLSerializer (pretty={pretty}, encoding={encoding})")

    def serialize(
        self,
        document: CanonicalLegalDocument,
        root_tag: str = "LegalDocument"
    ) -> str:
        """Serialize document to XML

        Args:
            document: Document to serialize
            root_tag: Root element tag name

        Returns:
            XML string
        """
        try:
            # Create root element
            root = ET.Element(root_tag)

            # Add document data
            self._add_document_to_element(document, root)

            # Convert to string
            xml_str = ET.tostring(root, encoding=self.encoding)

            # Pretty print if requested
            if self.pretty:
                dom = minidom.parseString(xml_str)
                xml_str = dom.toprettyxml(indent="  ", encoding=self.encoding)

            # Decode to string if bytes
            if isinstance(xml_str, bytes):
                xml_str = xml_str.decode(self.encoding)

            logger.debug(f"Serialized document {document.document_id} to XML")

            return xml_str

        except Exception as e:
            logger.error(f"XML serialization failed for {document.document_id}: {e}")
            raise

    def _add_document_to_element(
        self,
        document: CanonicalLegalDocument,
        parent: ET.Element
    ) -> None:
        """Add document data to XML element

        Args:
            document: Document
            parent: Parent XML element
        """
        # Basic fields
        self._add_field(parent, "document_id", document.document_id)
        self._add_field(parent, "document_type", document.document_type.value if hasattr(document.document_type, 'value') else str(document.document_type))
        self._add_field(parent, "document_status", document.document_status.value if hasattr(document.document_status, 'value') else str(document.document_status))

        # Numbers
        if document.law_number:
            self._add_field(parent, "law_number", document.law_number)
        if document.regulation_number:
            self._add_field(parent, "regulation_number", document.regulation_number)

        # Content
        self._add_field(parent, "title", document.title)
        if document.short_title:
            self._add_field(parent, "short_title", document.short_title)

        # Full text (as CDATA to preserve formatting)
        full_text_elem = ET.SubElement(parent, "full_text")
        full_text_elem.text = document.full_text

        # Articles
        if document.articles:
            articles_elem = ET.SubElement(parent, "articles")
            for article in document.articles:
                self._add_article_to_element(article, articles_elem)

        # Citations
        if document.citations:
            citations_elem = ET.SubElement(parent, "citations")
            for citation in document.citations:
                self._add_citation_to_element(citation, citations_elem)

        # Metadata
        if document.legal_domains:
            domains_elem = ET.SubElement(parent, "legal_domains")
            for domain in document.legal_domains:
                domain_elem = ET.SubElement(domains_elem, "domain")
                domain_elem.text = domain.value if hasattr(domain, 'value') else str(domain)

        if document.keywords:
            keywords_elem = ET.SubElement(parent, "keywords")
            for keyword in document.keywords:
                kw_elem = ET.SubElement(keywords_elem, "keyword")
                kw_elem.text = keyword

    def _add_article_to_element(self, article: Article, parent: ET.Element) -> None:
        """Add article to XML element

        Args:
            article: Article
            parent: Parent element
        """
        article_elem = ET.SubElement(parent, "article")

        self._add_field(article_elem, "article_id", article.article_id)
        self._add_field(article_elem, "article_number", article.article_number)

        if article.title:
            self._add_field(article_elem, "title", article.title)

        content_elem = ET.SubElement(article_elem, "content")
        content_elem.text = article.content

        # Status flags
        self._add_field(article_elem, "is_active", str(article.is_active).lower())
        self._add_field(article_elem, "is_repealed", str(article.is_repealed).lower())

        if article.is_temporary:
            self._add_field(article_elem, "is_temporary", "true")
        if article.is_additional:
            self._add_field(article_elem, "is_additional", "true")

        # Amendment info
        if article.amendment_type:
            self._add_field(article_elem, "amendment_type", article.amendment_type.value if hasattr(article.amendment_type, 'value') else str(article.amendment_type))
        if article.amended_by:
            self._add_field(article_elem, "amended_by", article.amended_by)

    def _add_citation_to_element(self, citation: Citation, parent: ET.Element) -> None:
        """Add citation to XML element

        Args:
            citation: Citation
            parent: Parent element
        """
        citation_elem = ET.SubElement(parent, "citation")

        self._add_field(citation_elem, "citation_id", citation.citation_id)
        self._add_field(citation_elem, "citation_type", citation.citation_type.value if hasattr(citation.citation_type, 'value') else str(citation.citation_type))
        self._add_field(citation_elem, "citation_text", citation.citation_text)

        if citation.target_law_number:
            self._add_field(citation_elem, "target_law_number", citation.target_law_number)
        if citation.target_article:
            self._add_field(citation_elem, "target_article", citation.target_article)

        self._add_field(citation_elem, "confidence", str(citation.confidence))

    def _add_field(self, parent: ET.Element, name: str, value: Any) -> None:
        """Add field to XML element

        Args:
            parent: Parent element
            name: Field name
            value: Field value
        """
        if value is None and self.exclude_none:
            return

        elem = ET.SubElement(parent, name)

        if value is not None:
            if isinstance(value, (date, datetime)):
                elem.text = value.isoformat()
            else:
                elem.text = str(value)

    def serialize_to_file(
        self,
        document: CanonicalLegalDocument,
        file_path: str,
        root_tag: str = "LegalDocument"
    ) -> None:
        """Serialize document to XML file

        Args:
            document: Document
            file_path: Output file path
            root_tag: Root tag name
        """
        xml_str = self.serialize(document, root_tag)

        with open(file_path, 'w', encoding=self.encoding) as f:
            f.write(xml_str)

        logger.info(f"Serialized {document.document_id} to {file_path}")


# ============================================================================
# YAML SERIALIZER
# ============================================================================

class YAMLSerializer:
    """Serializes canonical documents to YAML (human-readable)"""

    def __init__(
        self,
        exclude_none: bool = True,
        default_flow_style: bool = False
    ):
        """Initialize YAML serializer

        Args:
            exclude_none: Exclude None/null fields
            default_flow_style: Use flow style (compact)
        """
        self.exclude_none = exclude_none
        self.default_flow_style = default_flow_style

        # Try to import PyYAML
        try:
            import yaml
            self.yaml = yaml
            self.available = True
            logger.debug("Initialized YAMLSerializer")
        except ImportError:
            self.yaml = None
            self.available = False
            logger.warning("PyYAML not available - YAML serialization disabled")

    def serialize(
        self,
        document: CanonicalLegalDocument,
        fields: Optional[List[str]] = None
    ) -> str:
        """Serialize document to YAML

        Args:
            document: Document to serialize
            fields: Optional fields filter

        Returns:
            YAML string
        """
        if not self.available:
            raise ImportError("PyYAML is required for YAML serialization")

        try:
            # Get dict representation
            data = document.model_dump(
                mode='json',
                exclude_none=self.exclude_none
            )

            # Filter fields if specified
            if fields:
                data = {k: v for k, v in data.items() if k in fields}

            # Serialize to YAML
            yaml_str = self.yaml.dump(
                data,
                allow_unicode=True,
                default_flow_style=self.default_flow_style,
                sort_keys=False
            )

            logger.debug(f"Serialized document {document.document_id} to YAML")

            return yaml_str

        except Exception as e:
            logger.error(f"YAML serialization failed for {document.document_id}: {e}")
            raise

    def serialize_to_file(
        self,
        document: CanonicalLegalDocument,
        file_path: str,
        fields: Optional[List[str]] = None
    ) -> None:
        """Serialize document to YAML file

        Args:
            document: Document
            file_path: Output file path
            fields: Optional fields filter
        """
        yaml_str = self.serialize(document, fields)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(yaml_str)

        logger.info(f"Serialized {document.document_id} to {file_path}")


# ============================================================================
# CSV SERIALIZER
# ============================================================================

class CSVSerializer:
    """Serializes canonical documents to CSV (tabular format)"""

    def __init__(
        self,
        delimiter: str = ',',
        quotechar: str = '"',
        encoding: str = 'utf-8'
    ):
        """Initialize CSV serializer

        Args:
            delimiter: Field delimiter
            quotechar: Quote character
            encoding: Character encoding
        """
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.encoding = encoding

        logger.debug(f"Initialized CSVSerializer (delimiter='{delimiter}')")

    def serialize_articles(
        self,
        document: CanonicalLegalDocument,
        include_header: bool = True
    ) -> str:
        """Serialize document articles to CSV

        Args:
            document: Document
            include_header: Include CSV header row

        Returns:
            CSV string
        """
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(
            output,
            delimiter=self.delimiter,
            quotechar=self.quotechar,
            quoting=csv.QUOTE_MINIMAL
        )

        # Header
        if include_header:
            writer.writerow([
                'document_id',
                'article_number',
                'title',
                'content',
                'is_active',
                'is_repealed',
                'is_temporary',
                'is_additional',
                'amendment_type',
                'amended_by'
            ])

        # Articles
        for article in document.articles:
            writer.writerow([
                document.document_id,
                article.article_number,
                article.title or '',
                article.content,
                article.is_active,
                article.is_repealed,
                article.is_temporary,
                article.is_additional,
                article.amendment_type.value if article.amendment_type else '',
                article.amended_by or ''
            ])

        csv_str = output.getvalue()
        output.close()

        logger.debug(f"Serialized {len(document.articles)} articles to CSV")

        return csv_str

    def serialize_citations(
        self,
        document: CanonicalLegalDocument,
        include_header: bool = True
    ) -> str:
        """Serialize document citations to CSV

        Args:
            document: Document
            include_header: Include CSV header

        Returns:
            CSV string
        """
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(
            output,
            delimiter=self.delimiter,
            quotechar=self.quotechar,
            quoting=csv.QUOTE_MINIMAL
        )

        # Header
        if include_header:
            writer.writerow([
                'citation_id',
                'citation_type',
                'source_document_id',
                'source_article',
                'target_law_number',
                'target_article',
                'citation_text',
                'confidence',
                'is_resolved'
            ])

        # Citations
        for citation in document.citations:
            writer.writerow([
                citation.citation_id,
                citation.citation_type.value if hasattr(citation.citation_type, 'value') else str(citation.citation_type),
                citation.source_document_id,
                citation.source_article or '',
                citation.target_law_number or '',
                citation.target_article or '',
                citation.citation_text,
                citation.confidence,
                citation.is_resolved
            ])

        csv_str = output.getvalue()
        output.close()

        logger.debug(f"Serialized {len(document.citations)} citations to CSV")

        return csv_str

    def serialize_to_file(
        self,
        document: CanonicalLegalDocument,
        file_path: str,
        content_type: str = 'articles',
        include_header: bool = True
    ) -> None:
        """Serialize document to CSV file

        Args:
            document: Document
            file_path: Output file path
            content_type: 'articles' or 'citations'
            include_header: Include header row
        """
        if content_type == 'articles':
            csv_str = self.serialize_articles(document, include_header)
        elif content_type == 'citations':
            csv_str = self.serialize_citations(document, include_header)
        else:
            raise ValueError(f"Unknown content_type: {content_type}")

        with open(file_path, 'w', encoding=self.encoding) as f:
            f.write(csv_str)

        logger.info(f"Serialized {document.document_id} {content_type} to {file_path}")


# ============================================================================
# SERIALIZER FACTORY
# ============================================================================

class SerializerFactory:
    """Factory for creating serializers"""

    FORMATS = {
        'json': JSONSerializer,
        'xml': XMLSerializer,
        'yaml': YAMLSerializer,
        'csv': CSVSerializer
    }

    @classmethod
    def create(cls, format: str, **kwargs) -> Any:
        """Create serializer for format

        Args:
            format: Format name (json, xml, yaml, csv)
            **kwargs: Serializer-specific options

        Returns:
            Serializer instance
        """
        format_lower = format.lower()

        if format_lower not in cls.FORMATS:
            raise ValueError(f"Unknown format: {format}. Available: {list(cls.FORMATS.keys())}")

        serializer_class = cls.FORMATS[format_lower]
        return serializer_class(**kwargs)

    @classmethod
    def serialize(
        cls,
        document: CanonicalLegalDocument,
        format: str,
        **kwargs
    ) -> str:
        """Serialize document to format

        Args:
            document: Document to serialize
            format: Format name
            **kwargs: Serializer options

        Returns:
            Serialized string
        """
        serializer = cls.create(format, **kwargs)
        return serializer.serialize(document)


__all__ = [
    'JSONSerializer',
    'XMLSerializer',
    'YAMLSerializer',
    'CSVSerializer',
    'SerializerFactory',
    'CustomJSONEncoder'
]
