"""Canonical JSON-LD Export - Harvey/Legora CTO-Level Production-Grade
JSON-LD export for Turkish legal documents with semantic web interoperability

Production Features:
- Schema.org Legislation vocabulary compliance
- Custom Turkish legal ontology (@context)
- Rich semantic annotations
- Citation and relationship linking
- Amendment history semantics
- Temporal versioning support
- Legal domain classification
- Turkish language support (@language: tr)
- Linked data principles
- W3C JSON-LD 1.1 compliance
- Graph serialization
- Context management
- IRI/URI generation for documents and articles
"""
from typing import Dict, List, Any, Optional
from datetime import date, datetime
import json
import logging

from .models import (
    CanonicalLegalDocument, Article, Clause, Section,
    Citation, DocumentRelationship
)
from .enums import (
    DocumentType, DocumentStatus, AmendmentType,
    CitationType, RelationshipType, LegalDomain
)

logger = logging.getLogger(__name__)


# ============================================================================
# JSON-LD CONTEXT
# ============================================================================

# Standard JSON-LD context with Schema.org + custom Turkish legal ontology
JSONLD_CONTEXT = {
    "@vocab": "http://schema.org/",
    "tr-law": "http://mevzuat.gov.tr/ontology/",
    "dcterms": "http://purl.org/dc/terms/",
    "foaf": "http://xmlns.com/foaf/0.1/",

    # Document properties
    "LegalDocument": "tr-law:LegalDocument",
    "Legislation": "Legislation",
    "Article": "tr-law:Article",
    "Citation": "tr-law:Citation",

    # Turkish legal types
    "Kanun": "tr-law:Kanun",
    "Yonetmelik": "tr-law:Yonetmelik",
    "Karar": "tr-law:Karar",
    "Teblig": "tr-law:Teblig",

    # Properties
    "lawNumber": "tr-law:lawNumber",
    "articleNumber": "tr-law:articleNumber",
    "isRepealed": "tr-law:isRepealed",
    "isMulga": "tr-law:isMulga",
    "amendmentType": "tr-law:amendmentType",
    "amendedBy": "tr-law:amendedBy",
    "cites": "tr-law:cites",
    "citedBy": "tr-law:citedBy",
    "amends": "tr-law:amends",
    "amendedBy": "tr-law:amendedBy",
    "resmiGazeteTarih": "tr-law:resmiGazeteTarih",
    "resmiGazeteSayi": "tr-law:resmiGazeteSayi",

    # Standard properties
    "identifier": "dcterms:identifier",
    "title": "dcterms:title",
    "description": "dcterms:description",
    "datePublished": "datePublished",
    "dateModified": "dateModified",
    "inLanguage": "inLanguage",
    "keywords": "keywords",
    "legislationDate": "legislationDate",
    "legislationIdentifier": "legislationIdentifier",
    "legislationType": "legislationType"
}


# ============================================================================
# JSON-LD EXPORTER
# ============================================================================

class JSONLDExporter:
    """Exports canonical documents to JSON-LD format"""

    def __init__(
        self,
        base_uri: str = "http://mevzuat.gov.tr",
        include_context: bool = True,
        compact: bool = False
    ):
        """Initialize JSON-LD exporter

        Args:
            base_uri: Base URI for document IRIs
            include_context: Include @context in output
            compact: Compact output (no pretty printing)
        """
        self.base_uri = base_uri.rstrip('/')
        self.include_context = include_context
        self.compact = compact

        logger.debug(f"Initialized JSONLDExporter (base_uri={base_uri})")

    def export(
        self,
        document: CanonicalLegalDocument,
        graph_mode: bool = False
    ) -> str:
        """Export document to JSON-LD

        Args:
            document: Document to export
            graph_mode: Use @graph for multiple entities

        Returns:
            JSON-LD string
        """
        try:
            # Create JSON-LD document
            jsonld = self._create_jsonld(document, graph_mode)

            # Serialize
            if self.compact:
                json_str = json.dumps(jsonld, ensure_ascii=False)
            else:
                json_str = json.dumps(jsonld, ensure_ascii=False, indent=2)

            logger.debug(f"Exported document {document.document_id} to JSON-LD")

            return json_str

        except Exception as e:
            logger.error(f"JSON-LD export failed for {document.document_id}: {e}")
            raise

    def _create_jsonld(
        self,
        document: CanonicalLegalDocument,
        graph_mode: bool
    ) -> Dict[str, Any]:
        """Create JSON-LD structure

        Args:
            document: Document
            graph_mode: Use @graph

        Returns:
            JSON-LD dict
        """
        if graph_mode:
            # Graph mode: separate entities in @graph array
            return self._create_graph_jsonld(document)
        else:
            # Standard mode: main document with embedded entities
            return self._create_document_jsonld(document)

    def _create_document_jsonld(
        self,
        document: CanonicalLegalDocument
    ) -> Dict[str, Any]:
        """Create standard JSON-LD for document

        Args:
            document: Document

        Returns:
            JSON-LD dict
        """
        jsonld = {}

        # Context
        if self.include_context:
            jsonld["@context"] = JSONLD_CONTEXT

        # Type
        jsonld["@type"] = self._get_jsonld_type(document.document_type)

        # ID (IRI)
        jsonld["@id"] = self._generate_document_iri(document)

        # Basic properties
        jsonld["identifier"] = document.document_id

        if document.law_number:
            jsonld["lawNumber"] = document.law_number
            jsonld["legislationIdentifier"] = document.law_number

        jsonld["name"] = document.title
        if document.short_title:
            jsonld["alternateName"] = document.short_title

        jsonld["text"] = document.full_text

        # Language
        jsonld["inLanguage"] = "tr"  # Turkish

        # Dates
        if document.publication:
            jsonld["legislationDate"] = document.publication.publication_date.isoformat()
            jsonld["datePublished"] = document.publication.publication_date.isoformat()

            if document.publication.resmi_gazete_tarih:
                jsonld["resmiGazeteTarih"] = document.publication.resmi_gazete_tarih.isoformat()
            if document.publication.resmi_gazete_sayi:
                jsonld["resmiGazeteSayi"] = document.publication.resmi_gazete_sayi

        if document.enforcement and document.enforcement.effective_date:
            jsonld["legislationDateVersion"] = document.enforcement.effective_date.isoformat()

        jsonld["dateModified"] = document.updated_at.isoformat()

        # Legal domains (keywords)
        if document.legal_domains:
            jsonld["keywords"] = [
                domain.value if hasattr(domain, 'value') else str(domain)
                for domain in document.legal_domains
            ]

        if document.keywords:
            existing_keywords = jsonld.get("keywords", [])
            jsonld["keywords"] = existing_keywords + document.keywords

        # Articles
        if document.articles:
            jsonld["hasPart"] = [
                self._create_article_jsonld(article, document)
                for article in document.articles
            ]

        # Citations
        if document.citations:
            jsonld["citation"] = [
                self._create_citation_jsonld(citation)
                for citation in document.citations
            ]

        # Relationships
        if document.amends_document_ids:
            jsonld["amends"] = [
                {"@id": self._generate_document_iri_from_id(doc_id)}
                for doc_id in document.amends_document_ids
            ]

        if document.amended_by_document_ids:
            jsonld["amendedBy"] = [
                {"@id": self._generate_document_iri_from_id(doc_id)}
                for doc_id in document.amended_by_document_ids
            ]

        # Version info
        if document.version:
            jsonld["version"] = document.version

        if document.is_consolidated:
            jsonld["isConsolidated"] = True

        return jsonld

    def _create_graph_jsonld(
        self,
        document: CanonicalLegalDocument
    ) -> Dict[str, Any]:
        """Create graph-based JSON-LD

        Args:
            document: Document

        Returns:
            JSON-LD dict with @graph
        """
        jsonld = {}

        # Context
        if self.include_context:
            jsonld["@context"] = JSONLD_CONTEXT

        # Graph array
        graph = []

        # Main document entity
        doc_entity = self._create_document_jsonld(document)
        # Remove @context from embedded entity
        if "@context" in doc_entity:
            del doc_entity["@context"]
        graph.append(doc_entity)

        # Article entities
        for article in document.articles:
            article_entity = self._create_article_jsonld(article, document)
            graph.append(article_entity)

        # Citation entities
        for citation in document.citations:
            citation_entity = self._create_citation_jsonld(citation)
            # Only add if it has substantial data
            if len(citation_entity) > 2:  # More than just @type and @id
                graph.append(citation_entity)

        jsonld["@graph"] = graph

        return jsonld

    def _create_article_jsonld(
        self,
        article: Article,
        document: CanonicalLegalDocument
    ) -> Dict[str, Any]:
        """Create JSON-LD for article

        Args:
            article: Article
            document: Parent document

        Returns:
            JSON-LD dict
        """
        article_ld = {
            "@type": "Article",
            "@id": self._generate_article_iri(document, article),
            "identifier": article.article_id,
            "articleNumber": article.article_number,
            "name": article.title or f"Madde {article.article_number}",
            "text": article.content,
            "position": article.position
        }

        # Status
        article_ld["isActive"] = article.is_active
        article_ld["isRepealed"] = article.is_repealed

        if article.is_repealed:
            article_ld["isMulga"] = True

        if article.is_temporary:
            article_ld["isTemporary"] = True

        if article.is_additional:
            article_ld["isAdditional"] = True

        # Amendment info
        if article.amendment_type:
            article_ld["amendmentType"] = article.amendment_type.value if hasattr(article.amendment_type, 'value') else str(article.amendment_type)

        if article.amended_by:
            article_ld["amendedBy"] = article.amended_by

        if article.amendment_date:
            article_ld["amendmentDate"] = article.amendment_date.isoformat()

        # Parent document reference
        article_ld["isPartOf"] = {
            "@id": self._generate_document_iri(document)
        }

        return article_ld

    def _create_citation_jsonld(
        self,
        citation: Citation
    ) -> Dict[str, Any]:
        """Create JSON-LD for citation

        Args:
            citation: Citation

        Returns:
            JSON-LD dict
        """
        citation_ld = {
            "@type": "Citation",
            "@id": f"{self.base_uri}/citation/{citation.citation_id}",
            "identifier": citation.citation_id,
            "citationType": citation.citation_type.value if hasattr(citation.citation_type, 'value') else str(citation.citation_type),
            "text": citation.citation_text
        }

        # Source
        if citation.source_document_id:
            citation_ld["citingDocument"] = {
                "@id": self._generate_document_iri_from_id(citation.source_document_id)
            }

        if citation.source_article:
            citation_ld["citingArticle"] = citation.source_article

        # Target
        if citation.target_document_id:
            citation_ld["citedDocument"] = {
                "@id": self._generate_document_iri_from_id(citation.target_document_id)
            }

        if citation.target_law_number:
            citation_ld["citedLawNumber"] = citation.target_law_number

        if citation.target_article:
            citation_ld["citedArticle"] = citation.target_article

        # Metadata
        citation_ld["confidence"] = citation.confidence
        citation_ld["isResolved"] = citation.is_resolved

        return citation_ld

    def _get_jsonld_type(self, document_type: DocumentType) -> str:
        """Get JSON-LD @type for document type

        Args:
            document_type: Document type enum

        Returns:
            JSON-LD type string
        """
        # Map Turkish document types to JSON-LD types
        type_map = {
            DocumentType.KANUN: "Legislation",
            DocumentType.YONETMELIK: "Legislation",
            DocumentType.CUMHURBASKANLIGI_KARARNAMESI: "Legislation",
            DocumentType.YARGITAY_KARARI: "LegalDecision",
            DocumentType.DANISHTAY_KARARI: "LegalDecision",
            DocumentType.ANAYASA_MAHKEMESI_KARARI: "LegalDecision",
            DocumentType.GENELGE: "PublicNotice",
            DocumentType.TEBLIG: "PublicNotice"
        }

        doc_type_value = document_type.value if hasattr(document_type, 'value') else str(document_type)

        # Try enum-based lookup
        jsonld_type = type_map.get(document_type, "LegalDocument")

        return jsonld_type

    def _generate_document_iri(self, document: CanonicalLegalDocument) -> str:
        """Generate IRI for document

        Args:
            document: Document

        Returns:
            IRI string
        """
        if document.law_number:
            return f"{self.base_uri}/kanun/{document.law_number}"
        elif document.regulation_number:
            return f"{self.base_uri}/yonetmelik/{document.regulation_number}"
        elif document.decision_number:
            return f"{self.base_uri}/karar/{document.decision_number}"
        else:
            return f"{self.base_uri}/document/{document.document_id}"

    def _generate_document_iri_from_id(self, document_id: str) -> str:
        """Generate IRI from document ID

        Args:
            document_id: Document ID

        Returns:
            IRI string
        """
        return f"{self.base_uri}/document/{document_id}"

    def _generate_article_iri(
        self,
        document: CanonicalLegalDocument,
        article: Article
    ) -> str:
        """Generate IRI for article

        Args:
            document: Parent document
            article: Article

        Returns:
            IRI string
        """
        doc_iri = self._generate_document_iri(document)
        return f"{doc_iri}/madde/{article.article_number}"

    def export_to_file(
        self,
        document: CanonicalLegalDocument,
        file_path: str,
        graph_mode: bool = False
    ) -> None:
        """Export document to JSON-LD file

        Args:
            document: Document
            file_path: Output file path
            graph_mode: Use @graph mode
        """
        jsonld_str = self.export(document, graph_mode)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(jsonld_str)

        logger.info(f"Exported {document.document_id} to JSON-LD: {file_path}")

    def export_batch(
        self,
        documents: List[CanonicalLegalDocument],
        graph_mode: bool = True
    ) -> str:
        """Export multiple documents to JSON-LD

        Args:
            documents: Documents to export
            graph_mode: Use @graph (recommended for multiple docs)

        Returns:
            JSON-LD string
        """
        if not graph_mode:
            # Export as JSON array
            jsonld_docs = [
                self._create_document_jsonld(doc)
                for doc in documents
            ]

            if self.compact:
                return json.dumps(jsonld_docs, ensure_ascii=False)
            else:
                return json.dumps(jsonld_docs, ensure_ascii=False, indent=2)

        else:
            # Export as single document with @graph
            jsonld = {}

            if self.include_context:
                jsonld["@context"] = JSONLD_CONTEXT

            graph = []

            for document in documents:
                # Add document
                doc_entity = self._create_document_jsonld(document)
                if "@context" in doc_entity:
                    del doc_entity["@context"]
                graph.append(doc_entity)

                # Add articles
                for article in document.articles:
                    article_entity = self._create_article_jsonld(article, document)
                    graph.append(article_entity)

            jsonld["@graph"] = graph

            if self.compact:
                return json.dumps(jsonld, ensure_ascii=False)
            else:
                return json.dumps(jsonld, ensure_ascii=False, indent=2)


__all__ = [
    'JSONLDExporter',
    'JSONLD_CONTEXT'
]
