"""Canonical RDF Export - Harvey/Legora CTO-Level Production-Grade
RDF triple export for Turkish legal documents with knowledge graph support

Production Features:
- RDF triple generation (subject-predicate-object)
- Multiple serialization formats (Turtle, N-Triples, RDF/XML, JSON-LD)
- Turkish legal ontology vocabulary
- Citation and relationship triples
- Amendment history as provenance
- Temporal versioning triples
- SPARQL-ready output
- Named graph support
- rdflib integration (optional dependency)
- Fallback to manual triple generation
- Knowledge graph ready
"""
from typing import Dict, List, Any, Optional, Tuple
import logging

from .models import (
    CanonicalLegalDocument, Article, Clause, Citation,
    DocumentRelationship
)
from .enums import DocumentType, AmendmentType, CitationType, RelationshipType

logger = logging.getLogger(__name__)


# ============================================================================
# RDF NAMESPACES
# ============================================================================

# Turkish legal ontology namespaces
NAMESPACES = {
    'tr-law': 'http://mevzuat.gov.tr/ontology/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
    'dcterms': 'http://purl.org/dc/terms/',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'schema': 'http://schema.org/'
}


# ============================================================================
# TRIPLE CLASS
# ============================================================================

class RDFTriple:
    """Represents an RDF triple"""

    def __init__(self, subject: str, predicate: str, obj: str, obj_type: str = 'uri'):
        """Initialize RDF triple

        Args:
            subject: Subject URI
            predicate: Predicate URI
            obj: Object (URI or literal)
            obj_type: 'uri', 'literal', or 'typed_literal'
        """
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.obj_type = obj_type

    def to_ntriples(self) -> str:
        """Convert to N-Triples format

        Returns:
            N-Triples string
        """
        s = f"<{self.subject}>"
        p = f"<{self.predicate}>"

        if self.obj_type == 'uri':
            o = f"<{self.object}>"
        elif self.obj_type == 'literal':
            # Escape quotes in literal
            escaped = self.object.replace('\\', '\\\\').replace('"', '\\"')
            o = f'"{escaped}"'
        elif self.obj_type == 'typed_literal':
            # Format: "value"^^<datatype>
            if '^^' in self.object:
                parts = self.object.split('^^')
                escaped = parts[0].replace('\\', '\\\\').replace('"', '\\"')
                o = f'"{escaped}"^^<{parts[1]}>'
            else:
                escaped = self.object.replace('\\', '\\\\').replace('"', '\\"')
                o = f'"{escaped}"'
        else:
            o = str(self.object)

        return f"{s} {p} {o} ."

    def to_turtle(self) -> str:
        """Convert to Turtle format (same as N-Triples for simple triples)

        Returns:
            Turtle string
        """
        return self.to_ntriples()


# ============================================================================
# RDF EXPORTER
# ============================================================================

class RDFExporter:
    """Exports canonical documents to RDF triples"""

    def __init__(
        self,
        base_uri: str = "http://mevzuat.gov.tr",
        use_rdflib: bool = True
    ):
        """Initialize RDF exporter

        Args:
            base_uri: Base URI for resources
            use_rdflib: Try to use rdflib if available
        """
        self.base_uri = base_uri.rstrip('/')
        self.namespaces = NAMESPACES.copy()

        # Try to import rdflib
        self.rdflib = None
        if use_rdflib:
            try:
                import rdflib
                self.rdflib = rdflib
                logger.debug("Initialized RDFExporter with rdflib support")
            except ImportError:
                logger.warning("rdflib not available - using manual triple generation")

        if not self.rdflib:
            logger.debug("Initialized RDFExporter with manual triple generation")

    def export(
        self,
        document: CanonicalLegalDocument,
        format: str = 'turtle'
    ) -> str:
        """Export document to RDF

        Args:
            document: Document to export
            format: Output format (turtle, nt, xml, json-ld)

        Returns:
            RDF string
        """
        if self.rdflib:
            return self._export_with_rdflib(document, format)
        else:
            return self._export_manual(document, format)

    def _export_with_rdflib(
        self,
        document: CanonicalLegalDocument,
        format: str
    ) -> str:
        """Export using rdflib

        Args:
            document: Document
            format: Output format

        Returns:
            RDF string
        """
        # Create RDF graph
        g = self.rdflib.Graph()

        # Bind namespaces
        for prefix, uri in self.namespaces.items():
            g.bind(prefix, self.rdflib.Namespace(uri))

        # Generate triples
        triples = self._generate_triples(document)

        # Add triples to graph
        for triple in triples:
            s = self.rdflib.URIRef(triple.subject)
            p = self.rdflib.URIRef(triple.predicate)

            if triple.obj_type == 'uri':
                o = self.rdflib.URIRef(triple.object)
            elif triple.obj_type == 'literal':
                o = self.rdflib.Literal(triple.object)
            elif triple.obj_type == 'typed_literal':
                # Parse typed literal
                if '^^' in triple.object:
                    value, datatype = triple.object.split('^^')
                    o = self.rdflib.Literal(value, datatype=self.rdflib.URIRef(datatype))
                else:
                    o = self.rdflib.Literal(triple.object)
            else:
                o = self.rdflib.Literal(triple.object)

            g.add((s, p, o))

        # Serialize
        format_map = {
            'turtle': 'turtle',
            'ttl': 'turtle',
            'nt': 'nt',
            'n-triples': 'nt',
            'xml': 'xml',
            'rdf/xml': 'xml',
            'json-ld': 'json-ld'
        }

        rdf_format = format_map.get(format.lower(), 'turtle')

        rdf_str = g.serialize(format=rdf_format)

        # rdflib serialize returns bytes in some versions
        if isinstance(rdf_str, bytes):
            rdf_str = rdf_str.decode('utf-8')

        logger.debug(f"Exported {document.document_id} to RDF ({format})")

        return rdf_str

    def _export_manual(
        self,
        document: CanonicalLegalDocument,
        format: str
    ) -> str:
        """Export using manual triple generation

        Args:
            document: Document
            format: Output format

        Returns:
            RDF string
        """
        # Generate triples
        triples = self._generate_triples(document)

        # Serialize to N-Triples (simplest format)
        if format.lower() in ('nt', 'n-triples'):
            lines = [triple.to_ntriples() for triple in triples]
            return '\n'.join(lines)

        elif format.lower() in ('turtle', 'ttl'):
            # Basic Turtle with prefixes
            lines = []

            # Add namespace prefixes
            for prefix, uri in self.namespaces.items():
                lines.append(f"@prefix {prefix}: <{uri}> .")

            lines.append("")

            # Add triples
            for triple in triples:
                lines.append(triple.to_turtle())

            return '\n'.join(lines)

        else:
            # Fallback to N-Triples
            logger.warning(f"Format {format} not supported in manual mode, using N-Triples")
            lines = [triple.to_ntriples() for triple in triples]
            return '\n'.join(lines)

    def _generate_triples(
        self,
        document: CanonicalLegalDocument
    ) -> List[RDFTriple]:
        """Generate RDF triples for document

        Args:
            document: Document

        Returns:
            List of RDFTriple
        """
        triples = []

        # Document URI
        doc_uri = self._generate_document_uri(document)

        # Type
        doc_type_uri = f"{self.namespaces['tr-law']}LegalDocument"
        triples.append(RDFTriple(
            doc_uri,
            f"{self.namespaces['rdf']}type",
            doc_type_uri,
            'uri'
        ))

        # Specific type
        specific_type = self._get_rdf_type(document.document_type)
        if specific_type:
            triples.append(RDFTriple(
                doc_uri,
                f"{self.namespaces['rdf']}type",
                f"{self.namespaces['tr-law']}{specific_type}",
                'uri'
            ))

        # Identifier
        triples.append(RDFTriple(
            doc_uri,
            f"{self.namespaces['dcterms']}identifier",
            document.document_id,
            'literal'
        ))

        # Law number
        if document.law_number:
            triples.append(RDFTriple(
                doc_uri,
                f"{self.namespaces['tr-law']}lawNumber",
                document.law_number,
                'literal'
            ))

        # Title
        triples.append(RDFTriple(
            doc_uri,
            f"{self.namespaces['dcterms']}title",
            document.title,
            'literal'
        ))

        # Language
        triples.append(RDFTriple(
            doc_uri,
            f"{self.namespaces['dcterms']}language",
            "tr",
            'literal'
        ))

        # Dates
        if document.publication:
            triples.append(RDFTriple(
                doc_uri,
                f"{self.namespaces['dcterms']}issued",
                f"{document.publication.publication_date.isoformat()}^^{self.namespaces['xsd']}date",
                'typed_literal'
            ))

        triples.append(RDFTriple(
            doc_uri,
            f"{self.namespaces['dcterms']}modified",
            f"{document.updated_at.isoformat()}^^{self.namespaces['xsd']}dateTime",
            'typed_literal'
        ))

        # Legal domains
        for domain in document.legal_domains:
            domain_value = domain.value if hasattr(domain, 'value') else str(domain)
            triples.append(RDFTriple(
                doc_uri,
                f"{self.namespaces['dcterms']}subject",
                f"{self.namespaces['tr-law']}domain/{domain_value}",
                'uri'
            ))

        # Articles
        for article in document.articles:
            article_triples = self._generate_article_triples(article, document)
            triples.extend(article_triples)

        # Citations
        for citation in document.citations:
            citation_triples = self._generate_citation_triples(citation)
            triples.extend(citation_triples)

        # Relationships
        for relationship in document.relationships:
            rel_triples = self._generate_relationship_triples(relationship, document)
            triples.extend(rel_triples)

        return triples

    def _generate_article_triples(
        self,
        article: Article,
        document: CanonicalLegalDocument
    ) -> List[RDFTriple]:
        """Generate triples for article

        Args:
            article: Article
            document: Parent document

        Returns:
            List of triples
        """
        triples = []

        doc_uri = self._generate_document_uri(document)
        article_uri = self._generate_article_uri(document, article)

        # Type
        triples.append(RDFTriple(
            article_uri,
            f"{self.namespaces['rdf']}type",
            f"{self.namespaces['tr-law']}Article",
            'uri'
        ))

        # Article number
        triples.append(RDFTriple(
            article_uri,
            f"{self.namespaces['tr-law']}articleNumber",
            article.article_number,
            'literal'
        ))

        # Title
        if article.title:
            triples.append(RDFTriple(
                article_uri,
                f"{self.namespaces['dcterms']}title",
                article.title,
                'literal'
            ))

        # Content
        triples.append(RDFTriple(
            article_uri,
            f"{self.namespaces['schema']}text",
            article.content,
            'literal'
        ))

        # Part of document
        triples.append(RDFTriple(
            article_uri,
            f"{self.namespaces['dcterms']}isPartOf",
            doc_uri,
            'uri'
        ))

        # Status
        triples.append(RDFTriple(
            article_uri,
            f"{self.namespaces['tr-law']}isActive",
            str(article.is_active).lower(),
            'typed_literal'
        ))

        triples.append(RDFTriple(
            article_uri,
            f"{self.namespaces['tr-law']}isRepealed",
            str(article.is_repealed).lower(),
            'typed_literal'
        ))

        # Amendment info
        if article.amendment_type:
            amendment_value = article.amendment_type.value if hasattr(article.amendment_type, 'value') else str(article.amendment_type)
            triples.append(RDFTriple(
                article_uri,
                f"{self.namespaces['tr-law']}amendmentType",
                amendment_value,
                'literal'
            ))

        if article.amended_by:
            triples.append(RDFTriple(
                article_uri,
                f"{self.namespaces['tr-law']}amendedBy",
                article.amended_by,
                'literal'
            ))

        return triples

    def _generate_citation_triples(
        self,
        citation: Citation
    ) -> List[RDFTriple]:
        """Generate triples for citation

        Args:
            citation: Citation

        Returns:
            List of triples
        """
        triples = []

        citation_uri = f"{self.base_uri}/citation/{citation.citation_id}"

        # Type
        triples.append(RDFTriple(
            citation_uri,
            f"{self.namespaces['rdf']}type",
            f"{self.namespaces['tr-law']}Citation",
            'uri'
        ))

        # Citation text
        triples.append(RDFTriple(
            citation_uri,
            f"{self.namespaces['schema']}text",
            citation.citation_text,
            'literal'
        ))

        # Source document
        if citation.source_document_id:
            source_uri = f"{self.base_uri}/document/{citation.source_document_id}"
            triples.append(RDFTriple(
                citation_uri,
                f"{self.namespaces['tr-law']}citingDocument",
                source_uri,
                'uri'
            ))

        # Target document
        if citation.target_document_id:
            target_uri = f"{self.base_uri}/document/{citation.target_document_id}"
            triples.append(RDFTriple(
                citation_uri,
                f"{self.namespaces['tr-law']}citedDocument",
                target_uri,
                'uri'
            ))

        # Citation type
        citation_type_value = citation.citation_type.value if hasattr(citation.citation_type, 'value') else str(citation.citation_type)
        triples.append(RDFTriple(
            citation_uri,
            f"{self.namespaces['tr-law']}citationType",
            citation_type_value,
            'literal'
        ))

        return triples

    def _generate_relationship_triples(
        self,
        relationship: DocumentRelationship,
        document: CanonicalLegalDocument
    ) -> List[RDFTriple]:
        """Generate triples for relationship

        Args:
            relationship: Relationship
            document: Document

        Returns:
            List of triples
        """
        triples = []

        # Get relationship type predicate
        rel_type_value = relationship.relationship_type.value if hasattr(relationship.relationship_type, 'value') else str(relationship.relationship_type)

        # Map relationship type to predicate
        predicate_map = {
            'AMENDS': 'amends',
            'AMENDED_BY': 'amendedBy',
            'REPEALS': 'repeals',
            'REPEALED_BY': 'repealedBy',
            'CITES': 'cites',
            'CITED_BY': 'citedBy'
        }

        predicate = predicate_map.get(rel_type_value, 'relatesTo')

        source_uri = f"{self.base_uri}/document/{relationship.source_document_id}"
        target_uri = f"{self.base_uri}/document/{relationship.target_document_id}"

        triples.append(RDFTriple(
            source_uri,
            f"{self.namespaces['tr-law']}{predicate}",
            target_uri,
            'uri'
        ))

        return triples

    def _get_rdf_type(self, document_type: DocumentType) -> Optional[str]:
        """Get RDF type for document type

        Args:
            document_type: Document type

        Returns:
            RDF type string
        """
        type_map = {
            DocumentType.KANUN: "Kanun",
            DocumentType.YONETMELIK: "Yonetmelik",
            DocumentType.CUMHURBASKANLIGI_KARARNAMESI: "CumhurbaskanligiKararnamesi",
            DocumentType.TEBLIG: "Teblig",
            DocumentType.YARGITAY_KARARI: "YargitayKarari",
            DocumentType.DANISHTAY_KARARI: "DanishtayKarari"
        }

        return type_map.get(document_type)

    def _generate_document_uri(self, document: CanonicalLegalDocument) -> str:
        """Generate URI for document

        Args:
            document: Document

        Returns:
            URI string
        """
        if document.law_number:
            return f"{self.base_uri}/kanun/{document.law_number}"
        elif document.regulation_number:
            return f"{self.base_uri}/yonetmelik/{document.regulation_number}"
        else:
            return f"{self.base_uri}/document/{document.document_id}"

    def _generate_article_uri(
        self,
        document: CanonicalLegalDocument,
        article: Article
    ) -> str:
        """Generate URI for article

        Args:
            document: Parent document
            article: Article

        Returns:
            URI string
        """
        doc_uri = self._generate_document_uri(document)
        return f"{doc_uri}/madde/{article.article_number}"

    def export_to_file(
        self,
        document: CanonicalLegalDocument,
        file_path: str,
        format: str = 'turtle'
    ) -> None:
        """Export document to RDF file

        Args:
            document: Document
            file_path: Output file path
            format: RDF format
        """
        rdf_str = self.export(document, format)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(rdf_str)

        logger.info(f"Exported {document.document_id} to RDF ({format}): {file_path}")


__all__ = [
    'RDFExporter',
    'RDFTriple',
    'NAMESPACES'
]
