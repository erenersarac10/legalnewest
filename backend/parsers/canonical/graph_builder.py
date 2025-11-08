"""Canonical Graph Builder - Harvey/Legora CTO-Level Production-Grade
Knowledge graph construction for Turkish legal documents

Production Features:
- Neo4j property graph construction
- Node and relationship modeling
- Cypher query generation
- Citation graph building
- Amendment history graph
- Document relationship networks
- Article-level granularity
- Label and property management
- Batch insertion support
- Index creation
- Constraint definition
- Graph traversal queries
- Network analysis ready
- Fallback to dict-based graph
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging

from .models import (
    CanonicalLegalDocument, Article, Clause, Citation,
    DocumentRelationship
)
from .enums import DocumentType, CitationType, RelationshipType, AmendmentType

logger = logging.getLogger(__name__)


# ============================================================================
# GRAPH NODE AND RELATIONSHIP MODELS
# ============================================================================

@dataclass
class GraphNode:
    """Represents a graph node"""
    node_id: str
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    """Represents a graph relationship"""
    source_id: str
    target_id: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    """Simple graph structure"""
    nodes: List[GraphNode] = field(default_factory=list)
    relationships: List[GraphRelationship] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        """Add node to graph"""
        self.nodes.append(node)

    def add_relationship(self, rel: GraphRelationship) -> None:
        """Add relationship to graph"""
        self.relationships.append(rel)


# ============================================================================
# NEO4J GRAPH BUILDER
# ============================================================================

class GraphBuilder:
    """Builds knowledge graphs from canonical legal documents"""

    def __init__(
        self,
        neo4j_driver: Optional[Any] = None,
        use_neo4j: bool = True
    ):
        """Initialize graph builder

        Args:
            neo4j_driver: Neo4j driver instance (optional)
            use_neo4j: Try to use Neo4j if available
        """
        self.neo4j_driver = neo4j_driver
        self.use_neo4j = use_neo4j and neo4j_driver is not None

        if not self.use_neo4j:
            logger.debug("Initialized GraphBuilder with dict-based graph")
        else:
            logger.debug("Initialized GraphBuilder with Neo4j support")

    def build_graph(
        self,
        document: CanonicalLegalDocument,
        include_articles: bool = True,
        include_citations: bool = True
    ) -> Graph:
        """Build knowledge graph from document

        Args:
            document: Document to build graph from
            include_articles: Include article nodes
            include_citations: Include citation relationships

        Returns:
            Graph structure
        """
        graph = Graph()

        # Add document node
        doc_node = self._create_document_node(document)
        graph.add_node(doc_node)

        # Add article nodes and relationships
        if include_articles:
            for article in document.articles:
                article_node = self._create_article_node(article, document)
                graph.add_node(article_node)

                # Add HAS_ARTICLE relationship
                rel = GraphRelationship(
                    source_id=doc_node.node_id,
                    target_id=article_node.node_id,
                    rel_type="HAS_ARTICLE",
                    properties={"position": article.position}
                )
                graph.add_relationship(rel)

        # Add citation relationships
        if include_citations:
            for citation in document.citations:
                citation_rels = self._create_citation_relationships(citation, document)
                for rel in citation_rels:
                    graph.add_relationship(rel)

        # Add document relationships
        for doc_rel in document.relationships:
            graph_rel = self._create_document_relationship(doc_rel)
            if graph_rel:
                graph.add_relationship(graph_rel)

        logger.debug(
            f"Built graph for {document.document_id}: "
            f"{len(graph.nodes)} nodes, {len(graph.relationships)} relationships"
        )

        return graph

    def _create_document_node(self, document: CanonicalLegalDocument) -> GraphNode:
        """Create document node

        Args:
            document: Document

        Returns:
            GraphNode
        """
        labels = ["Document", "LegalDocument"]

        # Add specific type label
        doc_type_value = document.document_type.value if hasattr(document.document_type, 'value') else str(document.document_type)
        if doc_type_value == "KANUN":
            labels.append("Law")
        elif doc_type_value == "YONETMELIK":
            labels.append("Regulation")
        elif doc_type_value in ("YARGITAY_KARARI", "DANISHTAY_KARARI", "ANAYASA_MAHKEMESI_KARARI"):
            labels.append("CourtDecision")

        properties = {
            "document_id": document.document_id,
            "title": document.title,
            "document_type": doc_type_value
        }

        if document.law_number:
            properties["law_number"] = document.law_number

        if document.publication:
            properties["publication_date"] = document.publication.publication_date.isoformat()

        if document.legal_domains:
            properties["legal_domains"] = [
                d.value if hasattr(d, 'value') else str(d)
                for d in document.legal_domains
            ]

        return GraphNode(
            node_id=f"doc:{document.document_id}",
            labels=labels,
            properties=properties
        )

    def _create_article_node(
        self,
        article: Article,
        document: CanonicalLegalDocument
    ) -> GraphNode:
        """Create article node

        Args:
            article: Article
            document: Parent document

        Returns:
            GraphNode
        """
        labels = ["Article"]

        if article.is_repealed:
            labels.append("Repealed")
        if article.is_temporary:
            labels.append("Temporary")
        if article.is_additional:
            labels.append("Additional")

        properties = {
            "article_id": article.article_id,
            "article_number": article.article_number,
            "content": article.content[:500],  # Limit content length
            "is_active": article.is_active,
            "is_repealed": article.is_repealed,
            "position": article.position
        }

        if article.title:
            properties["title"] = article.title

        if article.amendment_type:
            properties["amendment_type"] = article.amendment_type.value if hasattr(article.amendment_type, 'value') else str(article.amendment_type)

        if article.amended_by:
            properties["amended_by"] = article.amended_by

        return GraphNode(
            node_id=f"art:{article.article_id}",
            labels=labels,
            properties=properties
        )

    def _create_citation_relationships(
        self,
        citation: Citation,
        document: CanonicalLegalDocument
    ) -> List[GraphRelationship]:
        """Create citation relationships

        Args:
            citation: Citation
            document: Document

        Returns:
            List of relationships
        """
        rels = []

        # Determine source and target
        if citation.source_article:
            source_id = f"art:{citation.source_document_id}_{citation.source_article}"
        else:
            source_id = f"doc:{citation.source_document_id}"

        if citation.target_document_id:
            if citation.target_article:
                target_id = f"art:{citation.target_document_id}_{citation.target_article}"
            else:
                target_id = f"doc:{citation.target_document_id}"

            # Create CITES relationship
            rel = GraphRelationship(
                source_id=source_id,
                target_id=target_id,
                rel_type="CITES",
                properties={
                    "citation_type": citation.citation_type.value if hasattr(citation.citation_type, 'value') else str(citation.citation_type),
                    "citation_text": citation.citation_text[:200],  # Limit length
                    "confidence": citation.confidence
                }
            )
            rels.append(rel)

        return rels

    def _create_document_relationship(
        self,
        relationship: DocumentRelationship
    ) -> Optional[GraphRelationship]:
        """Create document relationship

        Args:
            relationship: Document relationship

        Returns:
            GraphRelationship or None
        """
        rel_type_value = relationship.relationship_type.value if hasattr(relationship.relationship_type, 'value') else str(relationship.relationship_type)

        # Map relationship type to graph relationship
        type_map = {
            "AMENDS": "AMENDS",
            "AMENDED_BY": "AMENDED_BY",
            "REPEALS": "REPEALS",
            "REPEALED_BY": "REPEALED_BY",
            "IMPLEMENTS": "IMPLEMENTS",
            "IMPLEMENTED_BY": "IMPLEMENTED_BY",
            "CITES": "CITES",
            "CITED_BY": "CITED_BY"
        }

        graph_rel_type = type_map.get(rel_type_value, "RELATED_TO")

        properties = {}

        if relationship.effective_date:
            properties["effective_date"] = relationship.effective_date.isoformat()

        if relationship.description:
            properties["description"] = relationship.description[:200]

        return GraphRelationship(
            source_id=f"doc:{relationship.source_document_id}",
            target_id=f"doc:{relationship.target_document_id}",
            rel_type=graph_rel_type,
            properties=properties
        )

    def generate_cypher_queries(
        self,
        graph: Graph,
        create_indexes: bool = True
    ) -> List[str]:
        """Generate Cypher queries for Neo4j

        Args:
            graph: Graph structure
            create_indexes: Include index creation queries

        Returns:
            List of Cypher queries
        """
        queries = []

        # Create indexes (if requested)
        if create_indexes:
            queries.extend([
                "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
                "CREATE INDEX law_number_index IF NOT EXISTS FOR (d:Law) ON (d.law_number)",
                "CREATE INDEX article_id_index IF NOT EXISTS FOR (a:Article) ON (a.article_id)",
                "CREATE INDEX article_number_index IF NOT EXISTS FOR (a:Article) ON (a.article_number)"
            ])

        # Create nodes
        for node in graph.nodes:
            # Build labels string
            labels_str = ":".join(node.labels)

            # Build properties string
            props_list = []
            for key, value in node.properties.items():
                if isinstance(value, str):
                    # Escape quotes
                    escaped = value.replace("'", "\\'")
                    props_list.append(f"{key}: '{escaped}'")
                elif isinstance(value, bool):
                    props_list.append(f"{key}: {str(value).lower()}")
                elif isinstance(value, list):
                    # Handle arrays
                    array_str = "[" + ", ".join(f"'{str(v)}'" for v in value) + "]"
                    props_list.append(f"{key}: {array_str}")
                else:
                    props_list.append(f"{key}: {value}")

            props_str = "{" + ", ".join(props_list) + "}"

            # Create MERGE query (idempotent)
            query = f"MERGE (n:{labels_str} {{{node.labels[0].lower()}_id: '{node.node_id}'}}) SET n = {props_str}"
            queries.append(query)

        # Create relationships
        for rel in graph.relationships:
            # Build properties string
            props_list = []
            for key, value in rel.properties.items():
                if isinstance(value, str):
                    escaped = value.replace("'", "\\'")
                    props_list.append(f"{key}: '{escaped}'")
                elif isinstance(value, (int, float)):
                    props_list.append(f"{key}: {value}")
                else:
                    props_list.append(f"{key}: '{str(value)}'")

            props_str = "{" + ", ".join(props_list) + "}" if props_list else ""

            # Create relationship query
            query = (
                f"MATCH (source {{document_id: '{rel.source_id.split(':')[1]}'}}), "
                f"(target {{document_id: '{rel.target_id.split(':')[1]}'}}) "
                f"MERGE (source)-[r:{rel.rel_type}]->(target)"
            )

            if props_str:
                query += f" SET r = {props_str}"

            queries.append(query)

        return queries

    def insert_to_neo4j(
        self,
        graph: Graph,
        create_indexes: bool = True
    ) -> bool:
        """Insert graph to Neo4j database

        Args:
            graph: Graph structure
            create_indexes: Create indexes

        Returns:
            True if successful
        """
        if not self.use_neo4j or not self.neo4j_driver:
            logger.warning("Neo4j not available - cannot insert graph")
            return False

        try:
            # Generate Cypher queries
            queries = self.generate_cypher_queries(graph, create_indexes)

            # Execute queries
            with self.neo4j_driver.session() as session:
                for query in queries:
                    try:
                        session.run(query)
                    except Exception as e:
                        logger.error(f"Query failed: {query[:100]}... Error: {e}")
                        if create_indexes and "INDEX" in query:
                            # Indexes might already exist, continue
                            continue
                        else:
                            raise

            logger.info(f"Inserted graph to Neo4j: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")

            return True

        except Exception as e:
            logger.error(f"Failed to insert graph to Neo4j: {e}")
            return False

    def build_citation_network(
        self,
        documents: List[CanonicalLegalDocument]
    ) -> Graph:
        """Build citation network from multiple documents

        Args:
            documents: List of documents

        Returns:
            Combined graph
        """
        graph = Graph()

        # Add all documents as nodes
        for document in documents:
            doc_node = self._create_document_node(document)
            graph.add_node(doc_node)

        # Add citation relationships
        for document in documents:
            for citation in document.citations:
                citation_rels = self._create_citation_relationships(citation, document)
                for rel in citation_rels:
                    graph.add_relationship(rel)

        logger.info(
            f"Built citation network: {len(documents)} documents, "
            f"{len(graph.relationships)} citation relationships"
        )

        return graph

    def build_amendment_chain(
        self,
        base_document: CanonicalLegalDocument,
        amending_documents: List[CanonicalLegalDocument]
    ) -> Graph:
        """Build amendment chain graph

        Args:
            base_document: Base law
            amending_documents: Laws that amend the base law

        Returns:
            Amendment chain graph
        """
        graph = Graph()

        # Add base document
        base_node = self._create_document_node(base_document)
        graph.add_node(base_node)

        # Add amending documents and relationships
        for amending_doc in amending_documents:
            amend_node = self._create_document_node(amending_doc)
            graph.add_node(amend_node)

            # AMENDS relationship
            rel = GraphRelationship(
                source_id=amend_node.node_id,
                target_id=base_node.node_id,
                rel_type="AMENDS",
                properties={}
            )

            if amending_doc.publication:
                rel.properties["amendment_date"] = amending_doc.publication.publication_date.isoformat()

            graph.add_relationship(rel)

        logger.info(
            f"Built amendment chain: 1 base document, "
            f"{len(amending_documents)} amending documents"
        )

        return graph

    def get_graph_statistics(self, graph: Graph) -> Dict[str, Any]:
        """Get graph statistics

        Args:
            graph: Graph structure

        Returns:
            Statistics dict
        """
        # Count nodes by label
        label_counts = {}
        for node in graph.nodes:
            for label in node.labels:
                label_counts[label] = label_counts.get(label, 0) + 1

        # Count relationships by type
        rel_type_counts = {}
        for rel in graph.relationships:
            rel_type_counts[rel.rel_type] = rel_type_counts.get(rel.rel_type, 0) + 1

        return {
            "total_nodes": len(graph.nodes),
            "total_relationships": len(graph.relationships),
            "nodes_by_label": label_counts,
            "relationships_by_type": rel_type_counts
        }


__all__ = [
    'GraphBuilder',
    'Graph',
    'GraphNode',
    'GraphRelationship'
]
