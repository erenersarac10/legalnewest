"""Canonical Models - Harvey/Legora CTO-Level Production-Grade
Pydantic models for canonical Turkish legal document representation

Production Features:
- Comprehensive Pydantic v2 models
- Full Turkish legal document structure
- Nested article/clause hierarchy
- Citation and relationship modeling
- Temporal versioning support
- Metadata enrichment
- Validation and constraints
- Serialization-ready
- Graph-ready structure
- JSON-LD compatible
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import date, datetime
from decimal import Decimal

from .enums import (
    DocumentType, DocumentStatus, AmendmentType, ClauseType,
    AuthorityLevel, PublicationSource, EnforcementStatus,
    CitationType, RelationshipType, TemporalRelation,
    LegalDomain, ConfidenceLevel, ProcessingStatus, LanguageCode
)


# ============================================================================
# BASE MODELS
# ============================================================================

class CanonicalBase(BaseModel):
    """Base model with common configuration"""
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True
    )


# ============================================================================
# CITATION AND REFERENCE MODELS
# ============================================================================

class Citation(CanonicalBase):
    """Legal citation/reference"""
    citation_id: str = Field(..., description="Unique citation identifier")
    citation_type: CitationType = Field(..., description="Type of citation")

    # Source (citing document)
    source_document_id: str = Field(..., description="ID of citing document")
    source_article: Optional[str] = Field(None, description="Citing article number")
    source_clause: Optional[str] = Field(None, description="Citing clause")

    # Target (cited document)
    target_document_id: Optional[str] = Field(None, description="ID of cited document")
    target_law_number: Optional[str] = Field(None, description="Cited law number (e.g., 5237)")
    target_article: Optional[str] = Field(None, description="Cited article number")
    target_clause: Optional[str] = Field(None, description="Cited clause")

    # Citation text
    citation_text: str = Field(..., description="Actual citation text")
    context: Optional[str] = Field(None, description="Surrounding context")

    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_resolved: bool = Field(default=False, description="Whether citation is resolved")


class DocumentRelationship(CanonicalBase):
    """Relationship between legal documents"""
    relationship_id: str = Field(..., description="Unique relationship identifier")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")

    source_document_id: str = Field(..., description="Source document ID")
    target_document_id: str = Field(..., description="Target document ID")

    # Temporal aspect
    effective_date: Optional[date] = Field(None, description="When relationship became effective")
    end_date: Optional[date] = Field(None, description="When relationship ended")

    # Details
    description: Optional[str] = Field(None, description="Relationship description")
    articles_affected: List[str] = Field(default_factory=list, description="Affected article numbers")

    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ============================================================================
# CLAUSE AND ARTICLE MODELS
# ============================================================================

class Clause(CanonicalBase):
    """Legal clause/provision (Bent, Fıkra)"""
    clause_id: str = Field(..., description="Unique clause identifier")
    clause_type: ClauseType = Field(..., description="Type of clause")
    clause_number: Optional[str] = Field(None, description="Clause number/letter")

    # Content
    title: Optional[str] = Field(None, description="Clause title")
    content: str = Field(..., description="Clause text content")

    # Hierarchy
    parent_clause_id: Optional[str] = Field(None, description="Parent clause ID")
    sub_clauses: List['Clause'] = Field(default_factory=list, description="Sub-clauses")

    # Amendment tracking
    amendment_type: Optional[AmendmentType] = Field(None, description="Amendment type if amended")
    amendment_date: Optional[date] = Field(None, description="Amendment date")
    amended_by: Optional[str] = Field(None, description="Amending law/regulation")

    # Status
    is_active: bool = Field(default=True, description="Whether clause is currently active")
    is_repealed: bool = Field(default=False, description="Whether clause is repealed (Mülga)")

    # Metadata
    position: int = Field(..., description="Position in document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Article(CanonicalBase):
    """Legal article (Madde)"""
    article_id: str = Field(..., description="Unique article identifier")
    article_number: str = Field(..., description="Article number")
    article_type: ClauseType = Field(default=ClauseType.MADDE, description="Article type")

    # Content
    title: Optional[str] = Field(None, description="Article title/header")
    content: str = Field(..., description="Article text content")

    # Structure
    paragraphs: List[Clause] = Field(default_factory=list, description="Paragraphs (Fıkra)")
    sub_clauses: List[Clause] = Field(default_factory=list, description="Clauses (Bent)")

    # Amendment tracking
    amendment_type: Optional[AmendmentType] = Field(None, description="Amendment type")
    amendment_date: Optional[date] = Field(None, description="Amendment date")
    amended_by: Optional[str] = Field(None, description="Amending document")
    amendment_history: List[Dict[str, Any]] = Field(default_factory=list, description="Full amendment history")

    # Status
    is_active: bool = Field(default=True)
    is_repealed: bool = Field(default=False, description="Mülga")
    is_temporary: bool = Field(default=False, description="Geçici madde")
    is_additional: bool = Field(default=False, description="Ek madde")

    # Citations
    citations: List[Citation] = Field(default_factory=list, description="Citations from this article")

    # Metadata
    position: int = Field(..., description="Position in document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Section(CanonicalBase):
    """Document section (Bölüm, Kısım)"""
    section_id: str = Field(..., description="Unique section identifier")
    section_type: ClauseType = Field(..., description="Section type (BOLUM, KISIM, etc.)")
    section_number: Optional[str] = Field(None, description="Section number")

    title: str = Field(..., description="Section title")
    articles: List[Article] = Field(default_factory=list, description="Articles in this section")
    subsections: List['Section'] = Field(default_factory=list, description="Subsections")

    position: int = Field(..., description="Position in document")


# ============================================================================
# METADATA MODELS
# ============================================================================

class Authority(CanonicalBase):
    """Issuing authority information"""
    authority_name: str = Field(..., description="Authority name (e.g., TBMM, Cumhurbaşkanlığı)")
    authority_level: AuthorityLevel = Field(..., description="Authority level")
    authority_type: Optional[str] = Field(None, description="Authority type")


class Publication(CanonicalBase):
    """Publication information"""
    publication_source: PublicationSource = Field(..., description="Publication source")
    publication_date: date = Field(..., description="Publication date")

    # Resmi Gazete specifics
    resmi_gazete_tarih: Optional[date] = Field(None, description="Resmi Gazete date")
    resmi_gazete_sayi: Optional[str] = Field(None, description="Resmi Gazete number")

    # Other sources
    publication_number: Optional[str] = Field(None, description="Publication number")
    publication_url: Optional[str] = Field(None, description="Publication URL")


class EnforcementInfo(CanonicalBase):
    """Enforcement/effectivity information"""
    enforcement_status: EnforcementStatus = Field(..., description="Current enforcement status")
    effective_date: Optional[date] = Field(None, description="Date when document became effective")
    end_date: Optional[date] = Field(None, description="Date when document ceased to be effective")

    # Special cases
    is_immediate: bool = Field(default=False, description="Yayımı tarihinde yürürlüğe girer")
    delay_days: Optional[int] = Field(None, description="Delay period in days")
    delay_description: Optional[str] = Field(None, description="Delay description")


class ProcessingMetadata(CanonicalBase):
    """Processing pipeline metadata"""
    processing_status: ProcessingStatus = Field(..., description="Current processing status")

    # Timestamps
    ingested_at: Optional[datetime] = Field(None, description="When document was ingested")
    extracted_at: Optional[datetime] = Field(None, description="When text was extracted")
    parsed_at: Optional[datetime] = Field(None, description="When structure was parsed")
    validated_at: Optional[datetime] = Field(None, description="When validation completed")
    indexed_at: Optional[datetime] = Field(None, description="When indexed for search")

    # Processing info
    extraction_method: Optional[str] = Field(None, description="Extraction method used")
    parser_version: Optional[str] = Field(None, description="Parser version")
    validation_errors: int = Field(default=0, description="Number of validation errors")
    validation_warnings: int = Field(default=0, description="Number of validation warnings")

    # Confidence
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ============================================================================
# MAIN CANONICAL DOCUMENT MODEL
# ============================================================================

class CanonicalLegalDocument(CanonicalBase):
    """
    Canonical representation of Turkish legal document.

    This is the unified, standardized format for all Turkish legal documents
    regardless of their source or original format.
    """

    # ========== IDENTIFICATION ==========
    document_id: str = Field(..., description="Unique document identifier")
    document_type: DocumentType = Field(..., description="Document type")
    document_status: DocumentStatus = Field(..., description="Document lifecycle status")

    # Document numbers
    law_number: Optional[str] = Field(None, description="Law number (e.g., 5237, 6698)")
    regulation_number: Optional[str] = Field(None, description="Regulation number")
    decision_number: Optional[str] = Field(None, description="Decision number")

    # ========== CONTENT ==========
    title: str = Field(..., description="Official document title")
    short_title: Optional[str] = Field(None, description="Short/common title")

    # Full text
    full_text: str = Field(..., description="Complete document text")

    # Structured content
    preamble: Optional[str] = Field(None, description="Preamble/introduction (Dibace)")
    sections: List[Section] = Field(default_factory=list, description="Document sections")
    articles: List[Article] = Field(default_factory=list, description="All articles")
    temporary_articles: List[Article] = Field(default_factory=list, description="Temporary articles (Geçici maddeler)")
    additional_articles: List[Article] = Field(default_factory=list, description="Additional articles (Ek maddeler)")

    # ========== AUTHORITY & PUBLICATION ==========
    issuing_authority: Optional[Authority] = Field(None, description="Issuing authority")
    publication: Optional[Publication] = Field(None, description="Publication information")

    # ========== ENFORCEMENT ==========
    enforcement: Optional[EnforcementInfo] = Field(None, description="Enforcement information")

    # ========== RELATIONSHIPS ==========
    citations: List[Citation] = Field(default_factory=list, description="All citations in document")
    relationships: List[DocumentRelationship] = Field(default_factory=list, description="Relationships with other documents")

    # Parent/child relationships
    parent_document_id: Optional[str] = Field(None, description="Parent document (if this is implementing regulation)")
    child_document_ids: List[str] = Field(default_factory=list, description="Child documents (implementing regulations)")

    # Amendment chain
    amends_document_ids: List[str] = Field(default_factory=list, description="Documents this amends")
    amended_by_document_ids: List[str] = Field(default_factory=list, description="Documents that amend this")

    # ========== CLASSIFICATION ==========
    legal_domains: List[LegalDomain] = Field(default_factory=list, description="Legal domain classification")
    keywords: List[str] = Field(default_factory=list, description="Keywords/tags")
    subjects: List[str] = Field(default_factory=list, description="Subject matter")

    # ========== VERSIONING ==========
    version: str = Field(default="1.0", description="Document version")
    version_date: Optional[date] = Field(None, description="Version date")
    previous_version_id: Optional[str] = Field(None, description="Previous version ID")
    next_version_id: Optional[str] = Field(None, description="Next version ID")
    is_consolidated: bool = Field(default=False, description="Whether this is consolidated version")

    # ========== LANGUAGE ==========
    language: LanguageCode = Field(default=LanguageCode.TR, description="Document language")
    translations: Dict[LanguageCode, str] = Field(default_factory=dict, description="Translations")

    # ========== METADATA ==========
    processing_metadata: Optional[ProcessingMetadata] = Field(None, description="Processing metadata")

    # Source tracking
    source_url: Optional[str] = Field(None, description="Source URL")
    source_file_path: Optional[str] = Field(None, description="Source file path")
    source_hash: Optional[str] = Field(None, description="Source file hash (SHA-256)")

    # Custom metadata
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")

    # ========== TIMESTAMPS ==========
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    @field_validator('law_number')
    @classmethod
    def validate_law_number(cls, v: Optional[str]) -> Optional[str]:
        """Validate law number format"""
        if v and not v.isdigit():
            # Allow formats like "5237 sayılı"
            if 'sayılı' in v:
                parts = v.split()
                if parts and parts[0].isdigit():
                    return parts[0]
        return v

    def get_article(self, article_number: str) -> Optional[Article]:
        """Get article by number"""
        for article in self.articles:
            if article.article_number == article_number:
                return article
        return None

    def get_active_articles(self) -> List[Article]:
        """Get all active (non-repealed) articles"""
        return [a for a in self.articles if a.is_active and not a.is_repealed]

    def get_citations_to(self, target_doc_id: str) -> List[Citation]:
        """Get all citations to a specific document"""
        return [c for c in self.citations if c.target_document_id == target_doc_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump(mode='json', exclude_none=True)


# ============================================================================
# COLLECTION MODELS
# ============================================================================

class DocumentCollection(CanonicalBase):
    """Collection of related legal documents"""
    collection_id: str = Field(..., description="Collection identifier")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")

    documents: List[CanonicalLegalDocument] = Field(default_factory=list, description="Documents in collection")
    document_ids: List[str] = Field(default_factory=list, description="Document IDs in collection")

    collection_type: Optional[str] = Field(None, description="Collection type (e.g., 'law_with_amendments')")

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# Forward references
Section.model_rebuild()
Clause.model_rebuild()


__all__ = [
    'CanonicalLegalDocument',
    'Article',
    'Clause',
    'Section',
    'Citation',
    'DocumentRelationship',
    'Authority',
    'Publication',
    'EnforcementInfo',
    'ProcessingMetadata',
    'DocumentCollection'
]
