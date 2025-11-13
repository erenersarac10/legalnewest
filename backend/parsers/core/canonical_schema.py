"""
Canonical Legal Document Schema - Harvey/Legora CTO-Level

Pydantic models representing Turkish legal documents in a unified canonical format.
All parsers convert source documents to this schema for consistent downstream processing.

Document Types:
    - Statute: Kanun, Yönetmelik, Tüzük, CBK, Tebliğ, Genelge
    - CourtDecision: Yargıtay, Danıştay, AYM, İdare Mahkemesi
    - Regulation: SPK, BDDK, KVKK tebliğleri
    - International: AİHM, AB mevzuatı

Key Features:
    - Turkish legal system specifics (Resmi Gazete, e-Mevzuat)
    - Rich metadata (jurisdiction, hierarchy, effectivity)
    - Citation tracking and validation
    - Version management (amendments, repeals)
    - KVKK-compliant (no PII storage)

Author: Legal AI Team
Version: 1.0.0
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from uuid import uuid4


# ============================================================================
# ENUMS
# ============================================================================

class DocumentType(str, Enum):
    """Type of legal document."""
    KANUN = "kanun"  # Law
    YONETMELIK = "yonetmelik"  # Regulation
    TUZUK = "tuzuk"  # Statute
    CBK = "cbk"  # Cumhurbaşkanlığı Kararnamesi (Presidential Decree)
    TEBLIG = "teblig"  # Communique
    GENELGE = "genelge"  # Circular
    YARGITAY_KARARI = "yargitay_karari"  # Court of Cassation Decision
    DANISTAY_KARARI = "danistay_karari"  # Council of State Decision
    AYM_KARARI = "aym_karari"  # Constitutional Court Decision
    IDARE_MAHKEMESI_KARARI = "idare_mahkemesi_karari"  # Administrative Court
    AIHM_KARARI = "aihm_karari"  # ECHR Decision
    AB_MEVZUAT = "ab_mevzuat"  # EU Legislation
    UNKNOWN = "unknown"


class LegalHierarchy(int, Enum):
    """
    Turkish legal hierarchy (Türk Hukuk Hiyerarşisi).
    Lower number = higher authority.
    """
    ANAYASA = 1  # Constitution
    KANUN = 2  # Laws
    KHK = 3  # Decree Laws (historical)
    CBK = 4  # Presidential Decrees
    TUZUK = 5  # Statutes
    YONETMELIK = 6  # Regulations
    TEBLIG = 7  # Communiques
    GENELGE = 8  # Circulars
    ICTIHAT = 9  # Case law
    OZELGE = 10  # Private rulings
    DOKTRINEL = 11  # Academic doctrine


class JurisdictionType(str, Enum):
    """Turkish legal jurisdiction."""
    CEZA = "ceza"  # Criminal
    HUKUK = "hukuk"  # Civil
    IDARE = "idare"  # Administrative
    ANAYASA = "anayasa"  # Constitutional
    VERGI = "vergi"  # Tax
    TICARET = "ticaret"  # Commercial
    IS = "is"  # Labor
    ICRA_IFLAS = "icra_iflas"  # Execution & Bankruptcy
    ULUSLARARASI = "uluslararasi"  # International


class EffectivityStatus(str, Enum):
    """Document effectivity status."""
    YURURLUKTE = "yururlukte"  # In effect
    YURURLUK_BEKLIYOR = "yururluk_bekliyor"  # Pending effect
    YURURLUKTEN_KALKMIS = "yururlukten_kalkmis"  # Repealed
    DEGISTIRILMIS = "degistirilmis"  # Amended
    ASKIDA = "askida"  # Suspended
    IPTAL_EDILMIS = "iptal_edilmis"  # Annulled


class SourceType(str, Enum):
    """Legal document source."""
    RESMI_GAZETE = "resmi_gazete"
    MEVZUAT_GOV = "mevzuat_gov"
    YARGITAY = "yargitay"
    DANISTAY = "danistay"
    ANAYASA_MAHKEMESI = "anayasa_mahkemesi"
    KAZANCI = "kazanci"
    LEXPERA = "lexpera"
    KVKK = "kvkk"
    SPK = "spk"
    BDDK = "bddk"
    AIHM = "aihm"
    EUR_LEX = "eur_lex"


# ============================================================================
# CORE MODELS
# ============================================================================

class Citation(BaseModel):
    """Legal citation reference."""

    text: str = Field(..., description="Full citation text")
    cited_document_id: Optional[str] = Field(None, description="ID of cited document if resolved")
    citation_type: str = Field(..., description="Type: statute, case, doctrine")
    is_resolved: bool = Field(False, description="Whether citation was resolved to actual document")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Citation extraction confidence")

    # Turkish-specific fields
    rg_number: Optional[str] = Field(None, description="Resmi Gazete number")
    rg_date: Optional[date] = Field(None, description="Resmi Gazete publication date")
    law_number: Optional[str] = Field(None, description="Kanun No")
    article_number: Optional[str] = Field(None, description="Madde No")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "5237 Sayılı TCK Madde 141",
                "citation_type": "statute",
                "is_resolved": True,
                "law_number": "5237",
                "article_number": "141"
            }
        }


class LegalClause(BaseModel):
    """Single clause/article (Madde, Fıkra, Bent) in a legal document."""

    clause_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique clause ID")
    clause_number: str = Field(..., description="Madde/Fıkra/Bent number")
    clause_type: str = Field(..., description="Type: madde, fikra, bent, paragraf")
    title: Optional[str] = Field(None, description="Clause title (Madde Başlığı)")
    text: str = Field(..., description="Full clause text")

    # Hierarchy
    parent_clause_id: Optional[str] = Field(None, description="Parent clause (e.g., fıkra → madde)")
    child_clauses: List[LegalClause] = Field(default_factory=list, description="Child clauses")

    # References
    citations: List[Citation] = Field(default_factory=list, description="Citations within this clause")
    cross_references: List[str] = Field(default_factory=list, description="Cross-refs to other clauses")

    # Amendments
    amendment_history: List[Dict[str, Any]] = Field(default_factory=list, description="Amendment records")
    is_repealed: bool = Field(False, description="Whether clause is repealed")

    class Config:
        json_schema_extra = {
            "example": {
                "clause_number": "141",
                "clause_type": "madde",
                "title": "Devleti tahkir",
                "text": "Türkiye Cumhuriyeti Devletini..."
            }
        }


class Metadata(BaseModel):
    """Document metadata."""

    # Identifiers
    canonical_id: str = Field(default_factory=lambda: str(uuid4()), description="Canonical document ID")
    source_id: Optional[str] = Field(None, description="Source system ID")
    external_ids: Dict[str, str] = Field(default_factory=dict, description="External system IDs")

    # Classification
    document_type: DocumentType
    jurisdiction: JurisdictionType
    hierarchy_level: LegalHierarchy
    subject_areas: List[str] = Field(default_factory=list, description="Subject classification (e.g., Ceza Hukuku)")
    keywords: List[str] = Field(default_factory=list, description="Keywords")

    # Publication
    source: SourceType
    source_url: Optional[str] = Field(None, description="Original source URL")
    resmi_gazete_number: Optional[str] = Field(None, description="RG No")
    resmi_gazete_date: Optional[date] = Field(None, description="RG publication date")

    # Effectivity
    effectivity_status: EffectivityStatus = Field(EffectivityStatus.YURURLUKTE)
    effectivity_date: Optional[date] = Field(None, description="Yürürlük tarihi")
    repeal_date: Optional[date] = Field(None, description="Yürürlükten kaldırma tarihi")

    # Provenance
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    parser_version: str = Field("1.0.0")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Overall parsing confidence")

    class Config:
        use_enum_values = True


class LegalDocument(BaseModel):
    """
    Base canonical legal document model.
    All parsed documents conform to this schema.
    """

    # Core fields
    metadata: Metadata
    title: str = Field(..., description="Document title")
    summary: Optional[str] = Field(None, description="Document summary")
    full_text: str = Field(..., description="Complete document text")

    # Structure
    clauses: List[LegalClause] = Field(default_factory=list, description="All clauses in document")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Document sections (Bölüm, Kısım)")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Attachments (Ek)")

    # Citations & References
    citations: List[Citation] = Field(default_factory=list, description="All citations in document")
    cited_by: List[str] = Field(default_factory=list, description="Documents citing this one")

    # Relationships
    amends: List[str] = Field(default_factory=list, description="Documents this one amends")
    amended_by: List[str] = Field(default_factory=list, description="Documents that amended this one")
    repeals: List[str] = Field(default_factory=list, description="Documents this one repeals")
    repealed_by: Optional[str] = Field(None, description="Document that repealed this one")

    # Validation
    validation_errors: List[str] = Field(default_factory=list, description="Validation warnings")
    extraction_warnings: List[str] = Field(default_factory=list, description="Extraction warnings")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Türk Ceza Kanunu",
                "metadata": {
                    "document_type": "kanun",
                    "jurisdiction": "ceza",
                    "resmi_gazete_number": "25611"
                }
            }
        }

    @validator("full_text")
    def validate_text_not_empty(cls, v):
        """Ensure document text is not empty."""
        if not v or not v.strip():
            raise ValueError("Document text cannot be empty")
        return v

    @root_validator
    def validate_effectivity(cls, values):
        """Validate effectivity date logic."""
        status = values.get("metadata", {}).effectivity_status if isinstance(values.get("metadata"), Metadata) else None
        repeal_date = values.get("metadata", {}).repeal_date if isinstance(values.get("metadata"), Metadata) else None

        if status == EffectivityStatus.YURURLUKTEN_KALKMIS and not repeal_date:
            values.setdefault("validation_errors", []).append(
                "Document marked as repealed but no repeal_date provided"
            )

        return values


class Statute(LegalDocument):
    """
    Legislative document (Kanun, Yönetmelik, CBK, etc.).
    """

    # Statute-specific fields
    law_number: Optional[str] = Field(None, description="Kanun No (e.g., 5237)")
    regulation_number: Optional[str] = Field(None, description="Yönetmelik No")

    # Legislative process
    enacted_by: Optional[str] = Field(None, description="Enacting authority (TBMM, Cumhurbaşkanı)")
    enactment_date: Optional[date] = Field(None, description="Kabul tarihi")

    # Version control
    version: str = Field("1.0", description="Document version")
    amendment_count: int = Field(0, description="Number of amendments")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Türk Ceza Kanunu",
                "law_number": "5237",
                "enacted_by": "TBMM",
                "enactment_date": "2004-09-26"
            }
        }


class CourtDecision(LegalDocument):
    """
    Court decision (Yargıtay, Danıştay, AYM, etc.).
    """

    # Case identifiers
    case_number: str = Field(..., description="Esas No")
    decision_number: str = Field(..., description="Karar No")

    # Court info
    court_name: str = Field(..., description="Mahkeme adı")
    court_level: str = Field(..., description="Mahkeme seviyesi (Yargıtay, İlk Derece)")
    chamber: Optional[str] = Field(None, description="Daire No")

    # Decision details
    decision_date: date = Field(..., description="Karar tarihi")
    decision_type: str = Field(..., description="Karar türü (Bozma, Onam, Düzeltme)")
    verdict: str = Field(..., description="Hüküm")

    # Parties (KVKK-compliant: no PII, only anonymized roles)
    plaintiff_role: Optional[str] = Field(None, description="Davacı rolü (anonymized)")
    defendant_role: Optional[str] = Field(None, description="Davalı rolü (anonymized)")

    # Legal reasoning
    legal_reasoning: Optional[str] = Field(None, description="Gerekçe")
    precedent_value: str = Field("normal", description="İçtihat değeri (yüksek, normal, düşük)")
    is_unifying_decision: bool = Field(False, description="İçtihadı Birleştirme Kararı mı?")

    class Config:
        json_schema_extra = {
            "example": {
                "case_number": "2020/1234",
                "decision_number": "2021/567",
                "court_name": "Yargıtay 12. Ceza Dairesi",
                "decision_date": "2021-05-15"
            }
        }


# ============================================================================
# PARSING RESULT
# ============================================================================

class ParsingResult(BaseModel):
    """
    Result of parsing operation.
    Contains parsed document + metadata about parsing process.
    """

    success: bool = Field(..., description="Whether parsing succeeded")
    document: Optional[Union[LegalDocument, Statute, CourtDecision]] = Field(None, description="Parsed document")

    # Parsing metadata
    parser_name: str = Field(..., description="Name of parser used")
    parsing_duration_ms: float = Field(..., description="Parsing duration in milliseconds")

    # Quality metrics
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Overall confidence")
    completeness_score: float = Field(1.0, ge=0.0, le=1.0, description="Completeness (1.0 = all fields extracted)")

    # Errors & warnings
    errors: List[str] = Field(default_factory=list, description="Fatal errors")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")

    # Source tracking
    source_url: Optional[str] = Field(None, description="Source URL")
    source_checksum: Optional[str] = Field(None, description="Source document checksum")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "parser_name": "YargitayAdapter",
                "parsing_duration_ms": 245.3,
                "confidence_score": 0.95
            }
        }


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

__all__ = [
    # Enums
    "DocumentType",
    "LegalHierarchy",
    "JurisdictionType",
    "EffectivityStatus",
    "SourceType",

    # Models
    "Citation",
    "LegalClause",
    "Metadata",
    "LegalDocument",
    "Statute",
    "CourtDecision",
    "ParsingResult",
]
