"""Canonical Enums - Harvey/Legora CTO-Level Production-Grade
Enumeration types for canonical Turkish legal document models

Production Features:
- Comprehensive document type taxonomy
- Turkish legal status codes
- Clause and article types
- Amendment types (Değişik, Mülga, İhdas)
- Legal authority levels
- Publication status
- Enforcement status
- Citation types
- Relationship types
"""
from enum import Enum, auto


class DocumentType(Enum):
    """Turkish legal document types"""
    # Primary legislation
    KANUN = "KANUN"  # Law - Primary legislation
    KANUN_HUKMUNDE_KARARNAME = "KANUN_HUKMUNDE_KARARNAME"  # Decree Law (KHK)
    CUMHURBASKANLIGI_KARARNAMESI = "CUMHURBASKANLIGI_KARARNAMESI"  # Presidential Decree

    # Secondary legislation
    YONETMELIK = "YONETMELIK"  # Regulation
    TUZUK = "TUZUK"  # Bylaw
    YONERGE = "YONERGE"  # Directive
    TALIMAT = "TALIMAT"  # Instruction

    # Administrative acts
    GENELGE = "GENELGE"  # Circular
    TEBLIG = "TEBLIG"  # Communique
    SIRKULER = "SIRKULER"  # Circular letter
    DUYURU = "DUYURU"  # Announcement

    # Judicial decisions
    YARGITAY_KARARI = "YARGITAY_KARARI"  # Supreme Court decision
    DANISHTAY_KARARI = "DANISHTAY_KARARI"  # Council of State decision
    ANAYASA_MAHKEMESI_KARARI = "ANAYASA_MAHKEMESI_KARARI"  # Constitutional Court decision
    BOLGE_ADLIYE_MAHKEMESI_KARARI = "BOLGE_ADLIYE_MAHKEMESI_KARARI"  # Regional Court decision
    ILKDERECE_MAHKEMESI_KARARI = "ILKDERECE_MAHKEMESI_KARARI"  # First instance court decision

    # International
    ULUSLARARASI_ANTLASHMA = "ULUSLARARASI_ANTLASHMA"  # International treaty
    AVRUPA_BIRLIGI_DIREKTIFI = "AVRUPA_BIRLIGI_DIREKTIFI"  # EU directive
    AVRUPA_BIRLIGI_TUZUGU = "AVRUPA_BIRLIGI_TUZUGU"  # EU regulation

    # Other
    GORUSH = "GORUSH"  # Opinion
    ICTIHAT = "ICTIHAT"  # Case law
    UNKNOWN = "UNKNOWN"


class DocumentStatus(Enum):
    """Document lifecycle status"""
    DRAFT = "DRAFT"  # Taslak
    PROPOSED = "PROPOSED"  # Teklif
    UNDER_REVIEW = "UNDER_REVIEW"  # İnceleme altında
    APPROVED = "APPROVED"  # Onaylandı
    PUBLISHED = "PUBLISHED"  # Yayımlandı
    IN_FORCE = "IN_FORCE"  # Yürürlükte
    AMENDED = "AMENDED"  # Değiştirildi
    REPEALED = "REPEALED"  # Mülga
    SUSPENDED = "SUSPENDED"  # Askıya alındı
    EXPIRED = "EXPIRED"  # Süresi doldu
    SUPERSEDED = "SUPERSEDED"  # Yürürlükten kalktı


class AmendmentType(Enum):
    """Turkish legal amendment types"""
    DEGISHIK = "DEGISHIK"  # Amended/Modified
    MULGA = "MULGA"  # Repealed
    IHDAS = "IHDAS"  # Added/Introduced
    EK = "EK"  # Addendum
    GECICI = "GECICI"  # Temporary
    YURURLUKTEN_KALDIRILDI = "YURURLUKTEN_KALDIRILDI"  # Removed from force
    MADDE_DEGISHIKLIGI = "MADDE_DEGISHIKLIGI"  # Article amendment
    FIKRA_DEGISHIKLIGI = "FIKRA_DEGISHIKLIGI"  # Paragraph amendment
    BENT_DEGISHIKLIGI = "BENT_DEGISHIKLIGI"  # Clause amendment


class ClauseType(Enum):
    """Legal clause/provision types"""
    # Document structure
    MADDE = "MADDE"  # Article
    FIKRA = "FIKRA"  # Paragraph
    BENT = "BENT"  # Subparagraph/Clause
    ALT_BENT = "ALT_BENT"  # Sub-clause

    # Special sections
    GECICI_MADDE = "GECICI_MADDE"  # Temporary article
    EK_MADDE = "EK_MADDE"  # Additional article
    YURURLUK_MADDESI = "YURURLUK_MADDESI"  # Effectivity article

    # Document parts
    KISIM = "KISIM"  # Part
    BOLUM = "BOLUM"  # Chapter
    AYIRIM = "AYIRIM"  # Section
    KESIM = "KESIM"  # Subsection

    # Content types
    TANIM = "TANIM"  # Definition
    KAPSAM = "KAPSAM"  # Scope
    AMAC = "AMAC"  # Purpose
    TEMEL_ILKELER = "TEMEL_ILKELER"  # Basic principles
    YASAKLAR = "YASAKLAR"  # Prohibitions
    YÜKÜMLÜLÜKLER = "YÜKÜMLÜLÜKLER"  # Obligations
    CEZALAR = "CEZALAR"  # Penalties
    ATIFLAR = "ATIFLAR"  # References


class AuthorityLevel(Enum):
    """Legal authority hierarchy"""
    ANAYASA = "ANAYASA"  # Constitution - Highest
    KANUN = "KANUN"  # Law
    KHK = "KHK"  # Decree Law
    CUMHURBASKANLIGI_KARARNAMESI = "CUMHURBASKANLIGI_KARARNAMESI"  # Presidential Decree
    BAKANLAR_KURULU_KARARI = "BAKANLAR_KURULU_KARARI"  # Cabinet decision
    YONETMELIK = "YONETMELIK"  # Regulation
    TEBLIG = "TEBLIG"  # Communique
    GENELGE = "GENELGE"  # Circular
    MAHKEME_KARARI = "MAHKEME_KARARI"  # Court decision
    ULUSLARARASI = "ULUSLARARASI"  # International


class PublicationSource(Enum):
    """Official publication sources"""
    RESMI_GAZETE = "RESMI_GAZETE"  # Official Gazette
    TBMM = "TBMM"  # Parliament
    YARGITAY = "YARGITAY"  # Supreme Court
    DANISHTAY = "DANISHTAY"  # Council of State
    ANAYASA_MAHKEMESI = "ANAYASA_MAHKEMESI"  # Constitutional Court
    MEVZUAT_GOV_TR = "MEVZUAT_GOV_TR"  # mevzuat.gov.tr
    KAZANCI = "KAZANCI"  # Kazancı database
    LEGALBANK = "LEGALBANK"  # Legal Bank
    OTHER = "OTHER"


class EnforcementStatus(Enum):
    """Enforcement/effectivity status"""
    NOT_IN_FORCE = "NOT_IN_FORCE"  # Henüz yürürlükte değil
    IN_FORCE = "IN_FORCE"  # Yürürlükte
    PARTIALLY_IN_FORCE = "PARTIALLY_IN_FORCE"  # Kısmen yürürlükte
    SUSPENDED = "SUSPENDED"  # Askıya alınmış
    REPEALED = "REPEALED"  # Mülga
    EXPIRED = "EXPIRED"  # Süresi dolmuş


class CitationType(Enum):
    """Legal citation types"""
    EXPLICIT = "EXPLICIT"  # Açık atıf (madde numarasıyla)
    IMPLICIT = "IMPLICIT"  # Örtük atıf
    CROSS_REFERENCE = "CROSS_REFERENCE"  # Çapraz atıf
    DEFINITION = "DEFINITION"  # Tanım atfı
    AMENDMENT = "AMENDMENT"  # Değişiklik atfı
    REPEAL = "REPEAL"  # Mülga atfı
    JUDICIAL_INTERPRETATION = "JUDICIAL_INTERPRETATION"  # Yargısal yorum
    PRECEDENT = "PRECEDENT"  # İçtihat
    COMMENTARY = "COMMENTARY"  # Şerh/yorum


class RelationshipType(Enum):
    """Document relationship types"""
    AMENDS = "AMENDS"  # Değiştirir
    AMENDED_BY = "AMENDED_BY"  # Değiştirildi
    REPEALS = "REPEALS"  # Yürürlükten kaldırır
    REPEALED_BY = "REPEALED_BY"  # Yürürlükten kaldırıldı
    IMPLEMENTS = "IMPLEMENTS"  # Uygular
    IMPLEMENTED_BY = "IMPLEMENTED_BY"  # Uygulanır
    CITES = "CITES"  # Atıf yapar
    CITED_BY = "CITED_BY"  # Atıf yapıldı
    SUPERSEDES = "SUPERSEDES"  # Yerini alır
    SUPERSEDED_BY = "SUPERSEDED_BY"  # Yerini aldı
    RELATES_TO = "RELATES_TO"  # İlişkili
    PART_OF = "PART_OF"  # Parçası
    HAS_PART = "HAS_PART"  # Parça içerir
    DEPENDS_ON = "DEPENDS_ON"  # Bağlıdır
    CONFLICTS_WITH = "CONFLICTS_WITH"  # Çelişir


class TemporalRelation(Enum):
    """Temporal relationships between document versions"""
    PREDECESSOR = "PREDECESSOR"  # Önceki versiyon
    SUCCESSOR = "SUCCESSOR"  # Sonraki versiyon
    CONCURRENT = "CONCURRENT"  # Eşzamanlı
    EFFECTIVE_FROM = "EFFECTIVE_FROM"  # Geçerli başlangıç
    EFFECTIVE_UNTIL = "EFFECTIVE_UNTIL"  # Geçerli bitiş
    PUBLISHED_BEFORE = "PUBLISHED_BEFORE"  # Önce yayımlandı
    PUBLISHED_AFTER = "PUBLISHED_AFTER"  # Sonra yayımlandı


class LegalDomain(Enum):
    """Legal domain classification"""
    CEZA_HUKUKU = "CEZA_HUKUKU"  # Criminal Law
    MEDENI_HUKUK = "MEDENI_HUKUK"  # Civil Law
    TICARET_HUKUKU = "TICARET_HUKUKU"  # Commercial Law
    IDARE_HUKUKU = "IDARE_HUKUKU"  # Administrative Law
    ANAYASA_HUKUKU = "ANAYASA_HUKUKU"  # Constitutional Law
    IS_HUKUKU = "IS_HUKUKU"  # Labor Law
    VERGI_HUKUKU = "VERGI_HUKUKU"  # Tax Law
    ICRA_IFLAS_HUKUKU = "ICRA_IFLAS_HUKUKU"  # Enforcement and Bankruptcy Law
    USUL_HUKUKU = "USUL_HUKUKU"  # Procedural Law
    MILLETLERARASI_HUKUK = "MILLETLERARASI_HUKUK"  # International Law
    AILE_HUKUKU = "AILE_HUKUKU"  # Family Law
    MIRAS_HUKUKU = "MIRAS_HUKUKU"  # Inheritance Law
    BORÇLAR_HUKUKU = "BORÇLAR_HUKUKU"  # Law of Obligations
    ESYA_HUKUKU = "ESYA_HUKUKU"  # Property Law
    FIKRI_MULKIYET = "FIKRI_MULKIYET"  # Intellectual Property
    REKABET_HUKUKU = "REKABET_HUKUKU"  # Competition Law
    TUKETICI_HUKUKU = "TUKETICI_HUKUKU"  # Consumer Law
    ÇEVRE_HUKUKU = "ÇEVRE_HUKUKU"  # Environmental Law
    ENERJI_HUKUKU = "ENERJI_HUKUKU"  # Energy Law
    TELEKOMUNIKASYON_HUKUKU = "TELEKOMUNIKASYON_HUKUKU"  # Telecommunications Law
    BANKACILIR_HUKUKU = "BANKACILIR_HUKUKU"  # Banking Law
    SERMAYE_PIYASASI_HUKUKU = "SERMAYE_PIYASASI_HUKUKU"  # Capital Markets Law
    KIŞISEL_VERILERIN_KORUNMASI = "KIŞISEL_VERILERIN_KORUNMASI"  # Data Protection


class ConfidenceLevel(Enum):
    """Confidence levels for parsing/classification"""
    VERY_HIGH = "VERY_HIGH"  # >95%
    HIGH = "HIGH"  # 80-95%
    MEDIUM = "MEDIUM"  # 60-80%
    LOW = "LOW"  # 40-60%
    VERY_LOW = "VERY_LOW"  # <40%
    UNKNOWN = "UNKNOWN"


class ProcessingStatus(Enum):
    """Document processing pipeline status"""
    RAW = "RAW"  # Raw/unprocessed
    EXTRACTED = "EXTRACTED"  # Text extracted
    PARSED = "PARSED"  # Structure parsed
    VALIDATED = "VALIDATED"  # Validated
    ENRICHED = "ENRICHED"  # Semantically enriched
    INDEXED = "INDEXED"  # Indexed for search
    PUBLISHED = "PUBLISHED"  # Published/available
    ERROR = "ERROR"  # Processing error


class LanguageCode(Enum):
    """Language codes for multilingual support"""
    TR = "TR"  # Turkish
    EN = "EN"  # English
    FR = "FR"  # French
    DE = "DE"  # German
    AR = "AR"  # Arabic


class ValidationSeverity(Enum):
    """Validation issue severity"""
    ERROR = "ERROR"  # Critical - must fix
    WARNING = "WARNING"  # Important - should fix
    INFO = "INFO"  # Informational
    SUCCESS = "SUCCESS"  # Validation passed


__all__ = [
    'DocumentType',
    'DocumentStatus',
    'AmendmentType',
    'ClauseType',
    'AuthorityLevel',
    'PublicationSource',
    'EnforcementStatus',
    'CitationType',
    'RelationshipType',
    'TemporalRelation',
    'LegalDomain',
    'ConfidenceLevel',
    'ProcessingStatus',
    'LanguageCode',
    'ValidationSeverity'
]
