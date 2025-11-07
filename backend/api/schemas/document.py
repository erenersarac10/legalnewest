"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       DOCUMENT MANAGEMENT SCHEMAS                            ║
║                     Harvey/Legora World-Class Quality                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

BELGE YÖNETİMİ ŞEMALARI (DOCUMENT MANAGEMENT)
============================================

Bu modül, hukuki belge yönetimi, analiz, sürümleme ve paylaşım için gerekli tüm
Pydantic şemalarını içerir. Harvey ve Legora standartlarında kurumsal düzeyde
belge işleme, AI analizi, OCR, metadata extraction ve güvenli belge paylaşımı
özellikleri sağlar.

BÖLÜMLER
--------
1. Enums (Sabitler)
   - DocumentType: Belge tipleri (Mahkeme Kararı, Dilekçe, Sözleşme, vb.)
   - ProcessingStatus: İşlem durumu (UPLOADED → PROCESSING → COMPLETED)
   - AccessLevel: Erişim seviyesi (PRIVATE, TENANT, PUBLIC)
   - DocumentCategory: Belge kategorisi (LITIGATION, CONTRACT, COMPLIANCE, vb.)

2. Document CRUD Schemas
   - DocumentCreate: Yeni belge yükleme
   - DocumentUpdate: Belge güncelleme
   - DocumentMetadataUpdate: Metadata güncelleme

3. Response Schemas
   - DocumentResponse: Temel belge bilgileri
   - DocumentDetailResponse: Detaylı belge bilgileri + analiz sonuçları
   - DocumentListResponse: Sayfalı belge listesi

4. Document Analysis
   - AnalysisRequest: AI analiz talebi
   - AnalysisResponse: Analiz sonuçları (özet, varlıklar, tarihler)
   - DocumentSummary: Belge özeti
   - ExtractedEntity: Çıkarılan varlıklar (kişi, kurum, mahkeme, tarih)

5. Versioning & History
   - DocumentVersion: Belge sürümü
   - VersionHistory: Sürüm geçmişi
   - VersionCompare: İki sürüm karşılaştırması

6. Sharing & Collaboration
   - DocumentShareRequest: Belge paylaşım talebi
   - DocumentAccessGrant: Erişim izni
   - CollaboratorInfo: Ortak çalışan bilgisi
   - ShareLink: Paylaşım linki

7. Document Processing
   - OCRResult: OCR sonuçları
   - TextExtraction: Metin çıkarma
   - MetadataExtraction: Otomatik metadata çıkarma

8. Filter & Search
   - DocumentFilterParams: Belge filtreleme
   - DocumentSearchRequest: Gelişmiş arama
   - SearchResult: Arama sonucu

TÜRK HUKUK SİSTEMİNDE BELGE TİPLERİ
----------------------------------

Mahkeme Belgeleri (Court Documents):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. COURT_DECISION (Mahkeme Kararı)
   - İlk derece mahkeme kararları
   - Temyiz/istinaf kararları
   - Kesinleşmiş kararlar
   - Örnek: "Ankara 5. Asliye Hukuk Mahkemesi 2024/123 Esas Sayılı Karar"

2. PETITION (Dilekçe/İddia)
   - Dava dilekçesi
   - Cevap dilekçesi
   - Temyiz dilekçesi
   - İstinaf dilekçesi
   - Örnek: "İş Akdinin İptali Talebi Hakkında Dava Dilekçesi"

3. COURT_RULING (Ara Karar/Usul Kararı)
   - Bilirkişi tayini
   - Tanık dinleme
   - Delil tespiti
   - Duruşma ertelemesi

4. INDICTMENT (İddianame)
   - Ceza davaları için savcılık iddianamesi
   - Suç tasnifi, deliller, talep edilen ceza

İcra/İflas Belgeleri (Enforcement Documents):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
5. EXECUTION_ORDER (İcra Emri)
   - Ödeme emri
   - Haciz kararı
   - İhtiyati tedbir
   - Taşınmaz satış kararı

6. ENFORCEMENT_FILE (İcra Takip Dosyası)
   - İlamlı/ilamsız icra
   - Takip tarihi, borç tutarı
   - Haciz tutanakları

Sözleşmeler (Contracts):
~~~~~~~~~~~~~~~~~~~~~~
7. CONTRACT (Sözleşme)
   - İş sözleşmesi
   - Kira sözleşmesi
   - Alım-satım sözleşmesi
   - Hizmet sözleşmesi
   - Gizlilik sözleşmesi (NDA)

8. AGREEMENT (Protokol/Mutabakat)
   - Taraflar arası anlaşma
   - Uzlaşma tutanağı
   - Protokol

Resmi Belgeler (Official Documents):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
9. NOTARIZED_DOCUMENT (Noterde Tanzim)
   - Noter onaylı sözleşmeler
   - Vekaletnameler
   - Beyannameler

10. POWER_OF_ATTORNEY (Vekaletname)
    - Genel vekaletname
    - Özel vekaletname
    - Avukatlık vekaleti

11. OFFICIAL_LETTER (Resmi Yazı)
    - Kamu kurum yazıları
    - Bakanlık yazıları
    - Belediye yazıları

Danışmanlık Belgeleri (Advisory Documents):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
12. LEGAL_OPINION (Hukuki Görüş)
    - Danışmanlık mektubu
    - İnceleme raporu
    - Hukuki değerlendirme

13. MEMO (Hukuki Not/Memo)
    - İç yazışmalar
    - Dosya notları
    - Duruşma notları

Diğer Belgeler (Other Documents):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
14. EVIDENCE (Delil)
    - Fatura, makbuz
    - Fotoğraf, video
    - E-posta yazışmaları
    - Ses kayıtları

15. OTHER (Diğer)
    - Kategorize edilmemiş belgeler

BELGE İŞLEME PIPELINE'I (DOCUMENT PROCESSING)
--------------------------------------------

1. Upload (Yükleme):
   ↓ File upload (PDF, DOCX, image)
   ↓ Virus scan (ClamAV)
   ↓ File validation (size, type)
   ↓ Storage (S3/MinIO)

2. OCR & Text Extraction:
   ↓ PDF text extraction (PyPDF2)
   ↓ OCR for images (Tesseract Turkish)
   ↓ DOCX parsing (python-docx)
   ↓ Text normalization

3. AI Analysis:
   ↓ Document classification (ML model)
   ↓ Entity extraction (NER: kişi, kurum, tarih, tutar)
   ↓ Key phrase extraction
   ↓ Sentiment analysis
   ↓ Summarization (LLM: GPT-4/Claude)

4. Metadata Extraction:
   ↓ Automatic metadata detection
   ↓ Document date extraction
   ↓ Case number detection (2024/123 Esas)
   ↓ Court/Institution detection

5. Indexing:
   ↓ Full-text search indexing (Elasticsearch)
   ↓ Vector embedding (semantic search)
   ↓ Metadata indexing (PostgreSQL)

6. Completion:
   ↓ Status: COMPLETED
   ↓ Notification sent
   ↓ Ready for user access

Processing Status Lifecycle:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
UPLOADED → SCANNING → EXTRACTING → ANALYZING → INDEXING → COMPLETED
                ↓                      ↓
              FAILED               FAILED

ERİŞİM KONTROLÜ & GÜVENLİK (ACCESS CONTROL & SECURITY)
-----------------------------------------------------

Erişim Seviyeleri (Access Levels):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. PRIVATE (Özel)
   - Sadece belge sahibi erişebilir
   - Yükleyen kullanıcı + Admin

2. TENANT (Kiracı)
   - Tenant içindeki tüm yetkili kullanıcılar erişebilir
   - Role-based access control (RBAC)
   - Örnek: Hukuk bürosu içindeki tüm avukatlar

3. SHARED (Paylaşılmış)
   - Belirli kullanıcılarla paylaşılmış
   - Paylaşım linki ile erişim
   - Zaman sınırlı paylaşım

4. PUBLIC (Genel)
   - Kamuya açık içtihatlar
   - Yargıtay kararları
   - Mevzuat metinleri

Belge Güvenliği:
~~~~~~~~~~~~~~
✓ Encryption at rest: AES-256 (S3 server-side)
✓ Encryption in transit: TLS 1.3
✓ Access logging: Her erişim audit edilir
✓ Watermarking: PDF watermark (opsiyonel)
✓ DLP (Data Loss Prevention): Hassas veri tespiti
✓ Virus scanning: Her upload ClamAV taraması
✓ File size limits: Plan bazlı (FREE: 10MB, ENTERPRISE: 500MB)

SÜRÜMLEME (VERSIONING)
----------------------

Her belge güncellendiğinde yeni bir sürüm oluşturulur:

Version History:
~~~~~~~~~~~~~~
v1.0 (2024-01-15 10:30) - İlk yükleme (Ahmet Yılmaz)
  ↓ "Dava dilekçesi ilk taslak"
v1.1 (2024-01-16 14:20) - Güncelleme (Ayşe Kaya)
  ↓ "Maddi olay kısmı genişletildi"
v1.2 (2024-01-17 09:45) - Güncelleme (Ahmet Yılmaz)
  ↓ "Hukuki dayanak eklendi"
v2.0 (2024-01-20 16:00) - Major update (Ahmet Yılmaz)
  ↓ "Final versiyon, mahkemeye sunuldu"

Sürüm Özellikleri:
~~~~~~~~~~~~~~~~
- Otomatik sürüm numaralandırma (semantic versioning)
- Sürümler arası diff (karşılaştırma)
- Geri alma (rollback) desteği
- Sürüm etiketleri (draft, final, submitted)
- S3 versioning entegrasyonu

KVKK UYUMLULUĞU (TURKISH GDPR COMPLIANCE)
-----------------------------------------

Belgeler hassas kişisel veri içerebilir:

Hassas Veriler:
~~~~~~~~~~~~~
1. Kimlik Verisi:
   - TC Kimlik No
   - Pasaport no
   - Ehliyet no

2. İletişim Verisi:
   - Adres bilgileri
   - Telefon numaraları
   - E-posta adresleri

3. Mali Veri:
   - Banka hesap numaraları
   - Kredi kartı bilgileri
   - Gelir bilgileri

4. Hassas Veri (Özel Nitelikli):
   - Sağlık bilgileri
   - Suç kayıtları
   - Siyasi görüşler

KVKK İşleme Amaçları:
~~~~~~~~~~~~~~~~~~~
1. Hukuki Hizmet Sunumu: Dava takibi, danışmanlık
2. Yasal Yükümlülük: Avukatlık Kanunu, AAÜT
3. Veri Sahibinin Açık Rızası: Müvekkil onayı

Veri Saklama Süreleri:
~~~~~~~~~~~~~~~~~~~~
- Aktif dosyalar: Dava süresi boyunca
- Kapatılmış dosyalar: 10 yıl (Avukatlık Kanunu)
- Ceza davaları: 20 yıl veya zamanaşımı
- Silme talebi: 30 gün içinde soft delete
- Kesin silme: 1 yıl sonra hard delete

Veri Sahibi Hakları:
~~~~~~~~~~~~~~~~~~
1. Erişim: GET /api/v1/documents/{id}
2. Düzeltme: PATCH /api/v1/documents/{id}
3. Silme: DELETE /api/v1/documents/{id}
4. İtiraz: support@example.com
5. Kısıtlama: access_level değiştirme

PERFORMANS & ÖLÇEKLENDİRME
-------------------------

Depolama Stratejisi:
~~~~~~~~~~~~~~~~~~
- S3/MinIO: Belge dosyaları (object storage)
- PostgreSQL: Metadata, version history
- Elasticsearch: Full-text search, filtering
- Redis: Caching (document metadata, TTL 10 min)

Dosya Boyutu Limitleri:
~~~~~~~~~~~~~~~~~~~~~
- FREE plan: 10 MB per file, 500 MB total
- STARTER plan: 25 MB per file, 10 GB total
- PROFESSIONAL plan: 100 MB per file, 100 GB total
- ENTERPRISE plan: 500 MB per file, 1 TB+ total

CDN & Delivery:
~~~~~~~~~~~~~
- CloudFront: PDF preview delivery
- Presigned URLs: Güvenli download (expiry: 1 hour)
- Thumbnail generation: 200x200 preview images
- PDF.js: Browser-based PDF viewer

Background Jobs:
~~~~~~~~~~~~~~
- OCR processing: Celery worker
- AI analysis: Dedicated GPU worker
- Thumbnail generation: ImageMagick
- Virus scanning: Async ClamAV scan
- Index updates: Elasticsearch bulk indexing

Monitoring:
~~~~~~~~~
- Upload success rate: >99.5%
- Processing time: <30 seconds (avg)
- OCR accuracy: >95% (Turkish)
- Storage usage: Per-tenant tracking
- Failed uploads: Alert + retry

ENTEGRASYONLAR (INTEGRATIONS)
-----------------------------

Storage Backends:
~~~~~~~~~~~~~~~
- AWS S3: Primary storage (production)
- MinIO: Self-hosted S3-compatible (on-premise)
- Azure Blob: Alternative cloud storage
- Google Cloud Storage: Multi-cloud strategy

AI/ML Services:
~~~~~~~~~~~~~
- OpenAI GPT-4: Document summarization, Q&A
- Anthropic Claude: Legal document analysis
- Hugging Face: Custom NER models (Turkish legal entities)
- spaCy: Turkish NLP pipeline
- Tesseract OCR: Image to text (Turkish language pack)

Search Engines:
~~~~~~~~~~~~~
- Elasticsearch: Full-text search, filtering, aggregations
- Meilisearch: Lightweight alternative
- PostgreSQL Full-Text: Simple search fallback

Document Processing:
~~~~~~~~~~~~~~~~~~
- Apache Tika: Universal document parser
- PyPDF2/pdfplumber: PDF text extraction
- python-docx: DOCX parsing
- Pillow: Image processing
- pdf2image: PDF to image conversion

Security:
~~~~~~~
- ClamAV: Virus/malware scanning
- YARA: Custom malware rules
- Hashicorp Vault: Encryption key management
- AWS KMS: Key management service

ÖRNEKLER & KULLANIM (USAGE EXAMPLES)
-----------------------------------

1. Belge Yükleme (Document Upload):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.document import DocumentCreate, DocumentType, AccessLevel

# Mahkeme kararı yükleme
doc_data = DocumentCreate(
    title="Ankara 5. Asliye Hukuk Mahkemesi Kararı",
    document_type=DocumentType.COURT_DECISION,
    category=DocumentCategory.LITIGATION,
    access_level=AccessLevel.TENANT,
    tags=["mahkeme-kararı", "iş-hukuku", "kıdem-tazminatı"],
    case_number="2024/123 Esas, 2024/456 Karar",
    court_name="Ankara 5. Asliye Hukuk Mahkemesi",
    decision_date="2024-01-15"
)

# File upload (multipart/form-data)
files = {"file": open("mahkeme_karari.pdf", "rb")}
response = await client.post("/api/v1/documents", data=doc_data.dict(), files=files)
# Processing başlar: UPLOADED → SCANNING → EXTRACTING → ANALYZING → COMPLETED
```

2. AI Analiz Talebi (AI Analysis Request):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.document import AnalysisRequest

analysis_request = AnalysisRequest(
    document_id="550e8400-e29b-41d4-a716-446655440000",
    analysis_types=["summary", "entities", "sentiment", "key_phrases"],
    language="tr"
)

response = await document_service.analyze(analysis_request)
print(response.summary)  # "Bu kararda mahkeme, işverenin haksız fesih nedeniyle..."
print(response.entities)  # [{"type": "PERSON", "text": "Ahmet Yılmaz", ...}]
```

3. Belge Paylaşımı (Document Sharing):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.document import DocumentShareRequest

share_request = DocumentShareRequest(
    document_id="550e8400-e29b-41d4-a716-446655440000",
    recipient_emails=["musteri@example.com"],
    access_level="read",
    expires_at=datetime.now() + timedelta(days=7),
    message="Mahkeme kararının nihai halini paylaşıyorum.",
    require_password=True,
    allow_download=True
)

share_link = await document_service.share(share_request)
# Email gönderilir, link: https://app.example.com/shared/abc123xyz
```

4. Sürüm Karşılaştırma (Version Compare):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.document import VersionCompare

compare = VersionCompare(
    document_id="550e8400-e29b-41d4-a716-446655440000",
    version_a="1.0",
    version_b="2.0"
)

diff = await document_service.compare_versions(compare)
print(diff.added_text)  # "Ek olarak, taraflar arasında..."
print(diff.removed_text)  # "Önceki anlaşma hükümsüzdür"
```

5. Gelişmiş Arama (Advanced Search):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.document import DocumentSearchRequest

search = DocumentSearchRequest(
    query="iş akdi feshi kıdem tazminatı",
    document_types=[DocumentType.COURT_DECISION],
    date_range={"start": "2023-01-01", "end": "2024-01-31"},
    courts=["Ankara 5. Asliye Hukuk Mahkemesi"],
    sort_by="relevance"
)

results = await search_service.search(search)
# Elasticsearch semantic search + keyword matching
```

SORUN GİDERME (TROUBLESHOOTING)
-------------------------------

Sık Karşılaşılan Sorunlar:
~~~~~~~~~~~~~~~~~~~~~~~~

1. "Upload failed: File size exceeds quota":
   - Çözüm: Plan upgrade veya dosya sıkıştırma
   - Limit: FREE (10MB), STARTER (25MB), PROFESSIONAL (100MB)

2. "OCR failed: Unsupported file format":
   - Çözüm: PDF, DOCX, PNG, JPG desteklenir
   - Diğer formatlar için PDF'e dönüştürme

3. "Processing stuck in ANALYZING status":
   - Çözüm: Celery worker kontrol, queue temizleme
   - Timeout: 5 dakika sonra FAILED

4. "Virus detected, upload rejected":
   - Çözüm: Dosyayı temizleyip yeniden yükle
   - Log: ClamAV scan report inceleme

5. "Access denied to shared document":
   - Çözüm: Link expiry kontrol, şifre doğrulama
   - Debug: access_grants tablosu kontrol

VERSİYON GEÇMİŞİ
---------------
- v1.0.0 (2024-01): Initial implementation
  * Document upload & storage
  * Basic OCR & text extraction
  * Metadata management

- v1.1.0 (2024-02): AI Analysis
  * GPT-4 summarization
  * NER for Turkish legal entities
  * Sentiment analysis

- v1.2.0 (2024-03): Versioning & Sharing
  * Version history tracking
  * Secure document sharing
  * Collaboration features

YETKİLENDİRME & LİSANS
---------------------
© 2024 LegalAI Platform
Enterprise-grade document management system
Harvey AI & Legora inspired architecture
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import (
    Field,
    EmailStr,
    field_validator,
    model_validator,
    HttpUrl,
    ConfigDict
)

from .base import (
    BaseSchema,
    RequestSchema,
    ResponseSchema,
    IdentifierSchema,
    TimestampSchema,
    TenantSchema,
    AuditSchema
)


# ============================================================================
# ENUMS (SABİTLER)
# ============================================================================

class DocumentType(str, Enum):
    """
    Belge tipleri (Türk hukuk sistemi).

    Attributes:
        COURT_DECISION: Mahkeme kararı
        PETITION: Dilekçe/İddia
        COURT_RULING: Ara karar/Usul kararı
        INDICTMENT: İddianame
        EXECUTION_ORDER: İcra emri
        ENFORCEMENT_FILE: İcra takip dosyası
        CONTRACT: Sözleşme
        AGREEMENT: Protokol/Mutabakat
        NOTARIZED_DOCUMENT: Noterde tanzim
        POWER_OF_ATTORNEY: Vekaletname
        OFFICIAL_LETTER: Resmi yazı
        LEGAL_OPINION: Hukuki görüş
        MEMO: Hukuki not/Memo
        EVIDENCE: Delil (fatura, makbuz, fotoğraf)
        OTHER: Diğer
    """
    COURT_DECISION = "court_decision"
    PETITION = "petition"
    COURT_RULING = "court_ruling"
    INDICTMENT = "indictment"
    EXECUTION_ORDER = "execution_order"
    ENFORCEMENT_FILE = "enforcement_file"
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    NOTARIZED_DOCUMENT = "notarized_document"
    POWER_OF_ATTORNEY = "power_of_attorney"
    OFFICIAL_LETTER = "official_letter"
    LEGAL_OPINION = "legal_opinion"
    MEMO = "memo"
    EVIDENCE = "evidence"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """
    Belge işleme durumu (pipeline status).

    Status Lifecycle:
        UPLOADED → SCANNING → EXTRACTING → ANALYZING → INDEXING → COMPLETED
                       ↓            ↓           ↓
                     FAILED       FAILED      FAILED

    Attributes:
        UPLOADED: Yüklendi, işlem bekliyor
        SCANNING: Virus taraması yapılıyor
        EXTRACTING: Metin çıkarma (OCR) yapılıyor
        ANALYZING: AI analizi yapılıyor
        INDEXING: Arama indexi oluşturuluyor
        COMPLETED: İşlem tamamlandı
        FAILED: İşlem başarısız
    """
    UPLOADED = "uploaded"
    SCANNING = "scanning"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class AccessLevel(str, Enum):
    """
    Belge erişim seviyesi.

    Attributes:
        PRIVATE: Sadece belge sahibi
        TENANT: Tenant içindeki yetkili kullanıcılar
        SHARED: Belirli kullanıcılarla paylaşılmış
        PUBLIC: Kamuya açık (içtihatlar, mevzuat)
    """
    PRIVATE = "private"
    TENANT = "tenant"
    SHARED = "shared"
    PUBLIC = "public"


class DocumentCategory(str, Enum):
    """
    Belge kategorisi (high-level classification).

    Attributes:
        LITIGATION: Dava/Uyuşmazlık belgeleri
        CONTRACT: Sözleşmeler
        COMPLIANCE: Uyum/Compliance belgeleri
        ADVISORY: Danışmanlık belgeleri
        ENFORCEMENT: İcra/İflas belgeleri
        CORPORATE: Kurumsal belgeler
        EVIDENCE: Deliller
        OTHER: Diğer
    """
    LITIGATION = "litigation"
    CONTRACT = "contract"
    COMPLIANCE = "compliance"
    ADVISORY = "advisory"
    ENFORCEMENT = "enforcement"
    CORPORATE = "corporate"
    EVIDENCE = "evidence"
    OTHER = "other"


# ============================================================================
# DOCUMENT CRUD SCHEMAS
# ============================================================================

class DocumentCreate(RequestSchema, TenantSchema):
    """
    Yeni belge yükleme şeması.

    Multipart/form-data ile dosya + metadata gönderilir.

    Attributes:
        title: Belge başlığı
        document_type: Belge tipi
        category: Belge kategorisi
        access_level: Erişim seviyesi (varsayılan: PRIVATE)
        description: Belge açıklaması (opsiyonel)
        tags: Etiketler (arama için)
        case_number: Dava/Dosya numarası (opsiyonel)
        court_name: Mahkeme adı (opsiyonel)
        decision_date: Karar tarihi (opsiyonel)
        related_party_names: İlgili taraf isimleri
        auto_analyze: Otomatik AI analizi yap (varsayılan: True)

    Validation:
        - title: 3-500 karakter
        - tags: Maksimum 10 etiket
        - case_number: Türk mahkeme dosya formatı (2024/123 Esas)

    Example:
        >>> doc = DocumentCreate(
        ...     title="Ankara 5. Asliye Hukuk Mahkemesi Kararı",
        ...     document_type=DocumentType.COURT_DECISION,
        ...     category=DocumentCategory.LITIGATION,
        ...     access_level=AccessLevel.TENANT,
        ...     tags=["mahkeme", "iş-hukuku"],
        ...     case_number="2024/123 Esas",
        ...     court_name="Ankara 5. Asliye Hukuk Mahkemesi"
        ... )
    """
    title: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Belge başlığı"
    )
    document_type: DocumentType = Field(
        ...,
        description="Belge tipi"
    )
    category: DocumentCategory = Field(
        ...,
        description="Belge kategorisi"
    )
    access_level: AccessLevel = Field(
        default=AccessLevel.PRIVATE,
        description="Erişim seviyesi"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Belge açıklaması"
    )
    tags: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Etiketler (maksimum 10)"
    )
    case_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Dava/Dosya numarası (örn: 2024/123 Esas)"
    )
    court_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Mahkeme adı"
    )
    decision_date: Optional[datetime] = Field(
        None,
        description="Karar/Belge tarihi"
    )
    related_party_names: List[str] = Field(
        default_factory=list,
        description="İlgili taraf isimleri"
    )
    auto_analyze: bool = Field(
        default=True,
        description="Otomatik AI analizi yap"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Etiket validasyonu: küçük harf, maksimum 10."""
        if len(v) > 10:
            raise ValueError("Maksimum 10 etiket eklenebilir")
        return [tag.lower().strip() for tag in v if tag.strip()]

    @field_validator("title", "description")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """Whitespace temizleme."""
        if v:
            return v.strip()
        return v


class DocumentUpdate(RequestSchema):
    """
    Belge güncelleme şeması.

    Tüm alanlar opsiyonel.

    Attributes:
        title: Yeni başlık
        document_type: Yeni belge tipi
        category: Yeni kategori
        access_level: Yeni erişim seviyesi
        description: Yeni açıklama
        tags: Yeni etiketler
        case_number: Yeni dava numarası
        court_name: Yeni mahkeme
        decision_date: Yeni tarih
    """
    title: Optional[str] = Field(
        None,
        min_length=3,
        max_length=500,
        description="Belge başlığı"
    )
    document_type: Optional[DocumentType] = Field(
        None,
        description="Belge tipi"
    )
    category: Optional[DocumentCategory] = Field(
        None,
        description="Belge kategorisi"
    )
    access_level: Optional[AccessLevel] = Field(
        None,
        description="Erişim seviyesi"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Belge açıklaması"
    )
    tags: Optional[List[str]] = Field(
        None,
        max_length=10,
        description="Etiketler"
    )
    case_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Dava numarası"
    )
    court_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Mahkeme adı"
    )
    decision_date: Optional[datetime] = Field(
        None,
        description="Karar tarihi"
    )


class DocumentMetadataUpdate(RequestSchema):
    """
    Sadece metadata güncelleme şeması.

    Dosya değişmeden metadata güncellemesi için kullanılır.

    Attributes:
        custom_metadata: Özel metadata alanları (JSON)
    """
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Özel metadata (key-value pairs)"
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class DocumentResponse(ResponseSchema, IdentifierSchema, TimestampSchema, TenantSchema):
    """
    Temel belge yanıt şeması.

    Listeleme için kullanılır.

    Attributes:
        id: Belge UUID
        title: Başlık
        document_type: Belge tipi
        category: Kategori
        access_level: Erişim seviyesi
        file_name: Dosya adı
        file_size_bytes: Dosya boyutu (bytes)
        file_extension: Dosya uzantısı (.pdf, .docx)
        processing_status: İşlem durumu
        thumbnail_url: Önizleme resmi URL
        created_by: Yükleyen kullanıcı UUID
        created_at: Yükleme zamanı
        updated_at: Güncelleme zamanı
    """
    title: str = Field(..., description="Belge başlığı")
    document_type: DocumentType = Field(..., description="Belge tipi")
    category: DocumentCategory = Field(..., description="Kategori")
    access_level: AccessLevel = Field(..., description="Erişim seviyesi")
    file_name: str = Field(..., description="Dosya adı")
    file_size_bytes: int = Field(..., ge=0, description="Dosya boyutu")
    file_extension: str = Field(..., description="Dosya uzantısı")
    processing_status: ProcessingStatus = Field(..., description="İşlem durumu")
    thumbnail_url: Optional[str] = Field(None, description="Önizleme URL")
    created_by: UUID = Field(..., description="Yükleyen kullanıcı")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_id": "650e8400-e29b-41d4-a716-446655440001",
                "title": "Ankara 5. Asliye Hukuk Mahkemesi Kararı",
                "document_type": "court_decision",
                "category": "litigation",
                "access_level": "tenant",
                "file_name": "mahkeme_karari_2024_123.pdf",
                "file_size_bytes": 2457600,
                "file_extension": ".pdf",
                "processing_status": "completed",
                "thumbnail_url": "https://cdn.example.com/thumbnails/550e8400.jpg",
                "created_by": "750e8400-e29b-41d4-a716-446655440002",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }
    )


class DocumentDetailResponse(DocumentResponse, AuditSchema):
    """
    Detaylı belge yanıt şeması.

    Tek belge görüntüleme için kullanılır.

    Attributes:
        (DocumentResponse'dan tüm alanlar)
        description: Belge açıklaması
        tags: Etiketler
        case_number: Dava numarası
        court_name: Mahkeme adı
        decision_date: Karar tarihi
        related_party_names: İlgili taraf isimleri
        extracted_text: OCR ile çıkarılan metin (ilk 1000 karakter)
        summary: AI özeti (eğer analiz yapıldıysa)
        page_count: Sayfa sayısı
        word_count: Kelime sayısı
        download_url: İndirme URL (presigned)
        view_url: Görüntüleme URL (PDF.js viewer)
        version: Mevcut sürüm numarası
        version_count: Toplam sürüm sayısı
        custom_metadata: Özel metadata
    """
    description: Optional[str] = Field(None, description="Belge açıklaması")
    tags: List[str] = Field(default_factory=list, description="Etiketler")
    case_number: Optional[str] = Field(None, description="Dava numarası")
    court_name: Optional[str] = Field(None, description="Mahkeme adı")
    decision_date: Optional[datetime] = Field(None, description="Karar tarihi")
    related_party_names: List[str] = Field(default_factory=list, description="İlgili taraflar")
    extracted_text: Optional[str] = Field(None, description="Çıkarılan metin (preview)")
    summary: Optional[str] = Field(None, description="AI özeti")
    page_count: Optional[int] = Field(None, ge=0, description="Sayfa sayısı")
    word_count: Optional[int] = Field(None, ge=0, description="Kelime sayısı")
    download_url: Optional[str] = Field(None, description="İndirme URL")
    view_url: Optional[str] = Field(None, description="Görüntüleme URL")
    version: str = Field(..., description="Mevcut sürüm")
    version_count: int = Field(..., ge=1, description="Toplam sürüm sayısı")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Özel metadata")


# ============================================================================
# DOCUMENT ANALYSIS
# ============================================================================

class AnalysisRequest(RequestSchema):
    """
    AI analiz talebi şeması.

    Attributes:
        document_id: Belge UUID
        analysis_types: Analiz tipleri (summary, entities, sentiment, key_phrases)
        language: Dil kodu (tr, en)
        model: AI model (gpt-4, claude, custom)
    """
    document_id: UUID = Field(..., description="Belge UUID")
    analysis_types: List[str] = Field(
        default=["summary", "entities"],
        description="Analiz tipleri"
    )
    language: str = Field(
        default="tr",
        pattern=r"^[a-z]{2}$",
        description="Dil kodu"
    )
    model: str = Field(
        default="gpt-4",
        description="AI model"
    )


class ExtractedEntity(BaseSchema):
    """
    Çıkarılan varlık (NER entity).

    Attributes:
        type: Varlık tipi (PERSON, ORG, DATE, MONEY, COURT, CASE_NUMBER)
        text: Varlık metni
        confidence: Güven skoru (0-1)
        start_char: Başlangıç karakteri
        end_char: Bitiş karakteri
    """
    type: str = Field(..., description="Varlık tipi")
    text: str = Field(..., description="Varlık metni")
    confidence: float = Field(..., ge=0, le=1, description="Güven skoru")
    start_char: int = Field(..., ge=0, description="Başlangıç pozisyonu")
    end_char: int = Field(..., ge=0, description="Bitiş pozisyonu")


class DocumentSummary(BaseSchema):
    """
    Belge özeti.

    Attributes:
        short_summary: Kısa özet (2-3 cümle)
        detailed_summary: Detaylı özet (200-500 kelime)
        key_points: Ana noktalar (bullet points)
        sentiment: Duygu analizi (positive, neutral, negative)
        confidence: Özet güven skoru
    """
    short_summary: str = Field(..., max_length=500, description="Kısa özet")
    detailed_summary: str = Field(..., max_length=5000, description="Detaylı özet")
    key_points: List[str] = Field(default_factory=list, description="Ana noktalar")
    sentiment: str = Field(..., description="Duygu")
    confidence: float = Field(..., ge=0, le=1, description="Güven skoru")


class AnalysisResponse(ResponseSchema):
    """
    Analiz sonuçları.

    Attributes:
        document_id: Belge UUID
        summary: Belge özeti
        entities: Çıkarılan varlıklar
        key_phrases: Anahtar kelimeler
        processing_time_seconds: İşlem süresi
        model_used: Kullanılan model
        completed_at: Tamamlanma zamanı
    """
    document_id: UUID = Field(..., description="Belge UUID")
    summary: Optional[DocumentSummary] = Field(None, description="Özet")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Varlıklar")
    key_phrases: List[str] = Field(default_factory=list, description="Anahtar kelimeler")
    processing_time_seconds: float = Field(..., ge=0, description="İşlem süresi")
    model_used: str = Field(..., description="Kullanılan model")
    completed_at: datetime = Field(..., description="Tamamlanma zamanı")


# ============================================================================
# VERSIONING & HISTORY
# ============================================================================

class DocumentVersion(BaseSchema, IdentifierSchema, TimestampSchema):
    """
    Belge sürümü.

    Attributes:
        id: Sürüm UUID
        document_id: Ana belge UUID
        version: Sürüm numarası (1.0, 1.1, 2.0)
        file_url: Bu sürümün dosya URL'i
        file_size_bytes: Dosya boyutu
        change_summary: Değişiklik özeti
        created_by: Oluşturan kullanıcı
        created_at: Oluşturma zamanı
        is_current: Mevcut sürüm mü
    """
    document_id: UUID = Field(..., description="Ana belge UUID")
    version: str = Field(..., description="Sürüm numarası")
    file_url: str = Field(..., description="Dosya URL")
    file_size_bytes: int = Field(..., ge=0, description="Dosya boyutu")
    change_summary: Optional[str] = Field(None, max_length=500, description="Değişiklik özeti")
    created_by: UUID = Field(..., description="Oluşturan kullanıcı")
    is_current: bool = Field(..., description="Mevcut sürüm")


class VersionCompare(RequestSchema):
    """
    İki sürüm karşılaştırması talebi.

    Attributes:
        document_id: Belge UUID
        version_a: İlk sürüm
        version_b: İkinci sürüm
    """
    document_id: UUID = Field(..., description="Belge UUID")
    version_a: str = Field(..., description="İlk sürüm")
    version_b: str = Field(..., description="İkinci sürüm")


# ============================================================================
# SHARING & COLLABORATION
# ============================================================================

class DocumentShareRequest(RequestSchema):
    """
    Belge paylaşım talebi.

    Attributes:
        document_id: Belge UUID
        recipient_emails: Alıcı e-posta adresleri
        access_level: Erişim seviyesi (read, write)
        expires_at: Son kullanma tarihi
        message: Paylaşım mesajı
        require_password: Şifre gereksin mi
        allow_download: İndirmeye izin ver mi
    """
    document_id: UUID = Field(..., description="Belge UUID")
    recipient_emails: List[EmailStr] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Alıcı e-posta adresleri (max 10)"
    )
    access_level: str = Field(
        default="read",
        pattern=r"^(read|write)$",
        description="Erişim seviyesi"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Son kullanma tarihi"
    )
    message: Optional[str] = Field(
        None,
        max_length=500,
        description="Paylaşım mesajı"
    )
    require_password: bool = Field(
        default=False,
        description="Şifre gereksin mi"
    )
    allow_download: bool = Field(
        default=True,
        description="İndirmeye izin ver"
    )


class ShareLink(BaseSchema, IdentifierSchema):
    """
    Paylaşım linki bilgileri.

    Attributes:
        id: Link UUID
        document_id: Belge UUID
        token: Paylaşım token'ı
        share_url: Tam URL
        access_level: Erişim seviyesi
        expires_at: Son kullanma tarihi
        password_required: Şifre gerekli mi
        access_count: Erişim sayısı
        created_by: Oluşturan kullanıcı
        created_at: Oluşturma zamanı
    """
    document_id: UUID = Field(..., description="Belge UUID")
    token: str = Field(..., description="Paylaşım token'ı")
    share_url: HttpUrl = Field(..., description="Paylaşım URL")
    access_level: str = Field(..., description="Erişim seviyesi")
    expires_at: Optional[datetime] = Field(None, description="Son kullanma")
    password_required: bool = Field(..., description="Şifre gerekli")
    access_count: int = Field(..., ge=0, description="Erişim sayısı")
    created_by: UUID = Field(..., description="Oluşturan")
    created_at: datetime = Field(..., description="Oluşturma zamanı")


# ============================================================================
# FILTER & SEARCH
# ============================================================================

class DocumentFilterParams(BaseSchema):
    """
    Belge filtreleme parametreleri.

    Attributes:
        document_types: Belge tipleri filtresi
        categories: Kategori filtresi
        access_levels: Erişim seviyeleri
        processing_statuses: İşlem durumları
        tags: Etiket filtresi
        search: Arama terimi (başlık, açıklama)
        created_by: Yükleyen kullanıcı filtresi
        date_from: Başlangıç tarihi
        date_to: Bitiş tarihi
        min_size_bytes: Minimum dosya boyutu
        max_size_bytes: Maksimum dosya boyutu
    """
    document_types: Optional[List[DocumentType]] = Field(
        None,
        description="Belge tipleri"
    )
    categories: Optional[List[DocumentCategory]] = Field(
        None,
        description="Kategoriler"
    )
    access_levels: Optional[List[AccessLevel]] = Field(
        None,
        description="Erişim seviyeleri"
    )
    processing_statuses: Optional[List[ProcessingStatus]] = Field(
        None,
        description="İşlem durumları"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Etiketler"
    )
    search: Optional[str] = Field(
        None,
        min_length=2,
        max_length=200,
        description="Arama terimi"
    )
    created_by: Optional[UUID] = Field(
        None,
        description="Yükleyen kullanıcı"
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Başlangıç tarihi"
    )
    date_to: Optional[datetime] = Field(
        None,
        description="Bitiş tarihi"
    )
    min_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Minimum dosya boyutu"
    )
    max_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Maksimum dosya boyutu"
    )


class DocumentSearchRequest(RequestSchema):
    """
    Gelişmiş belge arama şeması.

    Elasticsearch semantic search + keyword matching.

    Attributes:
        query: Arama sorgusu
        document_types: Belge tipleri
        date_range: Tarih aralığı
        courts: Mahkeme isimleri
        case_numbers: Dava numaraları
        sort_by: Sıralama (relevance, date, title)
        limit: Sonuç limiti
    """
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Arama sorgusu"
    )
    document_types: Optional[List[DocumentType]] = Field(
        None,
        description="Belge tipleri"
    )
    date_range: Optional[Dict[str, str]] = Field(
        None,
        description="Tarih aralığı (start, end)"
    )
    courts: Optional[List[str]] = Field(
        None,
        description="Mahkeme isimleri"
    )
    case_numbers: Optional[List[str]] = Field(
        None,
        description="Dava numaraları"
    )
    sort_by: str = Field(
        default="relevance",
        pattern=r"^(relevance|date|title)$",
        description="Sıralama kriteri"
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Sonuç limiti"
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "DocumentType",
    "ProcessingStatus",
    "AccessLevel",
    "DocumentCategory",
    # CRUD
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentMetadataUpdate",
    # Responses
    "DocumentResponse",
    "DocumentDetailResponse",
    # Analysis
    "AnalysisRequest",
    "AnalysisResponse",
    "DocumentSummary",
    "ExtractedEntity",
    # Versioning
    "DocumentVersion",
    "VersionCompare",
    # Sharing
    "DocumentShareRequest",
    "ShareLink",
    # Filter & Search
    "DocumentFilterParams",
    "DocumentSearchRequest",
]
