"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TENANT MANAGEMENT SCHEMAS                             ║
║                     Harvey/Legora World-Class Quality                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

TENANT (ÇOK KİRACILI MİMARİ) ŞEMALARI
=====================================

Bu modül, çok kiracılı (multi-tenant) SaaS platformunun tenant (kiracı) yönetimi için
gerekli tüm Pydantic şemalarını içerir. Harvey ve Legora standartlarında kurumsal
düzeyde veri doğrulama, abonelik yönetimi, kota kontrolü ve faturalama entegrasyonu
sağlar.

BÖLÜMLER
--------
1. Enums (Sabitler)
   - SubscriptionPlan: Abonelik planları (FREE → ENTERPRISE)
   - TenantStatus: Kiracı durumları (TRIAL → ACTIVE → SUSPENDED)
   - OrganizationType: Organizasyon tipleri (Hukuk Bürosu, Baro, vb.)
   - BillingCycle: Faturalama dönemleri (MONTHLY, YEARLY)
   - PaymentMethod: Ödeme yöntemleri (CREDIT_CARD, BANK_TRANSFER, vb.)

2. Tenant CRUD Schemas
   - TenantCreate: Yeni tenant oluşturma (kayıt)
   - TenantUpdate: Tenant güncelleme (admin)
   - TenantSettingsUpdate: Tenant ayarları güncelleme
   - TenantPreferencesUpdate: Tenant tercihleri güncelleme

3. Response Schemas
   - TenantResponse: Temel tenant bilgileri
   - TenantDetailResponse: Detaylı tenant bilgileri (quota, subscription)
   - TenantPublicInfo: Genel tenant bilgileri (domain bazlı erişim)
   - TenantStatistics: Tenant kullanım istatistikleri

4. Subscription Management
   - SubscriptionInfo: Abonelik durumu ve detayları
   - SubscriptionPlanUpdate: Plan değişikliği isteği
   - SubscriptionUpgradeRequest: Upgrade talebi
   - SubscriptionDowngradeRequest: Downgrade talebi
   - SubscriptionCancellationRequest: İptal talebi

5. Quota & Usage Management
   - TenantQuotaInfo: Kota bilgileri (users, storage, API calls)
   - TenantUsageStatistics: Kullanım metrikleri
   - QuotaExceededError: Kota aşım hatası
   - UsageReport: Periyodik kullanım raporu

6. Billing & Payment
   - BillingInfo: Faturalama bilgileri
   - PaymentMethodInfo: Ödeme yöntemi bilgileri
   - InvoiceInfo: Fatura bilgileri
   - PaymentHistoryEntry: Ödeme geçmişi kaydı

7. Organization Management
   - OrganizationInfo: Organizasyon detayları
   - OrganizationMemberInfo: Üye bilgileri
   - OrganizationInviteRequest: Üye davet isteği

8. Filter & List Schemas
   - TenantFilterParams: Tenant filtreleme parametreleri
   - TenantListResponse: Sayfalı tenant listesi

MULTİ-TENANCY MİMARİSİ
----------------------

Row-Level Security (RLS) Stratejisi:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
✓ PostgreSQL RLS policies ile tenant isolation
✓ Her sorguya otomatik tenant_id filtreleme
✓ Veri sızıntısı (data leakage) koruması
✓ Cross-tenant erişim engellemesi

Tenant Hierarchy (Kiracı Hiyerarşisi):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TRIAL (Deneme)
  ↓ 14 gün deneme süresi
  ↓ Sınırlı özellikler (5 kullanıcı, 1GB depolama)

FREE (Ücretsiz)
  ↓ Süresiz kullanım
  ↓ Temel özellikler (3 kullanıcı, 500MB)

STARTER (Başlangıç)
  ↓ Küçük hukuk büroları için
  ↓ 10 kullanıcı, 10GB, temel AI özellikleri

PROFESSIONAL (Profesyonel)
  ↓ Orta ölçekli kurumlar için
  ↓ 50 kullanıcı, 100GB, gelişmiş AI + entegrasyonlar

ENTERPRISE (Kurumsal)
  ↓ Büyük kurumlar ve barolar için
  ↓ Sınırsız kullanıcı, özel depolama, SSO, SLA

CUSTOM (Özel)
  ↓ Özelleştirilmiş çözümler
  ↓ Özel fiyatlandırma ve özellikler

Subscription Lifecycle (Abonelik Yaşam Döngüsü):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. TRIAL (Deneme) → 14 gün
2. ACTIVE (Aktif) → Ödeme başarılı
3. PAST_DUE (Gecikmiş) → Ödeme başarısız, 7 gün ek süre
4. SUSPENDED (Askıya Alındı) → 7 gün sonra otomatik askıya alma
5. CANCELLED (İptal Edildi) → Kullanıcı iptali veya manuel müdahale
6. EXPIRED (Süresi Doldu) → Deneme süresi bitti, downgrade to FREE

KOTA YÖNETİMİ (QUOTA MANAGEMENT)
--------------------------------

Abonelik Planlarına Göre Kotalar:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FREE Plan Kotaları:
- users: 3 kullanıcı
- storage_gb: 0.5 GB (500 MB)
- api_calls_per_month: 1,000
- ai_queries_per_month: 50
- documents_per_month: 20
- chat_messages_per_day: 10
- features: ["basic_chat", "document_upload"]

STARTER Plan Kotaları:
- users: 10 kullanıcı
- storage_gb: 10 GB
- api_calls_per_month: 10,000
- ai_queries_per_month: 500
- documents_per_month: 200
- chat_messages_per_day: 100
- features: ["basic_chat", "document_upload", "document_analysis", "templates"]

PROFESSIONAL Plan Kotaları:
- users: 50 kullanıcı
- storage_gb: 100 GB
- api_calls_per_month: 100,000
- ai_queries_per_month: 5,000
- documents_per_month: 2,000
- chat_messages_per_day: 1,000
- features: [all STARTER + "advanced_ai", "integrations", "api_access", "webhooks"]

ENTERPRISE Plan Kotaları:
- users: Sınırsız
- storage_gb: 1,000 GB (1 TB) veya özel
- api_calls_per_month: Sınırsız
- ai_queries_per_month: 50,000
- documents_per_month: Sınırsız
- chat_messages_per_day: Sınırsız
- features: [all PROFESSIONAL + "sso", "ldap", "custom_models", "dedicated_support", "sla"]

Kota Aşımı Politikası:
~~~~~~~~~~~~~~~~~~~~
1. Soft Limit: %80 kullanımda warning notification
2. Hard Limit: %100'de işlem engelleme
3. Grace Period: 3 gün boyunca upgrade önerisi
4. Auto-downgrade: 7 gün sonra otomatik plan düşürme (ENTERPRISE hariç)

FATURALAMA & ÖDEME (BILLING & PAYMENT)
--------------------------------------

Faturalama Döngüleri:
~~~~~~~~~~~~~~~~~~~
- MONTHLY: Aylık faturalama (varsayılan)
- YEARLY: Yıllık faturalama (%20 indirim)
- QUARTERLY: 3 aylık faturalama (%10 indirim)

Ödeme Yöntemleri:
~~~~~~~~~~~~~~~~
- CREDIT_CARD: Kredi kartı (Stripe/Iyzico)
- BANK_TRANSFER: Banka havalesi (manuel onay)
- INVOICE: Fatura ile ödeme (ENTERPRISE plan)

Ödeme Politikası:
~~~~~~~~~~~~~~~
1. Otomatik yenileme: Abonelik bitiş tarihinden 1 gün önce
2. Başarısız ödeme: 3 retry attempt (1 gün arayla)
3. Past Due: 7 gün ek süre, sonra SUSPENDED
4. İade politikası: İlk 7 gün %100 iade

Fatura Özellikleri:
~~~~~~~~~~~~~~~~~
- E-fatura entegrasyonu (Türkiye)
- PDF fatura oluşturma
- Otomatik e-posta gönderimi
- KDV hesaplama (%20 Türkiye için)
- Vergi kimlik no doğrulama

ORGANİZASYON TİPLERİ (TURKISH LEGAL SYSTEM)
-------------------------------------------

Desteklenen Organizasyon Tipleri:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. LAW_FIRM (Hukuk Bürosu)
   - Bireysel veya ortaklık avukatlık büroları
   - Müvekkil yönetimi, dosya takibi
   - Örnek: Ankara Hukuk Bürosu

2. BAR_ASSOCIATION (Baro)
   - İl baroları
   - Üye avukat yönetimi, staj takibi
   - Örnek: İstanbul Barosu

3. CORPORATE_LEGAL (Şirket Hukuk Departmanı)
   - Şirket içi hukuk birimleri
   - Sözleşme yönetimi, compliance
   - Örnek: Koç Holding Hukuk Müşavirliği

4. COURT (Mahkeme)
   - Adliyeler, mahkemeler
   - Dava dosyası yönetimi
   - Örnek: Ankara 5. Asliye Hukuk Mahkemesi

5. PUBLIC_INSTITUTION (Kamu Kurumu)
   - Bakanlıklar, belediyeler
   - Mevzuat takibi, danışmanlık
   - Örnek: Adalet Bakanlığı

6. LEGAL_CONSULTANCY (Hukuk Danışmanlık)
   - Bağımsız danışmanlık firmaları
   - Compliance, risk yönetimi
   - Örnek: Legal Advisors Turkey

7. NOTARY (Noterlik)
   - Noter ofisleri
   - Belge tasdik, sözleşme düzenleme
   - Örnek: İstanbul 25. Noterliği

8. INDIVIDUAL (Bireysel)
   - Tekil kullanıcılar
   - Kişisel hukuki takip
   - Örnek: Bireysel avukat veya vatandaş

KVKK UYUMLULUĞU (TURKISH GDPR COMPLIANCE)
-----------------------------------------

Veri Kategorileri:
~~~~~~~~~~~~~~~~
1. Kimlik Verisi:
   - tenant_name: Tenant adı (şirket/baro adı)
   - tax_number: Vergi kimlik numarası (10 haneli)
   - contact_email: İletişim e-posta
   - contact_phone: İletişim telefonu

2. Abonelik Verisi:
   - subscription_plan: Abonelik planı
   - billing_info: Faturalama bilgileri
   - payment_methods: Ödeme yöntemleri

3. Kullanım Verisi:
   - usage_statistics: Kullanım metrikleri
   - quota_info: Kota bilgileri
   - audit_logs: Denetim kayıtları

İşleme Amaçları:
~~~~~~~~~~~~~~
1. Hizmet Sunumu: Platform erişimi ve özellikler
2. Faturalama: Ödeme işlemleri ve fatura kesimi
3. Destek: Teknik destek ve müşteri hizmetleri
4. İyileştirme: Kullanım analizleri (anonim)
5. Yasal Yükümlülük: E-fatura, vergi mevzuatı

Hukuki Dayanak:
~~~~~~~~~~~~~
- Sözleşmenin İfası (KVKK m.5/2-c): Abonelik sözleşmesi
- Hukuki Yükümlülük (KVKK m.5/2-ç): E-fatura, vergi kayıtları
- Meşru Menfaat (KVKK m.5/2-f): Platform güvenliği, dolandırıcılık önleme

Veri Saklama Süreleri:
~~~~~~~~~~~~~~~~~~~~
- Aktif tenant verileri: Abonelik süresi boyunca
- İptal edilmiş tenant: 30 gün (veri export süresi)
- Fatura kayıtları: 10 yıl (Vergi Usul Kanunu)
- Ödeme kayıtları: 10 yıl (Mali mevzuat)
- Kullanım logları: 2 yıl (anonim)

Veri Sahibi Hakları:
~~~~~~~~~~~~~~~~~~
1. Bilgi Talep Etme: GET /api/v1/tenants/{id}/data-export
2. Düzeltme: PATCH /api/v1/tenants/{id}
3. Silme: DELETE /api/v1/tenants/{id} (soft delete, 30 gün)
4. İtiraz: support@example.com üzerinden
5. Veri Taşınabilirliği: JSON/CSV export

Üçüncü Taraf Paylaşımı:
~~~~~~~~~~~~~~~~~~~~~
- Ödeme işlemcileri: Stripe (US), Iyzico (TR)
- E-fatura: GİB entegrasyonu
- Cloud depolama: AWS (Frankfurt region)
- Analytics: Self-hosted (Plausible/Matomo)
- KVKK Aydınlatma Metni: /legal/kvkk-tenant

GÜVENLİK & PERFORMANS (SECURITY & PERFORMANCE)
----------------------------------------------

Güvenlik Önlemleri:
~~~~~~~~~~~~~~~~~
1. Tenant Isolation:
   - PostgreSQL RLS policies
   - Tenant ID validation middleware
   - Cross-tenant query prevention

2. Veri Şifreleme:
   - tax_number: AES-256 encryption at rest
   - payment_info: PCI DSS compliant storage
   - Sensitive fields: json_schema_extra={"sensitive": True}

3. Access Control:
   - Tenant admin: Full tenant management
   - Tenant owner: Billing + subscription
   - Tenant member: Read-only tenant info

4. Rate Limiting:
   - Tenant creation: 1/hour per IP
   - Subscription changes: 5/day per tenant
   - Quota checks: 1000/min per tenant

Performans Optimizasyonları:
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Caching Stratejisi:
   - Tenant info: Redis, TTL 15 min
   - Quota info: Redis, TTL 5 min
   - Subscription: Redis, TTL 1 hour
   - Usage stats: Redis, TTL 1 min

2. Database Indexing:
   - tenant.slug: Unique index (domain lookups)
   - tenant.status: B-tree index (filtering)
   - tenant.subscription_plan: B-tree index (reporting)
   - tenant.created_at: B-tree index (analytics)

3. Query Optimization:
   - Eager loading: subscription + quota in single query
   - Pagination: Keyset pagination (created_at + id)
   - Count queries: Cached aggregate counts

4. Background Jobs:
   - Quota reset: Cron job (monthly, 00:00 UTC)
   - Usage aggregation: Every 5 minutes
   - Invoice generation: 1 day before renewal
   - Subscription renewals: 00:00 UTC daily batch

Monitoring & Alerting:
~~~~~~~~~~~~~~~~~~~~
- Quota usage: Alert at 80%, 90%, 95%
- Payment failures: Immediate alert
- Subscription expiry: 7 days, 3 days, 1 day before
- Tenant suspension: Critical alert
- Unusual usage: Anomaly detection (Sentry)

ENTEGRASYONLAR (INTEGRATIONS)
-----------------------------

Payment Gateways:
~~~~~~~~~~~~~~~
1. Stripe (International):
   - stripe.Customer → tenant_id mapping
   - stripe.Subscription → SubscriptionInfo sync
   - Webhook: subscription.updated, invoice.payment_failed

2. Iyzico (Turkey):
   - Türk lirası ödemeleri
   - 3D Secure zorunlu
   - Installment support (taksit)

E-Invoice (E-Fatura):
~~~~~~~~~~~~~~~~~~~
- GİB Portal entegrasyonu
- e-Arşiv fatura oluşturma
- Automatic VKN validation
- PDF + XML format support

CRM/ERP Entegrasyonları:
~~~~~~~~~~~~~~~~~~~~~~
- Salesforce: Tenant sync
- HubSpot: Lead tracking
- Zoho: Billing automation
- SAP: Enterprise invoicing

Analytics & Reporting:
~~~~~~~~~~~~~~~~~~~~
- Grafana: Real-time dashboards
- Metabase: Business intelligence
- DataDog: APM + monitoring
- Sentry: Error tracking

ÖRNEKLERin KULLANIMI (USAGE EXAMPLES)
------------------------------------

1. Yeni Tenant Oluşturma (TRIAL):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.tenant import TenantCreate, OrganizationType

# Hukuk bürosu kaydı
create_data = TenantCreate(
    tenant_name="Ankara Hukuk Bürosu",
    slug="ankara-hukuk",  # Otomatik oluşturulabilir
    organization_type=OrganizationType.LAW_FIRM,
    contact_email="info@ankarahukuk.com",
    contact_phone="+905321234567",
    tax_number="1234567890",
    subscription_plan=SubscriptionPlan.TRIAL,
    accept_terms=True,
    accept_kvkk=True
)

response = await tenant_service.create_tenant(create_data)
# 14 günlük TRIAL başlar, trial_end_date set edilir
```

2. Subscription Upgrade (STARTER → PROFESSIONAL):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.tenant import SubscriptionUpgradeRequest

upgrade_request = SubscriptionUpgradeRequest(
    target_plan=SubscriptionPlan.PROFESSIONAL,
    billing_cycle=BillingCycle.YEARLY,  # %20 indirim
    payment_method_id="pm_abc123",  # Stripe payment method
    prorate=True  # Kalan süre için proration
)

response = await subscription_service.upgrade(
    tenant_id=tenant.id,
    upgrade_data=upgrade_request
)
# Immediate upgrade, invoice generated
```

3. Kota Kontrolü:
~~~~~~~~~~~~~~~
```python
from api.dependencies.quota import check_quota

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    tenant: Tenant = Depends(get_current_tenant),
    _: None = Depends(check_quota("documents_per_month"))
):
    # Kota check geçerse dosya upload işlemi
    # Quota exceeded ise HTTP 429 + upgrade suggestion
    ...
```

4. Kullanım Raporlama:
~~~~~~~~~~~~~~~~~~~~
```python
from api.schemas.tenant import TenantUsageStatistics

stats = await tenant_service.get_usage_statistics(
    tenant_id=tenant.id,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

print(f"Users: {stats.users_count}/{stats.quota.users}")
print(f"Storage: {stats.storage_used_gb:.2f}/{stats.quota.storage_gb}")
print(f"API Calls: {stats.api_calls_count}/{stats.quota.api_calls_per_month}")
```

5. Fatura Oluşturma ve E-Fatura:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from api.services.billing import generate_invoice

invoice = await generate_invoice(
    tenant_id=tenant.id,
    billing_period_start=datetime(2024, 1, 1),
    billing_period_end=datetime(2024, 1, 31)
)

# E-fatura gönderimi (Türkiye)
if tenant.tax_number and tenant.country == "TR":
    await efatura_service.send_invoice(invoice)
```

SORUN GİDERME (TROUBLESHOOTING)
-------------------------------

Sık Karşılaşılan Sorunlar:
~~~~~~~~~~~~~~~~~~~~~~~~

1. "Quota exceeded for documents_per_month":
   - Çözüm: Plan upgrade veya yeni ayı bekle
   - Geçici: Manuel quota artırma (admin)

2. "Payment failed, tenant suspended":
   - Çözüm: Ödeme yöntemini güncelle
   - Grace period: 7 gün içinde ödeme

3. "Tenant slug already exists":
   - Çözüm: Farklı slug kullan (örn: ankara-hukuk-2)
   - Auto-generated: tenant_name'den slug oluştur

4. "Cross-tenant data access detected":
   - Çözüm: RLS policy kontrol et
   - Debug: tenant_id mismatch bulundu

5. "Subscription renewal failed":
   - Çözüm: Webhook retry mechanism
   - Fallback: Manuel renewal trigger

VERSİYON GEÇMİŞİ
---------------
- v1.0.0 (2024-01): Initial implementation
  * Multi-tenant architecture
  * Subscription management
  * Quota system

- v1.1.0 (2024-02): Billing enhancements
  * E-invoice integration
  * Multiple payment methods
  * Proration support

- v1.2.0 (2024-03): Enterprise features
  * Custom plans
  * SSO support
  * Dedicated support

YETKİLENDİRME & LİSANS
---------------------
© 2024 LegalAI Platform
Enterprise-grade multi-tenant SaaS platform
Harvey AI & Legora inspired architecture
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import (
    Field,
    EmailStr,
    field_validator,
    model_validator,
    ConfigDict
)

from .base import (
    BaseSchema,
    RequestSchema,
    ResponseSchema,
    IdentifierSchema,
    TimestampSchema,
    AuditSchema,
    SoftDeleteSchema,
    validate_turkish_phone
)


# ============================================================================
# ENUMS (SABİTLER)
# ============================================================================

class SubscriptionPlan(str, Enum):
    """
    Abonelik planları (subscription tiers).
    EXACT MATCH with database model (backend/core/database/models/tenant.py)

    Plan Hiyerarşisi:
        FREE < TRIAL < STARTER < PROFESSIONAL < ENTERPRISE < CUSTOM

    Attributes:
        FREE: Ücretsiz plan (3 kullanıcı, 500MB, temel özellikler)
        TRIAL: 14 günlük deneme (STARTER özellikleri)
        STARTER: Başlangıç planı (10 kullanıcı, 10GB, $49/ay)
        PROFESSIONAL: Profesyonel plan (50 kullanıcı, 100GB, $199/ay)
        ENTERPRISE: Kurumsal plan (sınırsız, özel SLA, $999/ay)
        CUSTOM: Özel plan (tamamen özelleştirilmiş)
    """
    FREE = "free"
    TRIAL = "trial"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class TenantStatus(str, Enum):
    """
    Tenant durumları (lifecycle states).
    EXACT MATCH with database model (backend/core/database/models/tenant.py)

    Status Lifecycle:
        TRIAL → ACTIVE → SUSPENDED → CANCELLED
                   ↓
              EXPIRED (trial bitişi)

    Attributes:
        ACTIVE: Aktif abonelik
        TRIAL: Deneme süresi (14 gün)
        SUSPENDED: Askıya alındı (ödeme veya kota ihlali)
        EXPIRED: Süresi doldu (trial bitişi)
        CANCELLED: İptal edildi (kullanıcı veya admin)
    """
    ACTIVE = "active"
    TRIAL = "trial"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class OrganizationType(str, Enum):
    """
    Organizasyon tipleri (Türk hukuk sistemi).
    EXACT MATCH with database model (backend/core/database/models/tenant.py)

    Attributes:
        LAW_FIRM: Hukuk bürosu (avukatlık ortaklığı)
        CORPORATE_LEGAL: Şirket hukuk departmanı
        PUBLIC_INSTITUTION: Kamu kurumu (bakanlık, belediye)
        COURT: Mahkeme/Adliye
        PROSECUTORS_OFFICE: Savcılık
        NOTARY: Noterlik
        BAR_ASSOCIATION: Baro (il barosu)
        INDIVIDUAL: Bireysel kullanıcı
        OTHER: Diğer
    """
    LAW_FIRM = "law_firm"
    CORPORATE_LEGAL = "corporate_legal"
    PUBLIC_INSTITUTION = "public_institution"
    COURT = "court"
    PROSECUTORS_OFFICE = "prosecutors_office"
    NOTARY = "notary"
    BAR_ASSOCIATION = "bar_association"
    INDIVIDUAL = "individual"
    OTHER = "other"


class BillingCycle(str, Enum):
    """
    Faturalama döngüleri.

    Attributes:
        MONTHLY: Aylık faturalama
        QUARTERLY: 3 aylık (%10 indirim)
        YEARLY: Yıllık (%20 indirim)
    """
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class PaymentMethod(str, Enum):
    """
    Ödeme yöntemleri.

    Attributes:
        CREDIT_CARD: Kredi kartı (Stripe/Iyzico)
        BANK_TRANSFER: Banka havalesi (manuel onay)
        INVOICE: Fatura ile ödeme (ENTERPRISE)
    """
    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    INVOICE = "invoice"


class PaymentStatus(str, Enum):
    """
    Ödeme durumları.

    Attributes:
        PENDING: Beklemede
        PROCESSING: İşleniyor
        SUCCEEDED: Başarılı
        FAILED: Başarısız
        REFUNDED: İade edildi
    """
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"


# ============================================================================
# TENANT CRUD SCHEMAS
# ============================================================================

class TenantCreate(RequestSchema):
    """
    Yeni tenant oluşturma şeması.

    İlk kayıt sırasında kullanılır. Varsayılan olarak TRIAL plana başlar.

    Attributes:
        tenant_name: Tenant adı (şirket/baro adı)
        slug: URL-friendly identifier (opsiyonel, otomatik oluşturulabilir)
        organization_type: Organizasyon tipi
        contact_email: İletişim e-posta adresi
        contact_phone: İletişim telefonu (Türk formatı)
        tax_number: Vergi kimlik no (10 haneli)
        address: Açık adres (opsiyonel)
        city: Şehir
        country: Ülke kodu (ISO 3166-1 alpha-2)
        subscription_plan: İlk abonelik planı (varsayılan: TRIAL)
        billing_cycle: Faturalama döngüsü (varsayılan: MONTHLY)
        accept_terms: Kullanım şartları onayı (zorunlu)
        accept_kvkk: KVKK aydınlatma metni onayı (zorunlu)

    Validation:
        - tenant_name: 2-255 karakter
        - slug: 3-100 karakter, lowercase, alphanumeric + hyphens
        - tax_number: 10 haneli numeric
        - contact_phone: Türk telefon formatı
        - accept_terms ve accept_kvkk: True olmalı

    Example:
        >>> create_data = TenantCreate(
        ...     tenant_name="Ankara Hukuk Bürosu",
        ...     slug="ankara-hukuk",
        ...     organization_type=OrganizationType.LAW_FIRM,
        ...     contact_email="info@ankarahukuk.com",
        ...     contact_phone="+905321234567",
        ...     tax_number="1234567890",
        ...     city="Ankara",
        ...     country="TR",
        ...     accept_terms=True,
        ...     accept_kvkk=True
        ... )
    """
    tenant_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Tenant adı (şirket/kurum adı)",
        examples=["Ankara Hukuk Bürosu", "İstanbul Barosu"]
    )
    slug: Optional[str] = Field(
        None,
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9-]+$",
        description="URL-friendly identifier (otomatik oluşturulabilir)",
        examples=["ankara-hukuk", "istanbul-baro"]
    )
    organization_type: OrganizationType = Field(
        ...,
        description="Organizasyon tipi"
    )
    contact_email: EmailStr = Field(
        ...,
        description="İletişim e-posta adresi"
    )
    contact_phone: str = Field(
        ...,
        description="İletişim telefonu (Türk formatı)",
        examples=["+905321234567"]
    )
    tax_number: Optional[str] = Field(
        None,
        min_length=10,
        max_length=11,
        pattern=r"^\d{10,11}$",
        description="Vergi kimlik no (10 haneli) veya TC kimlik no (11 haneli)",
        json_schema_extra={"sensitive": True}
    )
    address: Optional[str] = Field(
        None,
        max_length=500,
        description="Açık adres"
    )
    city: Optional[str] = Field(
        None,
        max_length=100,
        description="Şehir",
        examples=["İstanbul", "Ankara", "İzmir"]
    )
    country: str = Field(
        default="TR",
        min_length=2,
        max_length=2,
        pattern=r"^[A-Z]{2}$",
        description="Ülke kodu (ISO 3166-1 alpha-2)",
        examples=["TR", "US", "GB"]
    )
    subscription_plan: SubscriptionPlan = Field(
        default=SubscriptionPlan.TRIAL,
        description="İlk abonelik planı"
    )
    billing_cycle: BillingCycle = Field(
        default=BillingCycle.MONTHLY,
        description="Faturalama döngüsü"
    )
    accept_terms: bool = Field(
        ...,
        description="Kullanım şartları onayı (zorunlu)"
    )
    accept_kvkk: bool = Field(
        ...,
        description="KVKK aydınlatma metni onayı (zorunlu)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_name": "Ankara Hukuk Bürosu",
                "slug": "ankara-hukuk",
                "organization_type": "law_firm",
                "contact_email": "info@ankarahukuk.com",
                "contact_phone": "+905321234567",
                "tax_number": "1234567890",
                "address": "Kızılay Mah. Atatürk Bulvarı No:123 Çankaya",
                "city": "Ankara",
                "country": "TR",
                "subscription_plan": "trial",
                "billing_cycle": "monthly",
                "accept_terms": True,
                "accept_kvkk": True
            }
        }
    )

    @field_validator("contact_phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        """Türk telefon numarası validasyonu ve normalizasyonu."""
        return validate_turkish_phone(v)

    @field_validator("contact_email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> str:
        """E-posta adresini normalize et (lowercase)."""
        return v.lower().strip()

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: Optional[str]) -> Optional[str]:
        """Slug validasyonu ve normalizasyonu."""
        if v:
            v = v.lower().strip()
            if not v.replace("-", "").isalnum():
                raise ValueError(
                    "Slug sadece küçük harf, rakam ve tire (-) içerebilir"
                )
        return v

    @model_validator(mode="after")
    def validate_terms_acceptance(self):
        """Kullanım şartları ve KVKK onayı kontrolü."""
        if not self.accept_terms:
            raise ValueError("Kullanım şartlarını kabul etmelisiniz")
        if not self.accept_kvkk:
            raise ValueError("KVKK aydınlatma metnini kabul etmelisiniz")
        return self


class TenantUpdate(RequestSchema):
    """
    Tenant güncelleme şeması (admin/owner).

    Tüm alanlar opsiyonel. Sadece gönderilen alanlar güncellenir.

    Attributes:
        tenant_name: Yeni tenant adı
        organization_type: Yeni organizasyon tipi
        contact_email: Yeni iletişim e-posta
        contact_phone: Yeni iletişim telefonu
        tax_number: Yeni vergi kimlik no
        address: Yeni adres
        city: Yeni şehir
        country: Yeni ülke
        status: Yeni durum (admin only)
    """
    tenant_name: Optional[str] = Field(
        None,
        min_length=2,
        max_length=255,
        description="Tenant adı"
    )
    organization_type: Optional[OrganizationType] = Field(
        None,
        description="Organizasyon tipi"
    )
    contact_email: Optional[EmailStr] = Field(
        None,
        description="İletişim e-posta"
    )
    contact_phone: Optional[str] = Field(
        None,
        description="İletişim telefonu"
    )
    tax_number: Optional[str] = Field(
        None,
        min_length=10,
        max_length=11,
        pattern=r"^\d{10,11}$",
        description="Vergi kimlik no",
        json_schema_extra={"sensitive": True}
    )
    address: Optional[str] = Field(
        None,
        max_length=500,
        description="Adres"
    )
    city: Optional[str] = Field(
        None,
        max_length=100,
        description="Şehir"
    )
    country: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        pattern=r"^[A-Z]{2}$",
        description="Ülke kodu"
    )
    status: Optional[TenantStatus] = Field(
        None,
        description="Tenant durumu (admin only)"
    )

    @field_validator("contact_phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Telefon validasyonu."""
        if v:
            return validate_turkish_phone(v)
        return v

    @field_validator("contact_email")
    @classmethod
    def normalize_email(cls, v: Optional[EmailStr]) -> Optional[str]:
        """E-posta normalizasyonu."""
        if v:
            return v.lower().strip()
        return v


class TenantSettingsUpdate(RequestSchema):
    """
    Tenant ayarları güncelleme şeması.

    UI tercihleri, bildirim ayarları, güvenlik politikaları.

    Attributes:
        timezone: Saat dilimi (IANA timezone)
        language: Dil kodu (tr, en, de)
        date_format: Tarih formatı
        currency: Para birimi (TRY, USD, EUR)
        notifications_enabled: Bildirimler aktif mi
        email_notifications: E-posta bildirimleri
        sms_notifications: SMS bildirimleri
        webhook_url: Webhook URL (events için)
    """
    timezone: Optional[str] = Field(
        None,
        description="Saat dilimi (IANA timezone)",
        examples=["Europe/Istanbul", "UTC"]
    )
    language: Optional[str] = Field(
        None,
        pattern=r"^[a-z]{2}$",
        description="Dil kodu (ISO 639-1)",
        examples=["tr", "en", "de"]
    )
    date_format: Optional[str] = Field(
        None,
        description="Tarih formatı",
        examples=["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"]
    )
    currency: Optional[str] = Field(
        None,
        pattern=r"^[A-Z]{3}$",
        description="Para birimi (ISO 4217)",
        examples=["TRY", "USD", "EUR"]
    )
    notifications_enabled: Optional[bool] = Field(
        None,
        description="Bildirimler aktif mi"
    )
    email_notifications: Optional[bool] = Field(
        None,
        description="E-posta bildirimleri"
    )
    sms_notifications: Optional[bool] = Field(
        None,
        description="SMS bildirimleri"
    )
    webhook_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Webhook URL (events için)"
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class TenantResponse(ResponseSchema, IdentifierSchema, TimestampSchema):
    """
    Temel tenant yanıt şeması.

    Listeleme ve temel bilgiler için kullanılır.

    Attributes:
        id: Tenant UUID
        tenant_name: Tenant adı
        slug: URL slug
        organization_type: Organizasyon tipi
        status: Tenant durumu
        subscription_plan: Abonelik planı
        created_at: Oluşturma zamanı
        updated_at: Güncelleme zamanı
    """
    tenant_name: str = Field(..., description="Tenant adı")
    slug: str = Field(..., description="URL slug")
    organization_type: OrganizationType = Field(..., description="Organizasyon tipi")
    status: TenantStatus = Field(..., description="Tenant durumu")
    subscription_plan: SubscriptionPlan = Field(..., description="Abonelik planı")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_name": "Ankara Hukuk Bürosu",
                "slug": "ankara-hukuk",
                "organization_type": "law_firm",
                "status": "active",
                "subscription_plan": "professional",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T14:45:00Z"
            }
        }
    )


class TenantDetailResponse(TenantResponse):
    """
    Detaylı tenant yanıt şeması.

    Tek tenant görüntüleme için kullanılır. Abonelik ve kota bilgileri içerir.

    Attributes:
        (TenantResponse'dan tüm alanlar)
        contact_email: İletişim e-posta
        contact_phone: İletişim telefonu
        tax_number: Vergi kimlik no (maskelenmiş)
        address: Adres
        city: Şehir
        country: Ülke
        billing_cycle: Faturalama döngüsü
        trial_end_date: Deneme bitiş tarihi (eğer TRIAL)
        subscription_end_date: Abonelik bitiş tarihi
        users_count: Toplam kullanıcı sayısı
        storage_used_mb: Kullanılan depolama (MB)
        is_trial: Deneme mi
        is_active: Aktif mi
    """
    contact_email: EmailStr = Field(..., description="İletişim e-posta")
    contact_phone: str = Field(..., description="İletişim telefonu")
    tax_number: Optional[str] = Field(None, description="Vergi kimlik no (maskelenmiş)")
    address: Optional[str] = Field(None, description="Adres")
    city: Optional[str] = Field(None, description="Şehir")
    country: str = Field(..., description="Ülke kodu")
    billing_cycle: BillingCycle = Field(..., description="Faturalama döngüsü")
    trial_end_date: Optional[datetime] = Field(None, description="Deneme bitiş tarihi")
    subscription_end_date: Optional[datetime] = Field(None, description="Abonelik bitiş tarihi")
    users_count: int = Field(..., ge=0, description="Toplam kullanıcı sayısı")
    storage_used_mb: float = Field(..., ge=0, description="Kullanılan depolama (MB)")
    is_trial: bool = Field(..., description="Deneme süresi mi")
    is_active: bool = Field(..., description="Aktif mi")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_name": "Ankara Hukuk Bürosu",
                "slug": "ankara-hukuk",
                "organization_type": "law_firm",
                "status": "active",
                "subscription_plan": "professional",
                "contact_email": "info@ankarahukuk.com",
                "contact_phone": "+905321234567",
                "tax_number": "12345*****",
                "address": "Kızılay Mah. Atatürk Bulvarı No:123",
                "city": "Ankara",
                "country": "TR",
                "billing_cycle": "yearly",
                "trial_end_date": None,
                "subscription_end_date": "2025-01-15T00:00:00Z",
                "users_count": 25,
                "storage_used_mb": 45678.9,
                "is_trial": False,
                "is_active": True,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T14:45:00Z"
            }
        }
    )


class TenantPublicInfo(BaseSchema):
    """
    Genel tenant bilgileri (public).

    Domain bazlı tenant tespiti için kullanılır.
    Hassas bilgiler içermez.

    Attributes:
        tenant_name: Tenant adı
        slug: URL slug
        organization_type: Organizasyon tipi
        city: Şehir
        country: Ülke
    """
    tenant_name: str = Field(..., description="Tenant adı")
    slug: str = Field(..., description="URL slug")
    organization_type: OrganizationType = Field(..., description="Organizasyon tipi")
    city: Optional[str] = Field(None, description="Şehir")
    country: str = Field(..., description="Ülke kodu")


# ============================================================================
# SUBSCRIPTION MANAGEMENT
# ============================================================================

class SubscriptionInfo(BaseSchema):
    """
    Abonelik durumu ve detayları.

    Attributes:
        plan: Mevcut abonelik planı
        status: Abonelik durumu
        billing_cycle: Faturalama döngüsü
        current_period_start: Mevcut dönem başlangıcı
        current_period_end: Mevcut dönem bitişi
        trial_end_date: Deneme bitiş tarihi (eğer TRIAL)
        cancel_at_period_end: Dönem sonunda iptal edilecek mi
        cancelled_at: İptal tarihi
        auto_renew: Otomatik yenilenme aktif mi
        next_billing_date: Sonraki faturalama tarihi
        amount: Tutar (Decimal)
        currency: Para birimi
    """
    plan: SubscriptionPlan = Field(..., description="Mevcut plan")
    status: TenantStatus = Field(..., description="Abonelik durumu")
    billing_cycle: BillingCycle = Field(..., description="Faturalama döngüsü")
    current_period_start: datetime = Field(..., description="Mevcut dönem başlangıcı")
    current_period_end: datetime = Field(..., description="Mevcut dönem bitişi")
    trial_end_date: Optional[datetime] = Field(None, description="Deneme bitiş tarihi")
    cancel_at_period_end: bool = Field(False, description="Dönem sonunda iptal mi")
    cancelled_at: Optional[datetime] = Field(None, description="İptal tarihi")
    auto_renew: bool = Field(True, description="Otomatik yenilenme")
    next_billing_date: Optional[datetime] = Field(None, description="Sonraki fatura tarihi")
    amount: Decimal = Field(..., ge=0, description="Tutar")
    currency: str = Field(..., description="Para birimi")


class SubscriptionPlanUpdate(RequestSchema):
    """
    Abonelik planı değişikliği isteği (upgrade/downgrade).

    Attributes:
        target_plan: Hedef plan
        billing_cycle: Yeni faturalama döngüsü
        prorate: Kalan süre için proration yapılsın mı
        effective_date: Geçerlilik tarihi (None = immediate)
    """
    target_plan: SubscriptionPlan = Field(..., description="Hedef abonelik planı")
    billing_cycle: Optional[BillingCycle] = Field(None, description="Faturalama döngüsü")
    prorate: bool = Field(True, description="Proration uygulansın mı")
    effective_date: Optional[datetime] = Field(
        None,
        description="Geçerlilik tarihi (None = hemen)"
    )


class SubscriptionUpgradeRequest(RequestSchema):
    """
    Abonelik upgrade isteği.

    Attributes:
        target_plan: Hedef plan (mevcut plandan üst)
        billing_cycle: Faturalama döngüsü
        payment_method_id: Ödeme yöntemi ID (Stripe/Iyzico)
        prorate: Kalan süre için proration
    """
    target_plan: SubscriptionPlan = Field(..., description="Hedef plan")
    billing_cycle: BillingCycle = Field(
        default=BillingCycle.MONTHLY,
        description="Faturalama döngüsü"
    )
    payment_method_id: Optional[str] = Field(
        None,
        description="Ödeme yöntemi ID (Stripe PM ID)"
    )
    prorate: bool = Field(True, description="Proration uygulansın mı")


class SubscriptionDowngradeRequest(RequestSchema):
    """
    Abonelik downgrade isteği.

    Genellikle dönem sonunda uygulanır.

    Attributes:
        target_plan: Hedef plan (mevcut plandan alt)
        reason: Downgrade nedeni
        feedback: Kullanıcı geri bildirimi
        immediate: Hemen mi uygulansın (varsayılan: dönem sonu)
    """
    target_plan: SubscriptionPlan = Field(..., description="Hedef plan")
    reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Downgrade nedeni"
    )
    feedback: Optional[str] = Field(
        None,
        max_length=1000,
        description="Kullanıcı geri bildirimi"
    )
    immediate: bool = Field(False, description="Hemen uygulansın mı")


class SubscriptionCancellationRequest(RequestSchema):
    """
    Abonelik iptal isteği.

    Attributes:
        reason: İptal nedeni
        feedback: Kullanıcı geri bildirimi
        cancel_at_period_end: Dönem sonunda mı iptal edilsin
        request_refund: İade talep edilsin mi
    """
    reason: str = Field(..., max_length=500, description="İptal nedeni")
    feedback: Optional[str] = Field(
        None,
        max_length=1000,
        description="Geri bildirim"
    )
    cancel_at_period_end: bool = Field(
        True,
        description="Dönem sonunda iptal (False = hemen)"
    )
    request_refund: bool = Field(False, description="İade talebi")


# ============================================================================
# QUOTA & USAGE MANAGEMENT
# ============================================================================

class TenantQuotaInfo(BaseSchema):
    """
    Tenant kota bilgileri.

    Her abonelik planının farklı kotaları vardır.

    Attributes:
        plan: Abonelik planı
        users: Maksimum kullanıcı sayısı
        storage_gb: Maksimum depolama (GB)
        api_calls_per_month: Aylık API call limiti
        ai_queries_per_month: Aylık AI sorgu limiti
        documents_per_month: Aylık belge yükleme limiti
        chat_messages_per_day: Günlük chat mesaj limiti
        features: Erişilebilir özellikler listesi
    """
    plan: SubscriptionPlan = Field(..., description="Abonelik planı")
    users: int = Field(..., ge=0, description="Maksimum kullanıcı")
    storage_gb: float = Field(..., ge=0, description="Maksimum depolama (GB)")
    api_calls_per_month: int = Field(..., ge=0, description="Aylık API call limiti")
    ai_queries_per_month: int = Field(..., ge=0, description="Aylık AI sorgu limiti")
    documents_per_month: int = Field(..., ge=0, description="Aylık belge limiti")
    chat_messages_per_day: int = Field(..., ge=0, description="Günlük chat mesaj limiti")
    features: List[str] = Field(default_factory=list, description="Erişilebilir özellikler")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plan": "professional",
                "users": 50,
                "storage_gb": 100.0,
                "api_calls_per_month": 100000,
                "ai_queries_per_month": 5000,
                "documents_per_month": 2000,
                "chat_messages_per_day": 1000,
                "features": [
                    "basic_chat",
                    "document_upload",
                    "document_analysis",
                    "templates",
                    "advanced_ai",
                    "integrations",
                    "api_access",
                    "webhooks"
                ]
            }
        }
    )


class TenantUsageStatistics(BaseSchema):
    """
    Tenant kullanım istatistikleri.

    Mevcut kullanım ve kota bilgileri.

    Attributes:
        quota: Kota bilgileri
        users_count: Mevcut kullanıcı sayısı
        storage_used_gb: Kullanılan depolama (GB)
        api_calls_count: Bu ayki API call sayısı
        ai_queries_count: Bu ayki AI sorgu sayısı
        documents_count: Bu ayki belge yükleme sayısı
        chat_messages_today: Bugünkü chat mesaj sayısı
        usage_percentage: Genel kullanım yüzdesi
        quota_warnings: Kota uyarıları listesi
    """
    quota: TenantQuotaInfo = Field(..., description="Kota limitleri")
    users_count: int = Field(..., ge=0, description="Mevcut kullanıcı sayısı")
    storage_used_gb: float = Field(..., ge=0, description="Kullanılan depolama (GB)")
    api_calls_count: int = Field(..., ge=0, description="Bu ayki API call")
    ai_queries_count: int = Field(..., ge=0, description="Bu ayki AI sorgu")
    documents_count: int = Field(..., ge=0, description="Bu ayki belge")
    chat_messages_today: int = Field(..., ge=0, description="Bugünkü chat mesaj")
    usage_percentage: Dict[str, float] = Field(
        default_factory=dict,
        description="Kullanım yüzdeleri (users: 50.0, storage: 75.5)"
    )
    quota_warnings: List[str] = Field(
        default_factory=list,
        description="Kota uyarıları"
    )


class UsageReport(BaseSchema):
    """
    Periyodik kullanım raporu (aylık/yıllık).

    Attributes:
        tenant_id: Tenant UUID
        period_start: Rapor başlangıç tarihi
        period_end: Rapor bitiş tarihi
        statistics: Kullanım istatistikleri
        total_cost: Toplam maliyet
        currency: Para birimi
        generated_at: Rapor oluşturma zamanı
    """
    tenant_id: UUID = Field(..., description="Tenant UUID")
    period_start: datetime = Field(..., description="Dönem başlangıcı")
    period_end: datetime = Field(..., description="Dönem bitişi")
    statistics: TenantUsageStatistics = Field(..., description="Kullanım istatistikleri")
    total_cost: Decimal = Field(..., ge=0, description="Toplam maliyet")
    currency: str = Field(..., description="Para birimi")
    generated_at: datetime = Field(..., description="Rapor oluşturma zamanı")


# ============================================================================
# BILLING & PAYMENT
# ============================================================================

class BillingInfo(BaseSchema):
    """
    Faturalama bilgileri.

    Attributes:
        company_name: Şirket adı (faturada görünecek)
        tax_number: Vergi kimlik no
        tax_office: Vergi dairesi
        billing_email: Fatura e-posta adresi
        billing_address: Fatura adresi
        city: Şehir
        postal_code: Posta kodu
        country: Ülke
    """
    company_name: str = Field(..., max_length=255, description="Şirket adı")
    tax_number: str = Field(
        ...,
        min_length=10,
        max_length=11,
        description="Vergi kimlik no",
        json_schema_extra={"sensitive": True}
    )
    tax_office: Optional[str] = Field(None, max_length=100, description="Vergi dairesi")
    billing_email: EmailStr = Field(..., description="Fatura e-posta")
    billing_address: str = Field(..., max_length=500, description="Fatura adresi")
    city: str = Field(..., max_length=100, description="Şehir")
    postal_code: Optional[str] = Field(None, max_length=20, description="Posta kodu")
    country: str = Field(..., min_length=2, max_length=2, description="Ülke kodu")


class PaymentMethodInfo(BaseSchema):
    """
    Ödeme yöntemi bilgileri.

    Attributes:
        id: Payment method ID (Stripe PM ID)
        type: Ödeme yöntemi tipi
        card_last4: Kart son 4 hanesi (eğer kredi kartı)
        card_brand: Kart markası (Visa, Mastercard)
        expiry_month: Son kullanma ayı
        expiry_year: Son kullanma yılı
        is_default: Varsayılan ödeme yöntemi mi
        created_at: Eklenme tarihi
    """
    id: str = Field(..., description="Payment method ID")
    type: PaymentMethod = Field(..., description="Ödeme yöntemi tipi")
    card_last4: Optional[str] = Field(None, max_length=4, description="Kart son 4 hanesi")
    card_brand: Optional[str] = Field(None, description="Kart markası")
    expiry_month: Optional[int] = Field(None, ge=1, le=12, description="Son kullanma ayı")
    expiry_year: Optional[int] = Field(None, ge=2024, description="Son kullanma yılı")
    is_default: bool = Field(False, description="Varsayılan ödeme yöntemi")
    created_at: datetime = Field(..., description="Eklenme tarihi")


class InvoiceInfo(BaseSchema, IdentifierSchema):
    """
    Fatura bilgileri.

    Attributes:
        id: Fatura UUID
        invoice_number: Fatura numarası
        tenant_id: Tenant UUID
        amount: Tutar
        tax_amount: Vergi tutarı (KDV)
        total_amount: Toplam tutar (amount + tax)
        currency: Para birimi
        status: Ödeme durumu
        billing_period_start: Faturalanan dönem başlangıcı
        billing_period_end: Faturalanan dönem bitişi
        due_date: Son ödeme tarihi
        paid_at: Ödenme tarihi
        invoice_url: Fatura PDF URL
        created_at: Oluşturma tarihi
    """
    invoice_number: str = Field(..., description="Fatura numarası")
    tenant_id: UUID = Field(..., description="Tenant UUID")
    amount: Decimal = Field(..., ge=0, description="Tutar")
    tax_amount: Decimal = Field(..., ge=0, description="Vergi tutarı")
    total_amount: Decimal = Field(..., ge=0, description="Toplam tutar")
    currency: str = Field(..., description="Para birimi")
    status: PaymentStatus = Field(..., description="Ödeme durumu")
    billing_period_start: datetime = Field(..., description="Dönem başlangıcı")
    billing_period_end: datetime = Field(..., description="Dönem bitişi")
    due_date: datetime = Field(..., description="Son ödeme tarihi")
    paid_at: Optional[datetime] = Field(None, description="Ödenme tarihi")
    invoice_url: Optional[str] = Field(None, description="Fatura PDF URL")
    created_at: datetime = Field(..., description="Oluşturma tarihi")


class PaymentHistoryEntry(BaseSchema, IdentifierSchema):
    """
    Ödeme geçmişi kaydı.

    Attributes:
        id: Payment UUID
        tenant_id: Tenant UUID
        invoice_id: İlgili fatura UUID
        amount: Ödenen tutar
        currency: Para birimi
        payment_method: Ödeme yöntemi
        status: Ödeme durumu
        payment_date: Ödeme tarihi
        transaction_id: İşlem ID (Stripe/Iyzico)
        description: Açıklama
    """
    tenant_id: UUID = Field(..., description="Tenant UUID")
    invoice_id: Optional[UUID] = Field(None, description="Fatura UUID")
    amount: Decimal = Field(..., ge=0, description="Ödenen tutar")
    currency: str = Field(..., description="Para birimi")
    payment_method: PaymentMethod = Field(..., description="Ödeme yöntemi")
    status: PaymentStatus = Field(..., description="Ödeme durumu")
    payment_date: datetime = Field(..., description="Ödeme tarihi")
    transaction_id: Optional[str] = Field(None, description="İşlem ID")
    description: Optional[str] = Field(None, description="Açıklama")


# ============================================================================
# FILTER & LIST SCHEMAS
# ============================================================================

class TenantFilterParams(BaseSchema):
    """
    Tenant filtreleme parametreleri.

    Admin panelinde tenant listesi için kullanılır.

    Attributes:
        status: Durum filtresi
        subscription_plan: Abonelik planı filtresi
        organization_type: Organizasyon tipi filtresi
        country: Ülke filtresi
        search: Arama terimi (tenant_name, contact_email)
        created_after: Bu tarihten sonra oluşturulan
        created_before: Bu tarihten önce oluşturulan
        is_trial: Sadece trial tenantları
        is_active: Sadece aktif tenantları
    """
    status: Optional[TenantStatus] = Field(None, description="Durum filtresi")
    subscription_plan: Optional[SubscriptionPlan] = Field(None, description="Plan filtresi")
    organization_type: Optional[OrganizationType] = Field(None, description="Org tipi filtresi")
    country: Optional[str] = Field(None, description="Ülke filtresi")
    search: Optional[str] = Field(
        None,
        min_length=2,
        max_length=100,
        description="Arama terimi"
    )
    created_after: Optional[datetime] = Field(None, description="Bu tarihten sonra")
    created_before: Optional[datetime] = Field(None, description="Bu tarihten önce")
    is_trial: Optional[bool] = Field(None, description="Sadece trial")
    is_active: Optional[bool] = Field(None, description="Sadece aktif")


class TenantStatistics(BaseSchema):
    """
    Tenant istatistikleri (admin dashboard).

    Platform genelinde tenant metrikleri.

    Attributes:
        total_tenants: Toplam tenant sayısı
        active_tenants: Aktif tenant sayısı
        trial_tenants: Trial tenant sayısı
        suspended_tenants: Askıya alınmış tenant sayısı
        by_plan: Plan bazında dağılım
        by_organization_type: Organizasyon tipi bazında dağılım
        by_country: Ülke bazında dağılım
        total_revenue_monthly: Aylık toplam gelir
        total_users: Toplam kullanıcı sayısı (tüm tenantlar)
        total_storage_gb: Toplam kullanılan depolama (GB)
    """
    total_tenants: int = Field(..., ge=0, description="Toplam tenant")
    active_tenants: int = Field(..., ge=0, description="Aktif tenant")
    trial_tenants: int = Field(..., ge=0, description="Trial tenant")
    suspended_tenants: int = Field(..., ge=0, description="Askıya alınmış")
    by_plan: Dict[str, int] = Field(default_factory=dict, description="Plan dağılımı")
    by_organization_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Org tipi dağılımı"
    )
    by_country: Dict[str, int] = Field(default_factory=dict, description="Ülke dağılımı")
    total_revenue_monthly: Decimal = Field(..., ge=0, description="Aylık gelir")
    total_users: int = Field(..., ge=0, description="Toplam kullanıcı")
    total_storage_gb: float = Field(..., ge=0, description="Toplam depolama")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "SubscriptionPlan",
    "TenantStatus",
    "OrganizationType",
    "BillingCycle",
    "PaymentMethod",
    "PaymentStatus",
    # CRUD
    "TenantCreate",
    "TenantUpdate",
    "TenantSettingsUpdate",
    # Responses
    "TenantResponse",
    "TenantDetailResponse",
    "TenantPublicInfo",
    # Subscription
    "SubscriptionInfo",
    "SubscriptionPlanUpdate",
    "SubscriptionUpgradeRequest",
    "SubscriptionDowngradeRequest",
    "SubscriptionCancellationRequest",
    # Quota & Usage
    "TenantQuotaInfo",
    "TenantUsageStatistics",
    "UsageReport",
    # Billing & Payment
    "BillingInfo",
    "PaymentMethodInfo",
    "InvoiceInfo",
    "PaymentHistoryEntry",
    # Filters & Stats
    "TenantFilterParams",
    "TenantStatistics",
]
