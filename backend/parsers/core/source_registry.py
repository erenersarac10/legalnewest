"""
Source Registry - Harvey/Legora CTO-Level

Central registry of all Turkish legal document sources.
Maps source names to metadata (URLs, API endpoints, authentication, rate limits).

Supported Sources:
    Government:
        - Resmi Gazete (resmigazete.gov.tr)
        - Mevzuat Bilgi Sistemi (mevzuat.gov.tr)
        - UYAP (Ulusal Yargı Ağı Projesi)
        - TBMM (tbmm.gov.tr)

    Courts:
        - Yargıtay (yargitay.gov.tr)
        - Danıştay (danistay.gov.tr)
        - Anayasa Mahkemesi (anayasa.gov.tr)

    Regulators:
        - KVKK (kvkk.gov.tr)
        - SPK (spk.gov.tr)
        - BDDK (bddk.org.tr)
        - Rekabet Kurumu (rekabet.gov.tr)
        - EPDK (epdk.gov.tr)
        - GİB (gib.gov.tr)
        - SGK (sgk.gov.tr)
        - Sayıştay (sayistay.gov.tr)

    International:
        - AİHM (ECHR - hudoc.echr.coe.int)
        - EUR-Lex (eur-lex.europa.eu)

    Commercial:
        - Kazancı (kazanci.com.tr)
        - Lexpera (lexpera.com.tr)

Author: Legal AI Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class SourceStatus(str, Enum):
    """Source availability status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    OFFLINE = "offline"
    REQUIRES_AUTH = "requires_auth"


class SourceType(str, Enum):
    """Type of source."""
    GOVERNMENT = "government"
    COURT = "court"
    REGULATOR = "regulator"
    INTERNATIONAL = "international"
    COMMERCIAL = "commercial"


@dataclass
class SourceMetadata:
    """Metadata for a legal document source."""

    # Identifiers
    source_id: str
    name: str
    display_name: str

    # Classification
    source_type: SourceType
    status: SourceStatus = SourceStatus.ACTIVE

    # Access
    base_url: str
    api_endpoint: Optional[str] = None
    requires_auth: bool = False
    auth_type: Optional[str] = None  # "api_key", "oauth", "basic"

    # Rate limits
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    concurrent_requests: int = 1

    # Capabilities
    supports_search: bool = False
    supports_bulk_download: bool = False
    supports_real_time: bool = False

    # Document types
    document_types: List[str] = field(default_factory=list)

    # Metadata
    description: str = ""
    last_verified: Optional[str] = None
    notes: str = ""

    # Technical
    adapter_class: Optional[str] = None
    parser_class: Optional[str] = None


class SourceRegistry:
    """
    Central registry of all legal document sources.
    Provides unified access to source metadata and adapters.
    """

    def __init__(self):
        """Initialize source registry."""
        self._sources: Dict[str, SourceMetadata] = {}
        self._initialize_default_sources()
        logger.info(f"Initialized SourceRegistry with {len(self._sources)} sources")

    def _initialize_default_sources(self):
        """Initialize default Turkish legal sources."""

        # ====================================================================
        # GOVERNMENT SOURCES
        # ====================================================================

        self.register(SourceMetadata(
            source_id="resmi_gazete",
            name="Resmi Gazete",
            display_name="T.C. Resmi Gazete",
            source_type=SourceType.GOVERNMENT,
            base_url="https://www.resmigazete.gov.tr",
            supports_search=True,
            supports_bulk_download=False,
            document_types=["kanun", "yonetmelik", "cbk", "teblig", "genelge"],
            description="Official gazette of the Republic of Turkey",
            adapter_class="ResmiGazeteAdapter",
        ))

        self.register(SourceMetadata(
            source_id="mevzuat_gov",
            name="Mevzuat Bilgi Sistemi",
            display_name="Mevzuat.gov.tr",
            source_type=SourceType.GOVERNMENT,
            base_url="https://www.mevzuat.gov.tr",
            supports_search=True,
            supports_bulk_download=True,
            document_types=["kanun", "yonetmelik", "cbk", "tuzuk", "teblig"],
            description="Legislative information system",
            adapter_class="MevzuatGovAdapter",
        ))

        self.register(SourceMetadata(
            source_id="tbmm",
            name="TBMM",
            display_name="Türkiye Büyük Millet Meclisi",
            source_type=SourceType.GOVERNMENT,
            base_url="https://www.tbmm.gov.tr",
            supports_search=True,
            document_types=["kanun", "meclis_karari"],
            description="Grand National Assembly of Turkey",
            adapter_class="TBMMAdapter",
        ))

        # ====================================================================
        # COURTS
        # ====================================================================

        self.register(SourceMetadata(
            source_id="yargitay",
            name="Yargıtay",
            display_name="Yargıtay",
            source_type=SourceType.COURT,
            base_url="https://www.yargitay.gov.tr",
            supports_search=True,
            document_types=["yargitay_karari"],
            description="Court of Cassation",
            adapter_class="YargitayAdapter",
            rate_limit_per_minute=30,
        ))

        self.register(SourceMetadata(
            source_id="danistay",
            name="Danıştay",
            display_name="Danıştay",
            source_type=SourceType.COURT,
            base_url="https://www.danistay.gov.tr",
            supports_search=True,
            document_types=["danistay_karari"],
            description="Council of State",
            adapter_class="DanistayAdapter",
        ))

        self.register(SourceMetadata(
            source_id="anayasa_mahkemesi",
            name="Anayasa Mahkemesi",
            display_name="Anayasa Mahkemesi",
            source_type=SourceType.COURT,
            base_url="https://www.anayasa.gov.tr",
            supports_search=True,
            document_types=["aym_karari"],
            description="Constitutional Court",
            adapter_class="AYMAdapter",
        ))

        # ====================================================================
        # REGULATORS
        # ====================================================================

        self.register(SourceMetadata(
            source_id="kvkk",
            name="KVKK",
            display_name="Kişisel Verileri Koruma Kurumu",
            source_type=SourceType.REGULATOR,
            base_url="https://www.kvkk.gov.tr",
            supports_search=True,
            document_types=["kvkk_karari", "rehber", "teblig"],
            description="Personal Data Protection Authority",
            adapter_class="KVKKAdapter",
        ))

        self.register(SourceMetadata(
            source_id="spk",
            name="SPK",
            display_name="Sermaye Piyasası Kurulu",
            source_type=SourceType.REGULATOR,
            base_url="https://www.spk.gov.tr",
            supports_search=True,
            document_types=["spk_teblig", "spk_karari"],
            description="Capital Markets Board",
            adapter_class="SPKAdapter",
        ))

        self.register(SourceMetadata(
            source_id="bddk",
            name="BDDK",
            display_name="Bankacılık Düzenleme ve Denetleme Kurumu",
            source_type=SourceType.REGULATOR,
            base_url="https://www.bddk.org.tr",
            supports_search=True,
            document_types=["bddk_yonetmelik", "bddk_teblig"],
            description="Banking Regulation and Supervision Agency",
            adapter_class="BDDKAdapter",
        ))

        self.register(SourceMetadata(
            source_id="rekabet",
            name="Rekabet Kurumu",
            display_name="Rekabet Kurumu",
            source_type=SourceType.REGULATOR,
            base_url="https://www.rekabet.gov.tr",
            supports_search=True,
            document_types=["rekabet_karari"],
            description="Competition Authority",
            adapter_class="RekabetAdapter",
        ))

        self.register(SourceMetadata(
            source_id="epdk",
            name="EPDK",
            display_name="Enerji Piyasası Düzenleme Kurumu",
            source_type=SourceType.REGULATOR,
            base_url="https://www.epdk.gov.tr",
            supports_search=True,
            document_types=["epdk_karar", "epdk_teblig"],
            description="Energy Market Regulatory Authority",
            adapter_class="EPDKAdapter",
        ))

        self.register(SourceMetadata(
            source_id="gib",
            name="GİB",
            display_name="Gelir İdaresi Başkanlığı",
            source_type=SourceType.REGULATOR,
            base_url="https://www.gib.gov.tr",
            supports_search=True,
            document_types=["gib_teblig", "gib_genelge", "vergi_usul"],
            description="Revenue Administration",
            adapter_class="GIBAdapter",
        ))

        self.register(SourceMetadata(
            source_id="sgk",
            name="SGK",
            display_name="Sosyal Güvenlik Kurumu",
            source_type=SourceType.REGULATOR,
            base_url="https://www.sgk.gov.tr",
            supports_search=True,
            document_types=["sgk_genelge", "sgk_teblig"],
            description="Social Security Institution",
            adapter_class="SGKAdapter",
        ))

        self.register(SourceMetadata(
            source_id="sayistay",
            name="Sayıştay",
            display_name="Sayıştay Başkanlığı",
            source_type=SourceType.REGULATOR,
            base_url="https://www.sayistay.gov.tr",
            supports_search=True,
            document_types=["sayistay_rapor"],
            description="Court of Accounts",
            adapter_class="SayistayAdapter",
        ))

        # ====================================================================
        # INTERNATIONAL
        # ====================================================================

        self.register(SourceMetadata(
            source_id="echr",
            name="ECHR",
            display_name="European Court of Human Rights",
            source_type=SourceType.INTERNATIONAL,
            base_url="https://hudoc.echr.coe.int",
            api_endpoint="https://hudoc.echr.coe.int/api",
            supports_search=True,
            supports_bulk_download=True,
            document_types=["aihm_karari"],
            description="European Court of Human Rights",
            adapter_class="ECHRAdapter",
            rate_limit_per_minute=60,
        ))

        self.register(SourceMetadata(
            source_id="eur_lex",
            name="EUR-Lex",
            display_name="EUR-Lex",
            source_type=SourceType.INTERNATIONAL,
            base_url="https://eur-lex.europa.eu",
            supports_search=True,
            document_types=["ab_mevzuat"],
            description="EU legal database",
            adapter_class="EURLexAdapter",
        ))

        # ====================================================================
        # COMMERCIAL (Optional - requires authentication)
        # ====================================================================

        self.register(SourceMetadata(
            source_id="kazanci",
            name="Kazancı",
            display_name="Kazancı İçtihat Bilgi Bankası",
            source_type=SourceType.COMMERCIAL,
            base_url="https://www.kazanci.com.tr",
            status=SourceStatus.REQUIRES_AUTH,
            requires_auth=True,
            auth_type="api_key",
            supports_search=True,
            supports_bulk_download=True,
            document_types=["kanun", "yargitay_karari", "danistay_karari"],
            description="Commercial legal database (requires subscription)",
        ))

        self.register(SourceMetadata(
            source_id="lexpera",
            name="Lexpera",
            display_name="Lexpera Hukuk Bilgi Sistemi",
            source_type=SourceType.COMMERCIAL,
            base_url="https://www.lexpera.com.tr",
            status=SourceStatus.REQUIRES_AUTH,
            requires_auth=True,
            auth_type="oauth",
            supports_search=True,
            document_types=["kanun", "yargitay_karari"],
            description="Commercial legal database (requires subscription)",
        ))

    # ========================================================================
    # REGISTRY OPERATIONS
    # ========================================================================

    def register(self, source: SourceMetadata):
        """
        Register a source.

        Args:
            source: Source metadata
        """
        self._sources[source.source_id] = source
        logger.debug(f"Registered source: {source.source_id}")

    def get(self, source_id: str) -> Optional[SourceMetadata]:
        """
        Get source by ID.

        Args:
            source_id: Source identifier

        Returns:
            SourceMetadata or None
        """
        return self._sources.get(source_id)

    def list_sources(
        self,
        source_type: Optional[SourceType] = None,
        status: Optional[SourceStatus] = None
    ) -> List[SourceMetadata]:
        """
        List all sources with optional filtering.

        Args:
            source_type: Filter by source type
            status: Filter by status

        Returns:
            List of matching sources
        """
        sources = list(self._sources.values())

        if source_type:
            sources = [s for s in sources if s.source_type == source_type]

        if status:
            sources = [s for s in sources if s.status == status]

        return sources

    def get_by_type(self, source_type: SourceType) -> List[SourceMetadata]:
        """Get all sources of a specific type."""
        return self.list_sources(source_type=source_type)

    def get_active_sources(self) -> List[SourceMetadata]:
        """Get all active sources."""
        return self.list_sources(status=SourceStatus.ACTIVE)

    def find_by_document_type(self, document_type: str) -> List[SourceMetadata]:
        """
        Find sources that provide a specific document type.

        Args:
            document_type: Document type (e.g., "kanun", "yargitay_karari")

        Returns:
            List of sources
        """
        return [
            source for source in self._sources.values()
            if document_type in source.document_types
        ]


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

_global_registry = None


def get_registry() -> SourceRegistry:
    """
    Get the global source registry instance.

    Returns:
        Global SourceRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SourceRegistry()
    return _global_registry


__all__ = [
    "SourceStatus",
    "SourceType",
    "SourceMetadata",
    "SourceRegistry",
    "get_registry",
]
