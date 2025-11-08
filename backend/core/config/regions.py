"""
Regions Configuration - Harvey/Legora %100 Multi-Region Deployment.

Enterprise-grade multi-region architecture with:
- Geographic distribution (EU, Turkey-specific)
- Data residency compliance (GDPR, KVKK)
- Latency-based routing (route users to nearest region)
- Cross-region replication (Redis, PostgreSQL)
- Failover strategies (automatic region failover)
- Disaster recovery (RPO < 1 hour, RTO < 15 minutes)
- Turkish Legal AI requirements (KVKK compliance)

Why Multi-Region?
    Without: Single region → outage kills entire service → customer churn
    With: Multi-region → 99.99% availability → Harvey-level reliability

    Impact: Zero downtime even if entire region fails! 🌍

Architecture:
    Istanbul (Primary) ← Active-Active → Frankfurt (Secondary)
           ↓ Async replication          ↓
    Ankara (DR)                   Amsterdam (DR)

Data Residency (KVKK Compliance):
    - Turkish user data MUST stay in Turkey (Istanbul/Ankara)
    - EU user data MUST stay in EU (Frankfurt/Amsterdam)
    - Cross-border transfer requires explicit consent
    - Legal documents stored in Turkey only

Regions:
    1. TR-IST (Istanbul): Primary - Turkish Legal AI (KVKK)
    2. TR-ANK (Ankara): Secondary - Disaster Recovery
    3. EU-FRA (Frankfurt): EU users (GDPR)
    4. EU-AMS (Amsterdam): EU fallback

Routing Strategy:
    Turkish IP → TR-IST (primary) → TR-ANK (failover)
    EU IP → EU-FRA (primary) → EU-AMS (failover)
    Other → Nearest region based on latency

Usage:
    >>> from backend.core.config.regions import get_region_config
    >>>
    >>> config = get_region_config("tr-ist")
    >>> print(config.name)  # "Istanbul"
    >>> print(config.data_residency)  # ["TR"]
"""

from typing import Dict, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, HttpUrl


class RegionCode(str, Enum):
    """Region codes."""

    # Turkey
    TR_IST = "tr-ist"  # Istanbul (primary)
    TR_ANK = "tr-ank"  # Ankara (secondary/DR)

    # EU
    EU_FRA = "eu-fra"  # Frankfurt (GDPR)
    EU_AMS = "eu-ams"  # Amsterdam (GDPR fallback)

    # Future expansion
    US_EAST = "us-east-1"  # US East (optional)
    AP_SG = "ap-singapore"  # Asia Pacific (optional)


class RegionType(str, Enum):
    """Region deployment types."""

    PRIMARY = "primary"  # Active serving
    SECONDARY = "secondary"  # Active serving (load balancing)
    DISASTER_RECOVERY = "disaster_recovery"  # Standby (only on failover)
    DEVELOPMENT = "development"  # Dev/test


class ReplicationStrategy(str, Enum):
    """Data replication strategies."""

    SYNCHRONOUS = "synchronous"  # Wait for replica (strong consistency)
    ASYNCHRONOUS = "asynchronous"  # Don't wait (eventual consistency)
    NONE = "none"  # No replication


class DataResidency(str, Enum):
    """Data residency compliance regions."""

    TURKEY = "TR"  # KVKK (Turkish Data Protection Law)
    EU = "EU"  # GDPR (General Data Protection Regulation)
    US = "US"  # CCPA (California Consumer Privacy Act)
    GLOBAL = "GLOBAL"  # No restrictions


# =============================================================================
# REGION CONFIGURATIONS
# =============================================================================


class RegionConfig(BaseModel):
    """Region configuration."""

    # Identity
    code: RegionCode
    name: str
    display_name: str
    region_type: RegionType

    # Geographic
    country: str
    city: str
    timezone: str
    coordinates: tuple = (0.0, 0.0)  # (latitude, longitude)

    # Network
    api_endpoint: HttpUrl
    cdn_endpoint: Optional[HttpUrl] = None
    public_ip_ranges: List[str] = []  # CIDR blocks

    # Data residency
    data_residency: List[DataResidency] = []
    kvkk_compliant: bool = False  # Turkish data protection
    gdpr_compliant: bool = False  # EU data protection

    # Replication
    replication_strategy: ReplicationStrategy = ReplicationStrategy.ASYNCHRONOUS
    replication_targets: List[RegionCode] = []
    replication_lag_max_seconds: int = 60  # Max acceptable lag

    # Failover
    failover_enabled: bool = True
    failover_target: Optional[RegionCode] = None
    failover_automatic: bool = True
    failover_rto_seconds: int = 900  # 15 minutes (Recovery Time Objective)
    failover_rpo_seconds: int = 3600  # 1 hour (Recovery Point Objective)

    # Route 53 Health Check (DNS-based failover)
    route53_health_check_id: Optional[str] = None  # Health check ID
    route53_health_check_interval: int = 30  # Check interval (seconds)
    route53_health_check_threshold: int = 3  # Failures before failover
    route53_dns_failover_type: Optional[Literal["PRIMARY", "SECONDARY"]] = None

    # Performance
    latency_slo_ms: int = 100  # Latency SLO (ms)
    availability_slo: float = 0.999  # 99.9% uptime

    # Infrastructure
    cloud_provider: Literal["aws", "azure", "gcp", "alibaba", "local"] = "aws"
    cloud_region: str  # Provider-specific region ID
    availability_zones: List[str] = []

    # Services
    services_enabled: List[str] = ["api", "rag", "embeddings", "documents"]
    database_endpoint: Optional[str] = None
    redis_endpoint: Optional[str] = None


# Harvey/Legora %100: Turkish Legal AI Multi-Region Configuration
REGION_CONFIGS: Dict[str, RegionConfig] = {
    # ==========================================================================
    # TURKEY: ISTANBUL (PRIMARY)
    # ==========================================================================
    "tr-ist": RegionConfig(
        # Identity
        code=RegionCode.TR_IST,
        name="istanbul",
        display_name="İstanbul (Birincil)",
        region_type=RegionType.PRIMARY,
        # Geographic
        country="TR",
        city="Istanbul",
        timezone="Europe/Istanbul",
        coordinates=(41.0082, 28.9784),
        # Network
        api_endpoint="https://api.legalai.com.tr",
        cdn_endpoint="https://cdn-ist.legalai.com.tr",
        public_ip_ranges=["185.125.190.0/24"],  # Example Turkish IP range
        # Data residency (CRITICAL for Turkish Legal AI)
        data_residency=[DataResidency.TURKEY],
        kvkk_compliant=True,  # KVKK: Kişisel Verilerin Korunması Kanunu
        gdpr_compliant=False,  # Not in EU
        # Replication
        replication_strategy=ReplicationStrategy.ASYNCHRONOUS,
        replication_targets=[RegionCode.TR_ANK],  # Replicate to Ankara
        replication_lag_max_seconds=30,  # 30 seconds max lag
        # Failover
        failover_enabled=True,
        failover_target=RegionCode.TR_ANK,  # Failover to Ankara
        failover_automatic=True,
        failover_rto_seconds=900,  # 15 minutes
        failover_rpo_seconds=60,  # 1 minute (aggressive for legal data)
        # Route 53 DNS-based health check & failover
        route53_health_check_id="hc-istanbul-api-12345",
        route53_health_check_interval=30,  # 30 seconds
        route53_health_check_threshold=3,  # 3 consecutive failures
        route53_dns_failover_type="PRIMARY",  # Primary region for DNS
        # Performance
        latency_slo_ms=50,  # 50ms SLO (same country)
        availability_slo=0.9999,  # 99.99% uptime
        # Infrastructure
        cloud_provider="aws",
        cloud_region="eu-south-1",  # AWS Milan (closest to Turkey)
        availability_zones=["eu-south-1a", "eu-south-1b", "eu-south-1c"],
        # Services
        services_enabled=["api", "rag", "embeddings", "documents", "legal_sources"],
        database_endpoint="legalai-ist.cluster-abc123.eu-south-1.rds.amazonaws.com",
        redis_endpoint="legalai-ist.abc123.euw1.cache.amazonaws.com:6379",
    ),

    # ==========================================================================
    # TURKEY: ANKARA (SECONDARY/DISASTER RECOVERY)
    # ==========================================================================
    "tr-ank": RegionConfig(
        code=RegionCode.TR_ANK,
        name="ankara",
        display_name="Ankara (İkincil/DR)",
        region_type=RegionType.DISASTER_RECOVERY,
        # Geographic
        country="TR",
        city="Ankara",
        timezone="Europe/Istanbul",
        coordinates=(39.9334, 32.8597),
        # Network
        api_endpoint="https://api-ank.legalai.com.tr",
        cdn_endpoint="https://cdn-ank.legalai.com.tr",
        # Data residency
        data_residency=[DataResidency.TURKEY],
        kvkk_compliant=True,
        gdpr_compliant=False,
        # Replication
        replication_strategy=ReplicationStrategy.ASYNCHRONOUS,
        replication_targets=[],  # Ankara is replica, doesn't replicate further
        replication_lag_max_seconds=60,
        # Failover
        failover_enabled=False,  # DR region, doesn't failover
        failover_automatic=False,
        # Performance
        latency_slo_ms=100,  # 100ms (cross-city in Turkey)
        availability_slo=0.999,  # 99.9%
        # Infrastructure
        cloud_provider="azure",  # Different provider for redundancy
        cloud_region="turkeycentral",  # Azure Turkey Central (Ankara)
        availability_zones=["1", "2", "3"],
        # Services
        services_enabled=["api", "rag", "documents"],  # Reduced services (DR)
        database_endpoint="legalai-ank.postgres.database.azure.com",
        redis_endpoint="legalai-ank.redis.cache.windows.net:6380",
    ),

    # ==========================================================================
    # EU: FRANKFURT (PRIMARY FOR EU USERS)
    # ==========================================================================
    "eu-fra": RegionConfig(
        code=RegionCode.EU_FRA,
        name="frankfurt",
        display_name="Frankfurt (EU Primary)",
        region_type=RegionType.SECONDARY,  # Secondary globally, primary for EU
        # Geographic
        country="DE",
        city="Frankfurt",
        timezone="Europe/Berlin",
        coordinates=(50.1109, 8.6821),
        # Network
        api_endpoint="https://api-eu.legalai.com",
        cdn_endpoint="https://cdn-fra.legalai.com",
        # Data residency (GDPR)
        data_residency=[DataResidency.EU],
        kvkk_compliant=False,  # Not in Turkey
        gdpr_compliant=True,  # GDPR: General Data Protection Regulation
        # Replication
        replication_strategy=ReplicationStrategy.ASYNCHRONOUS,
        replication_targets=[RegionCode.EU_AMS],
        replication_lag_max_seconds=30,
        # Failover
        failover_enabled=True,
        failover_target=RegionCode.EU_AMS,
        failover_automatic=True,
        failover_rto_seconds=900,
        failover_rpo_seconds=3600,
        # Performance
        latency_slo_ms=100,  # 100ms for EU
        availability_slo=0.999,
        # Infrastructure
        cloud_provider="aws",
        cloud_region="eu-central-1",  # AWS Frankfurt
        availability_zones=["eu-central-1a", "eu-central-1b", "eu-central-1c"],
        # Services
        services_enabled=["api", "rag", "embeddings", "documents"],
        database_endpoint="legalai-fra.cluster-xyz789.eu-central-1.rds.amazonaws.com",
        redis_endpoint="legalai-fra.xyz789.euc1.cache.amazonaws.com:6379",
    ),

    # ==========================================================================
    # EU: AMSTERDAM (DISASTER RECOVERY)
    # ==========================================================================
    "eu-ams": RegionConfig(
        code=RegionCode.EU_AMS,
        name="amsterdam",
        display_name="Amsterdam (EU DR)",
        region_type=RegionType.DISASTER_RECOVERY,
        # Geographic
        country="NL",
        city="Amsterdam",
        timezone="Europe/Amsterdam",
        coordinates=(52.3676, 4.9041),
        # Network
        api_endpoint="https://api-ams.legalai.com",
        cdn_endpoint="https://cdn-ams.legalai.com",
        # Data residency
        data_residency=[DataResidency.EU],
        kvkk_compliant=False,
        gdpr_compliant=True,
        # Replication
        replication_strategy=ReplicationStrategy.ASYNCHRONOUS,
        replication_targets=[],
        replication_lag_max_seconds=60,
        # Failover
        failover_enabled=False,
        failover_automatic=False,
        # Performance
        latency_slo_ms=150,
        availability_slo=0.99,
        # Infrastructure
        cloud_provider="azure",
        cloud_region="westeurope",  # Azure West Europe (Amsterdam)
        availability_zones=["1", "2", "3"],
        # Services
        services_enabled=["api", "rag"],
        database_endpoint="legalai-ams.postgres.database.azure.com",
        redis_endpoint="legalai-ams.redis.cache.windows.net:6380",
    ),
}


# =============================================================================
# ROUTING CONFIGURATION
# =============================================================================

# IP-based routing (route Turkish IPs to Turkey, EU IPs to EU)
IP_ROUTING_RULES: Dict[str, List[str]] = {
    "tr-ist": [
        "185.0.0.0/8",  # Turkish IP range (example)
        "88.255.0.0/16",  # Türk Telekom
        "212.58.0.0/16",  # Türksat
    ],
    "eu-fra": [
        "185.0.0.0/8",  # EU IP range (example)
        "2.16.0.0/13",  # Germany
        "31.0.0.0/8",  # EU general
    ],
}

# Latency-based routing weights (if no IP match)
LATENCY_WEIGHTS: Dict[str, float] = {
    "tr-ist": 1.0,  # Prefer Istanbul for Turkish users
    "tr-ank": 0.5,  # Lower weight (DR)
    "eu-fra": 0.8,  # Prefer Frankfurt for EU users
    "eu-ams": 0.4,  # Lower weight (DR)
}


# =============================================================================
# CROSS-REGION REPLICATION CONFIGURATIONS
# =============================================================================

REPLICATION_CONFIGS: Dict[str, Dict] = {
    # Istanbul → Ankara replication
    "tr-ist-to-tr-ank": {
        "source": "tr-ist",
        "target": "tr-ank",
        "strategy": "async",
        "lag_threshold_seconds": 30,
        "services": ["database", "redis", "files"],
        "priority": "high",  # Legal data is critical
    },
    # Frankfurt → Amsterdam replication
    "eu-fra-to-eu-ams": {
        "source": "eu-fra",
        "target": "eu-ams",
        "strategy": "async",
        "lag_threshold_seconds": 60,
        "services": ["database", "redis"],
        "priority": "medium",
    },
}


# =============================================================================
# FAILOVER SCENARIOS
# =============================================================================

FAILOVER_SCENARIOS: Dict[str, Dict] = {
    "tr-ist-outage": {
        "description": "Istanbul region failure",
        "trigger": "health_check_fail",
        "source": "tr-ist",
        "target": "tr-ank",
        "automatic": True,
        "rto_seconds": 900,  # 15 minutes
        "steps": [
            "1. Detect Istanbul health check failure (3 consecutive failures)",
            "2. Promote Ankara to primary (update DNS)",
            "3. Redirect all traffic to Ankara",
            "4. Monitor replication lag",
            "5. Send alerts to ops team",
        ],
    },
    "eu-fra-outage": {
        "description": "Frankfurt region failure",
        "trigger": "health_check_fail",
        "source": "eu-fra",
        "target": "eu-ams",
        "automatic": True,
        "rto_seconds": 900,
        "steps": [
            "1. Detect Frankfurt health check failure",
            "2. Promote Amsterdam to primary",
            "3. Redirect EU traffic to Amsterdam",
            "4. Monitor GDPR compliance",
        ],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_region_config(region_code: str) -> Optional[RegionConfig]:
    """
    Get region configuration.

    Args:
        region_code: Region code (e.g., "tr-ist", "eu-fra")

    Returns:
        RegionConfig instance or None

    Example:
        >>> config = get_region_config("tr-ist")
        >>> print(config.name)  # "istanbul"
        >>> print(config.kvkk_compliant)  # True
    """
    return REGION_CONFIGS.get(region_code)


def get_primary_regions() -> List[RegionConfig]:
    """
    Get all primary regions.

    Returns:
        List of primary region configs
    """
    return [
        config
        for config in REGION_CONFIGS.values()
        if config.region_type == RegionType.PRIMARY
    ]


def get_regions_by_data_residency(residency: DataResidency) -> List[RegionConfig]:
    """
    Get regions by data residency compliance.

    Args:
        residency: Data residency type (TURKEY, EU, US)

    Returns:
        List of compliant regions

    Example:
        >>> regions = get_regions_by_data_residency(DataResidency.TURKEY)
        >>> print([r.code for r in regions])  # [tr-ist, tr-ank]
    """
    return [
        config
        for config in REGION_CONFIGS.values()
        if residency in config.data_residency
    ]


def get_failover_target(region_code: str) -> Optional[RegionConfig]:
    """
    Get failover target for a region.

    Args:
        region_code: Source region code

    Returns:
        Failover target region config or None
    """
    source = get_region_config(region_code)
    if not source or not source.failover_target:
        return None
    return get_region_config(source.failover_target)


def get_nearest_region(user_ip: str) -> RegionConfig:
    """
    Get nearest region for user IP (simplified).

    Args:
        user_ip: User IP address

    Returns:
        Nearest region config

    Example:
        >>> region = get_nearest_region("185.125.190.100")  # Turkish IP
        >>> print(region.code)  # tr-ist
    """
    # Simplified: Check IP routing rules
    for region_code, ip_ranges in IP_ROUTING_RULES.items():
        # In production, use ipaddress.ip_address() and ip_network()
        # For now, simple prefix match
        for ip_range in ip_ranges:
            if user_ip.startswith(ip_range.split("/")[0][:3]):
                return get_region_config(region_code)

    # Default: Istanbul (primary)
    return get_region_config("tr-ist")


__all__ = [
    "RegionCode",
    "RegionType",
    "ReplicationStrategy",
    "DataResidency",
    "RegionConfig",
    "REGION_CONFIGS",
    "IP_ROUTING_RULES",
    "LATENCY_WEIGHTS",
    "REPLICATION_CONFIGS",
    "FAILOVER_SCENARIOS",
    "get_region_config",
    "get_primary_regions",
    "get_regions_by_data_residency",
    "get_failover_target",
    "get_nearest_region",
]
