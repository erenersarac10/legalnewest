"""
Adapter Factory - Registry Pattern for Legal Source Adapters

This module implements a registry-based factory for dynamically selecting
the appropriate adapter based on legal source type. Supports all 16 Turkish
government legal sources with type-safe registration and retrieval.

World-Class Features:
    - Registry pattern with decorator-based registration
    - Type-safe adapter resolution
    - Lazy initialization for performance
    - Comprehensive error handling with actionable messages
    - Support for adapter metadata and capabilities

Example:
    >>> factory = AdapterFactory()
    >>> adapter = await factory.get_adapter("resmi_gazete")
    >>> document = await adapter.fetch_document("20180324")

    >>> # Or use source type enum
    >>> from backend.api.schemas.canonical import LegalSourceType
    >>> adapter = await factory.get_adapter(LegalSourceType.YARGITAY)

Architecture:
    AdapterFactory (Registry)
    ├── BaseAdapter (Abstract)
    │   ├── ResmiGazeteAdapter (Official Gazette - PDF)
    │   ├── MevzuatGovAdapter (Legislation - HTML)
    │   ├── YargitayAdapter (Supreme Court - JSON API)
    │   ├── DanistayAdapter (Council of State - HTML)
    │   ├── AYMAdapter (Constitutional Court - HTML)
    │   ├── KVKKAdapter (Data Protection - HTML)
    │   ├── TBMMAdapter (Parliament - XML)
    │   ├── BDDKAdapter (Banking Regulation - PDF)
    │   ├── SPKAdapter (Capital Markets - HTML)
    │   ├── EPDKAdapter (Energy Regulation - PDF)
    │   ├── BTKAdapter (Telecom Regulation - HTML)
    │   ├── RekabetAdapter (Competition Authority - PDF)
    │   ├── GIBAdapter (Revenue Admin - HTML)
    │   ├── SGKAdapter (Social Security - PDF)
    │   └── SayistayAdapter (Court of Accounts - PDF)

KVKK Compliance:
    - No PII processed in factory
    - Adapters responsible for their own data handling
    - Legal basis: Legitimate interest (Article 5/2-f)

Turkish Legal Context:
    Supports complete Turkish legal ecosystem:
    - Primary legislation (Resmi Gazete, Mevzuat)
    - Judicial decisions (Yargıtay, Danıştay, AYM)
    - Regulatory authorities (BDDK, SPK, EPDK, BTK, Rekabet)
    - Parliamentary documents (TBMM)
    - Public sector (GİB, SGK, Sayıştay)
    - Data protection (KVKK)
"""

from typing import Dict, Type, Optional, List
import asyncio
from abc import ABC

from backend.parsers.adapters.base_adapter import BaseAdapter
from backend.api.schemas.canonical import LegalSourceType


class AdapterNotFoundError(Exception):
    """
    Raised when requested adapter is not registered.

    Provides actionable error message with available adapters.
    """
    pass


class AdapterInitializationError(Exception):
    """
    Raised when adapter initialization fails.

    Wraps underlying errors with context.
    """
    pass


class AdapterMetadata:
    """
    Metadata for registered adapters.

    Tracks adapter capabilities, status, and configuration.

    Attributes:
        adapter_class: The adapter class type
        source_type: Legal source type enum
        description: Human-readable description
        supported_formats: List of document formats (PDF, HTML, JSON, XML)
        date_range: Tuple of (min_date, max_date) or None
        requires_auth: Whether adapter requires authentication
        is_enabled: Whether adapter is currently enabled
    """

    def __init__(
        self,
        adapter_class: Type[BaseAdapter],
        source_type: LegalSourceType,
        description: str,
        supported_formats: List[str],
        date_range: Optional[tuple] = None,
        requires_auth: bool = False,
        is_enabled: bool = True,
    ):
        self.adapter_class = adapter_class
        self.source_type = source_type
        self.description = description
        self.supported_formats = supported_formats
        self.date_range = date_range
        self.requires_auth = requires_auth
        self.is_enabled = is_enabled


class AdapterFactory:
    """
    Registry-based factory for legal source adapters.

    Implements the Factory + Registry pattern for dynamic adapter selection.
    Supports lazy initialization, type-safe retrieval, and comprehensive metadata.

    Thread-safe singleton pattern ensures single factory instance across application.

    Attributes:
        _registry: Dict mapping source names to adapter metadata
        _instances: Dict caching initialized adapter instances
        _lock: Async lock for thread-safe initialization

    Example:
        >>> factory = AdapterFactory()
        >>>
        >>> # Register adapter (usually done via decorator)
        >>> factory.register(
        ...     "resmi_gazete",
        ...     ResmiGazeteAdapter,
        ...     LegalSourceType.RESMI_GAZETE,
        ...     "Official Gazette - Primary legislation source",
        ...     ["PDF"],
        ...     date_range=(date(1920, 1, 1), date.today())
        ... )
        >>>
        >>> # Get adapter instance (lazy initialization)
        >>> adapter = await factory.get_adapter("resmi_gazete")
        >>>
        >>> # Check adapter capabilities
        >>> metadata = factory.get_metadata("resmi_gazete")
        >>> print(f"Supports: {metadata.supported_formats}")
        >>>
        >>> # List all available adapters
        >>> for source in factory.list_adapters():
        ...     print(f"{source}: {factory.get_metadata(source).description}")
    """

    # Singleton instance
    _instance: Optional["AdapterFactory"] = None

    def __new__(cls):
        """Singleton pattern - ensure single factory instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry and instance cache."""
        if self._initialized:
            return

        self._registry: Dict[str, AdapterMetadata] = {}
        self._instances: Dict[str, BaseAdapter] = {}
        self._lock = asyncio.Lock()
        self._initialized = True

    def register(
        self,
        source_name: str,
        adapter_class: Type[BaseAdapter],
        source_type: LegalSourceType,
        description: str,
        supported_formats: List[str],
        date_range: Optional[tuple] = None,
        requires_auth: bool = False,
        is_enabled: bool = True,
    ) -> None:
        """
        Register an adapter in the factory.

        Args:
            source_name: Unique identifier for the source (e.g., "resmi_gazete")
            adapter_class: The adapter class (must inherit from BaseAdapter)
            source_type: LegalSourceType enum value
            description: Human-readable description
            supported_formats: List of formats like ["PDF", "HTML", "JSON"]
            date_range: Optional (min_date, max_date) tuple
            requires_auth: Whether adapter needs authentication
            is_enabled: Whether adapter is enabled

        Raises:
            ValueError: If adapter_class doesn't inherit from BaseAdapter
            ValueError: If source_name already registered

        Example:
            >>> factory.register(
            ...     "yargitay",
            ...     YargitayAdapter,
            ...     LegalSourceType.YARGITAY,
            ...     "Supreme Court of Appeals - 1868-2025",
            ...     ["JSON", "HTML"],
            ...     date_range=(date(1868, 1, 1), date.today())
            ... )
        """
        # Validate adapter class
        if not issubclass(adapter_class, BaseAdapter):
            raise ValueError(
                f"Adapter class {adapter_class.__name__} must inherit from BaseAdapter"
            )

        # Check for duplicate registration
        if source_name in self._registry:
            raise ValueError(
                f"Adapter '{source_name}' already registered. "
                f"Use update() to modify existing registration."
            )

        # Create metadata and register
        metadata = AdapterMetadata(
            adapter_class=adapter_class,
            source_type=source_type,
            description=description,
            supported_formats=supported_formats,
            date_range=date_range,
            requires_auth=requires_auth,
            is_enabled=is_enabled,
        )

        self._registry[source_name] = metadata

        # Also register by enum value for convenience
        self._registry[source_type.value] = metadata

    async def get_adapter(
        self,
        source: str | LegalSourceType,
        force_new: bool = False
    ) -> BaseAdapter:
        """
        Get adapter instance for specified source.

        Implements lazy initialization - adapters are instantiated on first use
        and cached for subsequent requests. Thread-safe with async lock.

        Args:
            source: Source name string or LegalSourceType enum
            force_new: If True, create new instance instead of using cache

        Returns:
            Initialized adapter instance

        Raises:
            AdapterNotFoundError: If source not registered
            AdapterInitializationError: If adapter initialization fails

        Example:
            >>> # By string name
            >>> adapter = await factory.get_adapter("resmi_gazete")
            >>>
            >>> # By enum
            >>> adapter = await factory.get_adapter(LegalSourceType.YARGITAY)
            >>>
            >>> # Force new instance (useful for testing)
            >>> adapter = await factory.get_adapter("mevzuat_gov", force_new=True)
        """
        # Convert enum to string
        source_key = source.value if isinstance(source, LegalSourceType) else source

        # Check if adapter is registered
        if source_key not in self._registry:
            available = ", ".join(self.list_adapters())
            raise AdapterNotFoundError(
                f"Adapter '{source_key}' not found in registry. "
                f"Available adapters: {available}"
            )

        metadata = self._registry[source_key]

        # Check if adapter is enabled
        if not metadata.is_enabled:
            raise AdapterNotFoundError(
                f"Adapter '{source_key}' is registered but currently disabled. "
                f"Enable it first using enable_adapter()."
            )

        # Return cached instance if available
        if not force_new and source_key in self._instances:
            return self._instances[source_key]

        # Initialize new instance (thread-safe)
        async with self._lock:
            # Double-check after acquiring lock
            if not force_new and source_key in self._instances:
                return self._instances[source_key]

            try:
                # Instantiate adapter
                adapter = metadata.adapter_class()

                # Cache instance
                if not force_new:
                    self._instances[source_key] = adapter

                return adapter

            except Exception as e:
                raise AdapterInitializationError(
                    f"Failed to initialize adapter '{source_key}': {str(e)}"
                ) from e

    def get_metadata(self, source: str | LegalSourceType) -> AdapterMetadata:
        """
        Get metadata for registered adapter.

        Args:
            source: Source name string or LegalSourceType enum

        Returns:
            AdapterMetadata instance

        Raises:
            AdapterNotFoundError: If source not registered

        Example:
            >>> metadata = factory.get_metadata("yargitay")
            >>> print(f"Date range: {metadata.date_range}")
            >>> print(f"Formats: {metadata.supported_formats}")
        """
        source_key = source.value if isinstance(source, LegalSourceType) else source

        if source_key not in self._registry:
            available = ", ".join(self.list_adapters())
            raise AdapterNotFoundError(
                f"Adapter '{source_key}' not found. Available: {available}"
            )

        return self._registry[source_key]

    def list_adapters(self, include_disabled: bool = False) -> List[str]:
        """
        List all registered adapter names.

        Args:
            include_disabled: If True, include disabled adapters

        Returns:
            List of adapter source names

        Example:
            >>> adapters = factory.list_adapters()
            >>> print(f"Available: {len(adapters)} adapters")
            >>> for adapter in adapters:
            ...     print(f"  - {adapter}")
        """
        # Get unique source names (avoid enum duplicates)
        unique_sources = set()
        for source_key, metadata in self._registry.items():
            # Skip enum values (only include string keys)
            if not isinstance(source_key, str) or source_key == metadata.source_type.value:
                continue

            # Skip disabled if requested
            if not include_disabled and not metadata.is_enabled:
                continue

            unique_sources.add(source_key)

        return sorted(unique_sources)

    def enable_adapter(self, source: str | LegalSourceType) -> None:
        """
        Enable a disabled adapter.

        Args:
            source: Source name string or LegalSourceType enum

        Raises:
            AdapterNotFoundError: If source not registered
        """
        source_key = source.value if isinstance(source, LegalSourceType) else source

        if source_key not in self._registry:
            raise AdapterNotFoundError(f"Adapter '{source_key}' not found")

        self._registry[source_key].is_enabled = True

    def disable_adapter(self, source: str | LegalSourceType) -> None:
        """
        Disable an adapter (won't be returned by get_adapter).

        Args:
            source: Source name string or LegalSourceType enum

        Raises:
            AdapterNotFoundError: If source not registered
        """
        source_key = source.value if isinstance(source, LegalSourceType) else source

        if source_key not in self._registry:
            raise AdapterNotFoundError(f"Adapter '{source_key}' not found")

        self._registry[source_key].is_enabled = False

        # Remove from cache if present
        if source_key in self._instances:
            del self._instances[source_key]

    async def clear_cache(self, source: Optional[str | LegalSourceType] = None) -> None:
        """
        Clear cached adapter instance(s).

        Useful for testing or when adapter configuration changes.

        Args:
            source: If provided, clear only this adapter. If None, clear all.

        Example:
            >>> # Clear specific adapter
            >>> await factory.clear_cache("resmi_gazete")
            >>>
            >>> # Clear all cached adapters
            >>> await factory.clear_cache()
        """
        async with self._lock:
            if source is None:
                self._instances.clear()
            else:
                source_key = source.value if isinstance(source, LegalSourceType) else source
                if source_key in self._instances:
                    del self._instances[source_key]


# Global factory instance
_factory = AdapterFactory()


def get_factory() -> AdapterFactory:
    """
    Get global AdapterFactory instance.

    Returns:
        Singleton AdapterFactory instance

    Example:
        >>> from backend.parsers.adapters.adapter_factory import get_factory
        >>>
        >>> factory = get_factory()
        >>> adapter = await factory.get_adapter("resmi_gazete")
    """
    return _factory


# Auto-register all adapters on import
def _register_default_adapters():
    """
    Register all default adapters.

    Called automatically on module import.
    Imports are lazy to avoid circular dependencies.
    """
    from datetime import date

    # Import adapters (lazy to avoid circular imports at module level)
    try:
        from backend.parsers.adapters.resmi_gazete_adapter import ResmiGazeteAdapter

        _factory.register(
            "resmi_gazete",
            ResmiGazeteAdapter,
            LegalSourceType.RESMI_GAZETE,
            "T.C. Resmi Gazete - Official Gazette (1920-2025) - Primary legislation source",
            ["PDF"],
            date_range=(date(1920, 1, 1), date.today()),
            requires_auth=False,
            is_enabled=True,
        )
    except ImportError:
        pass  # Adapter not yet implemented

    try:
        from backend.parsers.adapters.mevzuat_gov_adapter import MevzuatGovAdapter

        _factory.register(
            "mevzuat_gov",
            MevzuatGovAdapter,
            LegalSourceType.MEVZUAT_GOV,
            "Mevzuat Bilgi Sistemi - Consolidated legislation (1920-2025)",
            ["HTML", "XML"],
            date_range=(date(1920, 1, 1), date.today()),
            requires_auth=False,
            is_enabled=True,
        )
    except ImportError:
        pass

    # Yargıtay - Supreme Court of Appeals
    try:
        from backend.parsers.adapters.yargitay_adapter import YargitayAdapter

        _factory.register(
            "yargitay",
            YargitayAdapter,
            LegalSourceType.YARGITAY,
            "Yargıtay - Supreme Court of Appeals (1868-2025) - İçtihat database",
            ["JSON", "HTML"],
            date_range=(date(1868, 1, 1), date.today()),
            requires_auth=False,
            is_enabled=True,
        )
    except ImportError:
        pass

    # Danıştay - Council of State
    try:
        from backend.parsers.adapters.danistay_adapter import DanistayAdapter

        _factory.register(
            "danistay",
            DanistayAdapter,
            LegalSourceType.DANISTAY,
            "Danıştay - Council of State - Administrative law decisions",
            ["HTML"],
            requires_auth=False,
            is_enabled=True,
        )
    except ImportError:
        pass

    # AYM - Constitutional Court
    try:
        from backend.parsers.adapters.aym_adapter import AYMAdapter

        _factory.register(
            "aym",
            AYMAdapter,
            LegalSourceType.AYM,
            "Anayasa Mahkemesi - Constitutional Court decisions",
            ["HTML", "PDF"],
            requires_auth=False,
            is_enabled=True,
        )
    except ImportError:
        pass

    # Additional adapters will be registered as they're implemented
    # KVKK, TBMM, BDDK, SPK, EPDK, BTK, Rekabet, GİB, SGK, Sayıştay


# Auto-register on import
_register_default_adapters()
