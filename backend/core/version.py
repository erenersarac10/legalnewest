"""
Semantic versioning for Turkish Legal AI.

This module provides version information using semantic versioning (semver)
specification. The version is the single source of truth for the entire
application.

Version Format: MAJOR.MINOR.PATCH (e.g., "1.0.0")
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

Usage:
    >>> from backend.core.version import __version__, get_full_version
    >>> __version__
    '1.0.0'
    >>> get_full_version()
    '1.0.0 (Genesis) | python=3.11.x | git=abc1234 | build=2025-10-30T12:00:00Z'
"""
import sys
from typing import Final

# =============================================================================
# VERSION - SINGLE SOURCE OF TRUTH
# =============================================================================

__version__: Final[str] = "1.0.0"
"""
Application version in semantic versioning format.

IMPORTANT: This must be kept in sync with pyproject.toml.
CI/CD should validate version consistency across files.
"""

__version_info__: Final[tuple[int, int, int]] = (1, 0, 0)
"""Version as a tuple of integers (major, minor, patch)."""

# =============================================================================
# RELEASE INFORMATION
# =============================================================================

CODENAME: Final[str] = "Genesis"
"""
Release codename.

This should be updated for each major release:
- 1.x.x: Genesis
- 2.x.x: Evolution
- 3.x.x: Transcendence
"""

API_VERSION: Final[str] = "v1"
"""Current API version prefix."""

# =============================================================================
# BUILD METADATA (CI/CD injected)
# =============================================================================

BUILD_DATE: Final[str] = "2025-10-30T00:00:00Z"
"""Build timestamp in ISO 8601 UTC format. Set by CI/CD."""

GIT_COMMIT: Final[str] = "unknown"
"""Git commit SHA (short). Set by CI/CD at build time."""

GIT_BRANCH: Final[str] = "main"
"""Git branch name. Set by CI/CD."""

# =============================================================================
# PYTHON COMPATIBILITY
# =============================================================================

MIN_PYTHON_VERSION: Final[tuple[int, int]] = (3, 11)
"""Minimum required Python version (inclusive)."""

MAX_PYTHON_VERSION: Final[tuple[int, int]] = (3, 12)
"""Maximum supported Python version (inclusive)."""

# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def get_version() -> str:
    """
    Get the current version string.
    
    Returns exactly the __version__ constant with no modifications.
    This is the canonical way to retrieve the application version.
    
    Returns:
        str: Version string in semver format (e.g., "1.0.0")
    
    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_full_version() -> str:
    """
    Get full version string with metadata for operational visibility.
    
    Format: "VERSION (CODENAME) | python=X.Y.Z | git=SHORT_SHA | build=ISO8601"
    
    This provides complete version information for:
    - Log aggregation
    - Error tracking (Sentry tags)
    - Health check responses
    - Debugging production issues
    
    Returns:
        str: Full version with build metadata
        
    Example:
        >>> get_full_version()
        '1.0.0 (Genesis) | python=3.11.7 | git=abc1234 | build=2025-10-30T12:00:00Z'
    """
    # Base: version + codename (required by tests)
    parts = [f"{__version__} ({CODENAME})"]
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    parts.append(f"python={py_version}")
    
    # Git commit (if available)
    if GIT_COMMIT and GIT_COMMIT != "unknown":
        short_commit = GIT_COMMIT[:7]
        parts.append(f"git={short_commit}")
    
    # Build timestamp
    if BUILD_DATE:
        parts.append(f"build={BUILD_DATE}")
    
    return " | ".join(parts)


def is_compatible_python(major: int, minor: int) -> bool:
    """
    Check if given Python version is compatible.
    
    Compatible range: 3.11 <= version <= 3.12 (inclusive both ends)
    
    Args:
        major: Python major version (e.g., 3)
        minor: Python minor version (e.g., 11)
        
    Returns:
        bool: True if version is compatible, False otherwise
        
    Examples:
        >>> is_compatible_python(3, 11)
        True
        >>> is_compatible_python(3, 12)
        True
        >>> is_compatible_python(3, 10)
        False
        >>> is_compatible_python(3, 13)
        False
    """
    version = (major, minor)
    return MIN_PYTHON_VERSION <= version <= MAX_PYTHON_VERSION


def get_version_dict() -> dict[str, str | tuple[int, int, int] | bool]:
    """
    Get version information as a structured dictionary.
    
    Useful for:
    - API health check responses
    - Structured logging
    - Metrics labels
    - System info endpoints
    
    Returns:
        dict: Dictionary containing all version information
        
    Example:
        >>> get_version_dict()
        {
            'version': '1.0.0',
            'version_info': (1, 0, 0),
            'codename': 'Genesis',
            'api_version': 'v1',
            'build_date': '2025-10-30T00:00:00Z',
            'git_commit': 'abc1234',
            'git_branch': 'main',
            'python_version': '3.11.7',
            'python_compatible': True,
            'full_version': '1.0.0 (Genesis) | python=3.11.7 | ...'
        }
    """
    return {
        "version": __version__,
        "version_info": __version_info__,
        "codename": CODENAME,
        "api_version": API_VERSION,
        "build_date": BUILD_DATE,
        "git_commit": GIT_COMMIT,
        "git_branch": GIT_BRANCH,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_compatible": is_compatible_python(
            sys.version_info.major, 
            sys.version_info.minor
        ),
        "full_version": get_full_version(),
    }


def validate_version_format() -> bool:
    """
    Validate that __version__ follows strict semver format.
    
    This is used by CI/CD to ensure version string is valid.
    
    Returns:
        bool: True if version is valid semver, False otherwise
        
    Example:
        >>> validate_version_format()
        True
    """
    try:
        parts = __version__.split(".")
        if len(parts) != 3:
            return False
        
        # All parts must be non-negative integers
        major, minor, patch = parts
        return (
            major.isdigit() and
            minor.isdigit() and
            patch.isdigit() and
            int(major) >= 0 and
            int(minor) >= 0 and
            int(patch) >= 0
        )
    except (ValueError, AttributeError):
        return False


# =============================================================================
# RUNTIME VALIDATION (Development only)
# =============================================================================

def _check_python_compatibility() -> None:
    """
    Check Python version compatibility at import time.
    
    Raises:
        RuntimeError: If current Python version is not compatible
    """
    current_version = (sys.version_info.major, sys.version_info.minor)
    
    if not is_compatible_python(*current_version):
        raise RuntimeError(
            f"Python {current_version[0]}.{current_version[1]} is not supported. "
            f"Required: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} - "
            f"{MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}"
        )


# Validate Python version on import (fail fast)
_check_python_compatibility()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Version constants
    "__version__",
    "__version_info__",
    "CODENAME",
    "API_VERSION",
    
    # Build metadata
    "BUILD_DATE",
    "GIT_COMMIT",
    "GIT_BRANCH",
    
    # Compatibility
    "MIN_PYTHON_VERSION",
    "MAX_PYTHON_VERSION",
    
    # Functions
    "get_version",
    "get_version_info",
    "get_full_version",
    "is_compatible_python",
    "get_version_dict",
    "validate_version_format",
]


# Alias for backward compatibility
def get_version_info() -> tuple[int, int, int]:
    """
    Get version information as a tuple.
    
    Returns:
        tuple[int, int, int]: Tuple of (major, minor, patch)
    
    Example:
        >>> get_version_info()
        (1, 0, 0)
    """
    return __version_info__      