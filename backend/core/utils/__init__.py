"""
Utility functions for Turkish Legal AI.

This module provides common utility functions:
- Date/time utilities
- String manipulation
- File operations
- Validation helpers
- Turkish-specific utilities (TC No, IBAN, etc.)
- Encryption/decryption helpers
- JSON serialization
- UUID generation
- Hash functions

Usage:
    >>> from backend.core.utils import generate_uuid, validate_tc_no
    >>> 
    >>> user_id = generate_uuid()
    >>> is_valid = validate_tc_no("12345678901")
"""

import hashlib
import re
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any


# =============================================================================
# UUID UTILITIES
# =============================================================================

def generate_uuid() -> str:
    """
    Generate a UUID4 string.
    
    Returns:
        str: UUID4 string
        
    Example:
        >>> user_id = generate_uuid()
        >>> print(user_id)
        'a3f8d9e1-2b3c-4d5e-6f7a-8b9c0d1e2f3a'
    """
    return str(uuid.uuid4())


def is_valid_uuid(value: str) -> bool:
    """
    Check if a string is a valid UUID.
    
    Args:
        value: String to validate
        
    Returns:
        bool: True if valid UUID
        
    Example:
        >>> is_valid_uuid("a3f8d9e1-2b3c-4d5e-6f7a-8b9c0d1e2f3a")
        True
        >>> is_valid_uuid("not-a-uuid")
        False
    """
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


# =============================================================================
# DATE/TIME UTILITIES
# =============================================================================

def utc_now() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        datetime: Current UTC time
        
    Example:
        >>> now = utc_now()
        >>> print(now.tzinfo)
        UTC
    """
    return datetime.now(timezone.utc)


def timestamp_to_datetime(timestamp: int) -> datetime:
    """
    Convert Unix timestamp to datetime.
    
    Args:
        timestamp: Unix timestamp (seconds)
        
    Returns:
        datetime: Datetime object
        
    Example:
        >>> dt = timestamp_to_datetime(1698765432)
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: Datetime object
        
    Returns:
        int: Unix timestamp (seconds)
        
    Example:
        >>> timestamp = datetime_to_timestamp(utc_now())
    """
    return int(dt.timestamp())


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object
        fmt: Format string
        
    Returns:
        str: Formatted datetime
        
    Example:
        >>> format_datetime(utc_now())
        '2025-10-30 15:45:30'
    """
    return dt.strftime(fmt)


# =============================================================================
# STRING UTILITIES
# =============================================================================

def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to slugify
        
    Returns:
        str: Slugified text
        
    Example:
        >>> slugify("Türk Medeni Kanunu")
        'turk-medeni-kanunu'
    """
    # Turkish character replacements
    replacements = {
        'ı': 'i', 'İ': 'i', 'ğ': 'g', 'Ğ': 'g',
        'ü': 'u', 'Ü': 'u', 'ş': 's', 'Ş': 's',
        'ö': 'o', 'Ö': 'o', 'ç': 'c', 'Ç': 'c',
    }
    
    # Replace Turkish characters
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Convert to lowercase and replace spaces/special chars with hyphens
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    
    return text


def truncate(text: str, length: int, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        str: Truncated text
        
    Example:
        >>> truncate("This is a long text", 10)
        'This is...'
    """
    if len(text) <= length:
        return text
    
    return text[:length - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Replaces multiple spaces/newlines with single space.
    
    Args:
        text: Text to normalize
        
    Returns:
        str: Normalized text
        
    Example:
        >>> normalize_whitespace("Hello    world\\n\\nTest")
        'Hello world Test'
    """
    return re.sub(r'\s+', ' ', text).strip()


# =============================================================================
# TURKISH VALIDATION UTILITIES
# =============================================================================

def validate_tc_no(tc_no: str) -> bool:
    """
    Validate Turkish National ID number (TC Kimlik No) with enhanced algorithm.
    
    TC Kimlik No Algorithm:
    1. Must be exactly 11 digits
    2. First digit cannot be 0
    3. Sum of first 10 digits mod 10 must equal 11th digit
    4. ((sum of 1st,3rd,5th,7th,9th digits * 7) - (sum of 2nd,4th,6th,8th digits)) mod 10 
       must equal 10th digit
    
    Args:
        tc_no: TC Kimlik No to validate
        
    Returns:
        bool: True if valid TC Kimlik No
        
    Example:
        >>> validate_tc_no("12345678901")
        False  # Invalid checksum
        >>> validate_tc_no("10000000146")
        True   # Valid TC Kimlik No
    """
    # Input validation
    if not isinstance(tc_no, str):
        return False
    
    # Remove spaces and dashes
    tc_no = tc_no.replace(' ', '').replace('-', '')
    
    # Must be exactly 11 digits
    if not tc_no.isdigit() or len(tc_no) != 11:
        return False
    
    # First digit cannot be 0
    if tc_no[0] == '0':
        return False
    
    # Convert to digit array
    digits = [int(d) for d in tc_no]
    
    # Validate 10th digit (first checksum)
    # ((1+3+5+7+9)*7 - (2+4+6+8)) mod 10 = 10th digit
    odd_sum = sum(digits[0:9:2])   # 1st, 3rd, 5th, 7th, 9th positions (index 0,2,4,6,8)
    even_sum = sum(digits[1:8:2])  # 2nd, 4th, 6th, 8th positions (index 1,3,5,7)
    
    check_10th = ((odd_sum * 7) - even_sum) % 10
    if check_10th != digits[9]:
        return False
    
    # Validate 11th digit (second checksum)
    # Sum of first 10 digits mod 10 = 11th digit
    check_11th = sum(digits[0:10]) % 10
    if check_11th != digits[10]:
        return False
    
    return True


def validate_iban(iban: str) -> bool:
    """
    Validate Turkish IBAN.
    
    Turkish IBANs:
    - Start with "TR"
    - Total 26 characters (including TR)
    - Format: TR00 0000 0000 0000 0000 0000 00
    
    Args:
        iban: IBAN to validate
        
    Returns:
        bool: True if valid
        
    Example:
        >>> validate_iban("TR330006100519786457841326")
        True
    """
    # Remove spaces
    iban = iban.replace(' ', '').upper()
    
    # Must start with TR and be 26 characters
    if not iban.startswith('TR') or len(iban) != 26:
        return False
    
    # Check if all characters after TR are digits
    if not iban[2:].isdigit():
        return False
    
    # IBAN checksum validation (mod 97)
    # Move first 4 chars to end and convert letters to numbers
    rearranged = iban[4:] + iban[:4]
    
    # Convert to numbers (A=10, B=11, ..., Z=35)
    numeric = ''
    for char in rearranged:
        if char.isdigit():
            numeric += char
        else:
            numeric += str(ord(char) - ord('A') + 10)
    
    # Check if mod 97 equals 1
    return int(numeric) % 97 == 1


def validate_turkish_phone(phone: str) -> bool:
    """
    Validate Turkish phone number.
    
    Valid formats:
    - +90 5XX XXX XX XX
    - 0 5XX XXX XX XX
    - 5XX XXX XX XX
    
    Args:
        phone: Phone number to validate
        
    Returns:
        bool: True if valid
        
    Example:
        >>> validate_turkish_phone("+90 532 123 45 67")
        True
    """
    # Remove spaces, dashes, parentheses
    phone = re.sub(r'[\s\-\(\)]', '', phone)
    
    # Remove country code if present
    if phone.startswith('+90'):
        phone = phone[3:]
    elif phone.startswith('90'):
        phone = phone[2:]
    elif phone.startswith('0'):
        phone = phone[1:]
    
    # Must be 10 digits starting with 5
    if not phone.isdigit() or len(phone) != 10:
        return False
    
    if not phone.startswith('5'):
        return False
    
    return True


def format_turkish_phone(phone: str) -> str:
    """
    Format Turkish phone number to standard format.
    
    Args:
        phone: Phone number (any format)
        
    Returns:
        str: Formatted phone number (+90 5XX XXX XX XX)
        
    Example:
        >>> format_turkish_phone("5321234567")
        '+90 532 123 45 67'
    """
    # Clean phone
    phone = re.sub(r'[\s\-\(\)]', '', phone)
    
    # Remove country code if present
    if phone.startswith('+90'):
        phone = phone[3:]
    elif phone.startswith('90'):
        phone = phone[2:]
    elif phone.startswith('0'):
        phone = phone[1:]
    
    # Validate
    if not validate_turkish_phone(phone):
        return phone  # Return original if invalid
    
    # Format: +90 5XX XXX XX XX
    return f"+90 {phone[:3]} {phone[3:6]} {phone[6:8]} {phone[8:]}"


def validate_vkn(vkn: str) -> bool:
    """
    Validate Turkish Tax ID (Vergi Kimlik Numarası).
    
    VKN is 10 digits with a checksum algorithm.
    
    Args:
        vkn: Tax ID to validate
        
    Returns:
        bool: True if valid
        
    Example:
        >>> validate_vkn("1234567890")
        False  # Invalid checksum
    """
    # Remove spaces and dashes
    vkn = vkn.replace(' ', '').replace('-', '')
    
    # Must be exactly 10 digits
    if not vkn.isdigit() or len(vkn) != 10:
        return False
    
    # Convert to digit array
    digits = [int(d) for d in vkn]
    
    # VKN checksum algorithm
    v = [9, 8, 7, 6, 5, 4, 3, 2]
    sum_val = 0
    
    for i in range(8):
        temp = (digits[i] + v[i]) % 10
        sum_val += (temp * (2 ** (8 - i))) % 9
        if temp != 0 and (temp * (2 ** (8 - i))) % 9 == 0:
            sum_val += 9
    
    check_digit = (10 - (sum_val % 10)) % 10
    
    return check_digit == digits[9]


# =============================================================================
# HASH UTILITIES
# =============================================================================

def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    Hash a string using specified algorithm.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        str: Hex-encoded hash
        
    Example:
        >>> hash_string("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    hash_func = getattr(hashlib, algorithm)
    return hash_func(text.encode()).hexdigest()


def generate_random_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Token length
        
    Returns:
        str: Random token
        
    Example:
        >>> token = generate_random_token(32)
    """
    return secrets.token_urlsafe(length)


# =============================================================================
# FILE UTILITIES
# =============================================================================

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename
        
    Returns:
        str: File extension (lowercase, without dot)
        
    Example:
        >>> get_file_extension("document.PDF")
        'pdf'
    """
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


def is_allowed_file_type(filename: str, allowed_types: list[str]) -> bool:
    """
    Check if file type is allowed.
    
    Args:
        filename: Filename
        allowed_types: List of allowed extensions
        
    Returns:
        bool: True if allowed
        
    Example:
        >>> is_allowed_file_type("doc.pdf", ["pdf", "docx"])
        True
    """
    ext = get_file_extension(filename)
    return ext in [t.lower() for t in allowed_types]


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size
        
    Example:
        >>> format_file_size(1536)
        '1.5 KB'
        >>> format_file_size(1048576)
        '1.0 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
        
    Example:
        >>> sanitize_filename("my../../../etc/passwd")
        'my_etc_passwd'
    """
    # Remove path traversal attempts
    filename = filename.replace('..', '')
    filename = filename.replace('/', '_')
    filename = filename.replace('\\', '_')
    
    # Remove non-alphanumeric except .-_
    filename = re.sub(r'[^\w\s.-]', '', filename)
    
    # Replace multiple spaces/underscores
    filename = re.sub(r'[\s_]+', '_', filename)
    
    return filename.strip('._')


# =============================================================================
# JSON UTILITIES
# =============================================================================

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.
    
    Returns default value if parsing fails.
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
        
    Returns:
        Any: Parsed JSON or default
        
    Example:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_loads('invalid json', default={})
        {}
    """
    import json
    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def is_valid_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email to validate
        
    Returns:
        bool: True if valid
        
    Example:
        >>> is_valid_email("user@example.com")
        True
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if valid
        
    Example:
        >>> is_valid_url("https://example.com")
        True
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    Mask sensitive data for logging/display.
    
    Args:
        data: Data to mask
        mask_char: Character to use for masking
        visible_chars: Number of characters to show at end
        
    Returns:
        str: Masked data
        
    Example:
        >>> mask_sensitive_data("1234567890", visible_chars=4)
        '******7890'
        >>> mask_sensitive_data("user@example.com", visible_chars=4)
        '**************com'
    """
    if len(data) <= visible_chars:
        return mask_char * len(data)
    
    masked_length = len(data) - visible_chars
    return (mask_char * masked_length) + data[-visible_chars:]


def mask_email(email: str) -> str:
    """
    Mask email address for KVKK compliance.
    
    Args:
        email: Email address
        
    Returns:
        str: Masked email
        
    Example:
        >>> mask_email("john.doe@example.com")
        'j***@example.com'
    """
    if '@' not in email:
        return mask_sensitive_data(email)
    
    local, domain = email.split('@', 1)
    
    if len(local) <= 2:
        masked_local = local[0] + '*'
    else:
        masked_local = local[0] + ('*' * (len(local) - 1))
    
    return f"{masked_local}@{domain}"


def mask_phone(phone: str) -> str:
    """
    Mask phone number for KVKK compliance.
    
    Args:
        phone: Phone number
        
    Returns:
        str: Masked phone
        
    Example:
        >>> mask_phone("+90 532 123 45 67")
        '+90 532 *** ** 67'
    """
    # Extract digits
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) >= 10:
        # Show country code, first 3, and last 2
        return f"+90 {digits[2:5]} *** ** {digits[-2:]}"
    
    return mask_sensitive_data(phone)


def mask_tc_no(tc_no: str) -> str:
    """
    Mask TC Kimlik No for KVKK compliance.
    
    Args:
        tc_no: TC Kimlik No
        
    Returns:
        str: Masked TC No
        
    Example:
        >>> mask_tc_no("12345678901")
        '123*****901'
    """
    if len(tc_no) == 11:
        return f"{tc_no[:3]}*****{tc_no[-3:]}"
    
    return mask_sensitive_data(tc_no)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # UUID
    "generate_uuid",
    "is_valid_uuid",
    # Date/Time
    "utc_now",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    "format_datetime",
    # String
    "slugify",
    "truncate",
    "normalize_whitespace",
    # Turkish Validation
    "validate_tc_no",
    "validate_iban",
    "validate_turkish_phone",
    "format_turkish_phone",
    "validate_vkn",
    # Hash
    "hash_string",
    "generate_random_token",
    # File
    "get_file_extension",
    "is_allowed_file_type",
    "format_file_size",
    "sanitize_filename",
    # JSON
    "safe_json_loads",
    # Validation
    "is_valid_email",
    "is_valid_url",
    # Data Masking (KVKK)
    "mask_sensitive_data",
    "mask_email",
    "mask_phone",
    "mask_tc_no",
]