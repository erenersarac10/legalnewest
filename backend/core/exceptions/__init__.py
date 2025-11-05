"""
Custom exceptions for Turkish Legal AI.

This module provides a comprehensive exception hierarchy:
- Base exception class
- HTTP exceptions (400, 401, 403, 404, 409, 422, 429, 500, 503)
- Database exceptions
- Cache exceptions
- Security exceptions
- Business logic exceptions
- Document processing exceptions
- LLM exceptions
- Turkish legal specific exceptions

All exceptions include:
- Error code
- Status code
- Detailed error message
- Additional context (details dict)
- Serialization support

Usage:
    >>> from backend.core.exceptions import NotFoundException
    >>> 
    >>> if not user:
    ...     raise NotFoundException(
    ...         message="Kullanıcı bulunamadı",
    ...         details={"user_id": user_id}
    ...     )
"""
from typing import Any

from backend.core.constants import (
    ERR_CONFLICT,
    ERR_DOCUMENT_INVALID_FORMAT,
    ERR_DOCUMENT_PARSE_FAILED,
    ERR_DOCUMENT_TOO_LARGE,
    ERR_FORBIDDEN,
    ERR_INTERNAL,
    ERR_INVALID_INPUT,
    ERR_LLM_INVALID_RESPONSE,
    ERR_LLM_QUOTA_EXCEEDED,
    ERR_LLM_TIMEOUT,
    ERR_NOT_FOUND,
    ERR_OCR_FAILED,
    ERR_RATE_LIMIT,
    ERR_SERVICE_UNAVAILABLE,
    ERR_UNAUTHORIZED,
    HTTP_BAD_REQUEST,
    HTTP_CONFLICT,
    HTTP_FORBIDDEN,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_NOT_FOUND,
    HTTP_TOO_MANY_REQUESTS,
    HTTP_UNAUTHORIZED,
    HTTP_UNPROCESSABLE_ENTITY,
)


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class BaseAppException(Exception):
    """
    Base exception for all application exceptions.
    
    All custom exceptions should inherit from this class.
    
    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional error context
        status_code: HTTP status code (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> None:
        """
        Initialize base exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional context
            status_code: HTTP status code
        """
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.status_code = status_code
        
        super().__init__(self.message)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for serialization.
        
        Returns:
            dict: Exception data
            
        Example:
            >>> exc = NotFoundException(message="Kaynak bulunamadı")
            >>> print(exc.to_dict())
            {
                'error': 'Kaynak bulunamadı',
                'code': 'NOT_FOUND',
                'details': {}
            }
        """
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details,
        }
    
    def __str__(self) -> str:
        """String representation."""
        if self.details:
            return f"{self.message} (code: {self.code}, details: {self.details})"
        return f"{self.message} (code: {self.code})"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r}, "
            f"status_code={self.status_code})"
        )


# =============================================================================
# HTTP EXCEPTIONS
# =============================================================================

class HTTPException(BaseAppException):
    """
    Base HTTP exception.
    
    All HTTP-related exceptions inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize HTTP exception."""
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=status_code,
        )


class BadRequestException(HTTPException):
    """
    400 Bad Request.
    
    İstek hatalı veya geçersiz format.
    
    Example:
        >>> raise BadRequestException(
        ...     message="Geçersiz email formatı",
        ...     details={"field": "email", "value": "invalid"}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Geçersiz istek",
        code: str = ERR_INVALID_INPUT,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_BAD_REQUEST,
        )


class UnauthorizedException(HTTPException):
    """
    401 Unauthorized.
    
    Kimlik doğrulama gerekli veya başarısız.
    
    Example:
        >>> raise UnauthorizedException(
        ...     message="Geçersiz kimlik bilgileri"
        ... )
    """
    
    def __init__(
        self,
        message: str = "Yetkisiz erişim",
        code: str = ERR_UNAUTHORIZED,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_UNAUTHORIZED,
        )


class ForbiddenException(HTTPException):
    """
    403 Forbidden.
    
    Kullanıcı kimliği doğrulanmış ancak yetkisi yok.
    
    Example:
        >>> raise ForbiddenException(
        ...     message="Bu işlem için yetkiniz yok",
        ...     details={"required_role": "admin"}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Erişim yasak",
        code: str = ERR_FORBIDDEN,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_FORBIDDEN,
        )


class NotFoundException(HTTPException):
    """
    404 Not Found.
    
    İstenen kaynak bulunamadı.
    
    Example:
        >>> raise NotFoundException(
        ...     message="Kullanıcı bulunamadı",
        ...     details={"user_id": "123"}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Kaynak bulunamadı",
        code: str = ERR_NOT_FOUND,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_NOT_FOUND,
        )


class ConflictException(HTTPException):
    """
    409 Conflict.
    
    İstek mevcut durumla çakışıyor (örn: email zaten kayıtlı).
    
    Example:
        >>> raise ConflictException(
        ...     message="Email adresi zaten kullanımda",
        ...     details={"email": "user@example.com"}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Kaynak çakışması",
        code: str = ERR_CONFLICT,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_CONFLICT,
        )


class UnprocessableEntityException(HTTPException):
    """
    422 Unprocessable Entity.
    
    İstek formatı doğru ancak semantik olarak geçersiz.
    
    Example:
        >>> raise UnprocessableEntityException(
        ...     message="Doğrulama başarısız",
        ...     details={"errors": [{"field": "age", "message": "18 yaşından büyük olmalı"}]}
        ... )
    """
    
    def __init__(
        self,
        message: str = "İşlenemeyen varlık",
        code: str = ERR_INVALID_INPUT,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_UNPROCESSABLE_ENTITY,
        )


class TooManyRequestsException(HTTPException):
    """
    429 Too Many Requests.
    
    İstek limiti aşıldı.
    
    Example:
        >>> raise TooManyRequestsException(
        ...     message="İstek limiti aşıldı",
        ...     details={"limit": 100, "window": "1 saat", "retry_after": 3600}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Çok fazla istek",
        code: str = ERR_RATE_LIMIT,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_TOO_MANY_REQUESTS,
        )


class InternalServerException(HTTPException):
    """
    500 Internal Server Error.
    
    Beklenmeyen sunucu hatası.
    
    Example:
        >>> raise InternalServerException(
        ...     message="Veritabanı bağlantısı başarısız",
        ...     details={"error": str(e)}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Sunucu hatası",
        code: str = ERR_INTERNAL,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_INTERNAL_SERVER_ERROR,
        )


class ServiceUnavailableException(HTTPException):
    """
    503 Service Unavailable.
    
    Servis geçici olarak kullanılamıyor.
    
    Example:
        >>> raise ServiceUnavailableException(
        ...     message="Veritabanı bakımda",
        ...     details={"retry_after": 300}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Servis kullanılamıyor",
        code: str = ERR_SERVICE_UNAVAILABLE,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=503,
        )


# =============================================================================
# DATABASE EXCEPTIONS
# =============================================================================

class DatabaseException(BaseAppException):
    """
    Base database exception.
    
    All database-related errors inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        code: str = "DB_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_INTERNAL_SERVER_ERROR,
        )


class DatabaseConnectionException(DatabaseException):
    """Veritabanı bağlantısı başarısız."""
    
    def __init__(
        self,
        message: str = "Veritabanı bağlantısı başarısız",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="DB_CONNECTION_ERROR",
            details=details,
        )


class DatabaseQueryException(DatabaseException):
    """Veritabanı sorgusu başarısız."""
    
    def __init__(
        self,
        message: str = "Veritabanı sorgusu başarısız",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="DB_QUERY_ERROR",
            details=details,
        )


class DatabaseIntegrityException(DatabaseException):
    """Veritabanı bütünlük kısıtlaması ihlali."""
    
    def __init__(
        self,
        message: str = "Veritabanı bütünlük kısıtlaması ihlal edildi",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="DB_INTEGRITY_ERROR",
            details=details,
        )


# =============================================================================
# CACHE EXCEPTIONS
# =============================================================================

class CacheException(BaseAppException):
    """
    Base cache exception.
    
    All cache-related errors inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        code: str = "CACHE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
        )


class CacheConnectionException(CacheException):
    """Önbellek bağlantısı başarısız."""
    
    def __init__(
        self,
        message: str = "Önbellek bağlantısı başarısız",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="CACHE_CONNECTION_ERROR",
            details=details,
        )


class CacheOperationException(CacheException):
    """Önbellek işlemi başarısız."""
    
    def __init__(
        self,
        message: str = "Önbellek işlemi başarısız",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="CACHE_OPERATION_ERROR",
            details=details,
        )


# =============================================================================
# SECURITY EXCEPTIONS
# =============================================================================

class SecurityException(BaseAppException):
    """
    Base security exception.
    
    All security-related errors inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        code: str = "SECURITY_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_FORBIDDEN,
        )


class InvalidTokenException(SecurityException):
    """Token geçersiz veya süresi dolmuş."""
    
    def __init__(
        self,
        message: str = "Geçersiz veya süresi dolmuş token",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INVALID_TOKEN",
            details=details,
        )


class TokenExpiredException(SecurityException):
    """Token süresi dolmuş."""
    
    def __init__(
        self,
        message: str = "Token süresi dolmuş",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="TOKEN_EXPIRED",
            details=details,
        )


class InsufficientPermissionsException(SecurityException):
    """Kullanıcı gerekli yetkilere sahip değil."""
    
    def __init__(
        self,
        message: str = "Yeterli yetki yok",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INSUFFICIENT_PERMISSIONS",
            details=details,
        )


class InvalidCredentialsException(SecurityException):
    """Geçersiz kimlik bilgileri."""
    
    def __init__(
        self,
        message: str = "Geçersiz kullanıcı adı veya şifre",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INVALID_CREDENTIALS",
            details=details,
        )


class AccountLockedException(SecurityException):
    """Hesap kilitli."""
    
    def __init__(
        self,
        message: str = "Hesabınız kilitlenmiştir",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="ACCOUNT_LOCKED",
            details=details,
        )


# =============================================================================
# BUSINESS LOGIC EXCEPTIONS
# =============================================================================

class BusinessLogicException(BaseAppException):
    """
    Base business logic exception.
    
    Used for domain-specific business rule violations.
    """
    
    def __init__(
        self,
        message: str,
        code: str = "BUSINESS_LOGIC_ERROR",
        details: dict[str, Any] | None = None,
        status_code: int = HTTP_UNPROCESSABLE_ENTITY,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=status_code,
        )


class ValidationException(BusinessLogicException):
    """Veri doğrulama başarısız."""
    
    def __init__(
        self,
        message: str = "Doğrulama başarısız",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=details,
            status_code=HTTP_UNPROCESSABLE_ENTITY,
        )


class InvalidStateException(BusinessLogicException):
    """Geçersiz durum."""
    
    def __init__(
        self,
        message: str = "İşlem bu durumda gerçekleştirilemez",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INVALID_STATE",
            details=details,
        )


class QuotaExceededException(BusinessLogicException):
    """Kota aşıldı."""
    
    def __init__(
        self,
        message: str = "Kota limiti aşıldı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="QUOTA_EXCEEDED",
            details=details,
        )


# =============================================================================
# DOCUMENT PROCESSING EXCEPTIONS
# =============================================================================

class DocumentException(BaseAppException):
    """
    Base document processing exception.
    
    All document-related errors inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_UNPROCESSABLE_ENTITY,
        )


class DocumentTooLargeException(DocumentException):
    """Doküman boyut limitini aşıyor."""
    
    def __init__(
        self,
        message: str = "Doküman çok büyük",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ERR_DOCUMENT_TOO_LARGE,
            details=details,
        )


class DocumentInvalidFormatException(DocumentException):
    """Doküman formatı desteklenmiyor."""
    
    def __init__(
        self,
        message: str = "Geçersiz doküman formatı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ERR_DOCUMENT_INVALID_FORMAT,
            details=details,
        )


class DocumentParseException(DocumentException):
    """Doküman ayrıştırma başarısız."""
    
    def __init__(
        self,
        message: str = "Doküman ayrıştırılamadı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ERR_DOCUMENT_PARSE_FAILED,
            details=details,
        )


class OCRException(DocumentException):
    """OCR işlemi başarısız."""
    
    def __init__(
        self,
        message: str = "OCR işlemi başarısız",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ERR_OCR_FAILED,
            details=details,
        )


class DocumentCorruptedException(DocumentException):
    """Doküman bozuk veya okunamıyor."""
    
    def __init__(
        self,
        message: str = "Doküman bozuk veya okunamıyor",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="DOCUMENT_CORRUPTED",
            details=details,
        )


# =============================================================================
# LLM EXCEPTIONS
# =============================================================================

class LLMException(BaseAppException):
    """
    Base LLM exception.
    
    All LLM-related errors inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_INTERNAL_SERVER_ERROR,
        )


class LLMTimeoutException(LLMException):
    """LLM isteği zaman aşımına uğradı."""
    
    def __init__(
        self,
        message: str = "LLM isteği zaman aşımına uğradı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ERR_LLM_TIMEOUT,
            details=details,
        )


class LLMQuotaExceededException(LLMException):
    """LLM kotası aşıldı."""
    
    def __init__(
        self,
        message: str = "LLM kotası aşıldı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ERR_LLM_QUOTA_EXCEEDED,
            details=details,
        )


class LLMInvalidResponseException(LLMException):
    """LLM geçersiz yanıt döndü."""
    
    def __init__(
        self,
        message: str = "Geçersiz LLM yanıtı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ERR_LLM_INVALID_RESPONSE,
            details=details,
        )


class LLMConnectionException(LLMException):
    """LLM bağlantısı başarısız."""
    
    def __init__(
        self,
        message: str = "LLM servisine bağlanılamadı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="LLM_CONNECTION_ERROR",
            details=details,
        )


class LLMRateLimitException(LLMException):
    """LLM rate limit aşıldı."""
    
    def __init__(
        self,
        message: str = "LLM istek limiti aşıldı",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="LLM_RATE_LIMIT",
            details=details,
        )


# =============================================================================
# TURKISH LEGAL EXCEPTIONS
# =============================================================================

class TurkishLegalException(BaseAppException):
    """
    Base Turkish legal exception.
    
    For Turkish-specific legal domain errors.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=HTTP_UNPROCESSABLE_ENTITY,
        )


class InvalidTCNoException(TurkishLegalException):
    """Geçersiz TC Kimlik No."""
    
    def __init__(
        self,
        message: str = "Geçersiz TC Kimlik No",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INVALID_TC_NO",
            details=details,
        )


class InvalidIBANException(TurkishLegalException):
    """Geçersiz IBAN."""
    
    def __init__(
        self,
        message: str = "Geçersiz IBAN",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INVALID_IBAN",
            details=details,
        )


class InvalidVKNException(TurkishLegalException):
    """Geçersiz Vergi Kimlik Numarası."""
    
    def __init__(
        self,
        message: str = "Geçersiz Vergi Kimlik Numarası",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INVALID_VKN",
            details=details,
        )


class KVKKComplianceException(TurkishLegalException):
    """KVKK uyumluluk ihlali."""
    
    def __init__(
        self,
        message: str = "KVKK uyumluluk ihlali",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="KVKK_VIOLATION",
            details=details,
        )


class PersonalDataAccessException(TurkishLegalException):
    """Kişisel veri erişim ihlali."""
    
    def __init__(
        self,
        message: str = "Bu kişisel veriye erişim yetkiniz yok",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="PERSONAL_DATA_ACCESS_DENIED",
            details=details,
        )


class ConsentRequiredException(TurkishLegalException):
    """İzin/onay gerekli."""
    
    def __init__(
        self,
        message: str = "Bu işlem için kullanıcı onayı gereklidir",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="CONSENT_REQUIRED",
            details=details,
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base
    "BaseAppException",
    # HTTP
    "HTTPException",
    "BadRequestException",
    "UnauthorizedException",
    "ForbiddenException",
    "NotFoundException",
    "ConflictException",
    "UnprocessableEntityException",
    "TooManyRequestsException",
    "InternalServerException",
    "ServiceUnavailableException",
    # Database
    "DatabaseException",
    "DatabaseConnectionException",
    "DatabaseQueryException",
    "DatabaseIntegrityException",
    # Cache
    "CacheException",
    "CacheConnectionException",
    "CacheOperationException",
    # Security
    "SecurityException",
    "InvalidTokenException",
    "TokenExpiredException",
    "InsufficientPermissionsException",
    "InvalidCredentialsException",
    "AccountLockedException",
    # Business Logic
    "BusinessLogicException",
    "ValidationException",
    "InvalidStateException",
    "QuotaExceededException",
    # Document Processing
    "DocumentException",
    "DocumentTooLargeException",
    "DocumentInvalidFormatException",
    "DocumentParseException",
    "OCRException",
    "DocumentCorruptedException",
    # LLM
    "LLMException",
    "LLMTimeoutException",
    "LLMQuotaExceededException",
    "LLMInvalidResponseException",
    "LLMConnectionException",
    "LLMRateLimitException",
    # Turkish Legal
    "TurkishLegalException",
    "InvalidTCNoException",
    "InvalidIBANException",
    "InvalidVKNException",
    "KVKKComplianceException",
    "PersonalDataAccessException",
    "ConsentRequiredException",
]
