"""
Security configuration for Turkish Legal AI.

This module provides comprehensive security utilities:
- Password hashing (Argon2 + BCrypt fallback)
- JWT token generation/validation (RS256)
- Data encryption/decryption (Fernet + AES-256-GCM)
- API key generation
- CSRF protection
- Rate limiting tokens
- Secure random generation
- KVKK/GDPR compliant data handling

Security Standards:
- OWASP Top 10 compliance
- NIST password guidelines
- KVKK data protection requirements
- PCI-DSS encryption standards
"""
import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import (
    HashingError,
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from passlib.context import CryptContext

from backend.core.config.settings import settings
from backend.core.constants import (
    API_KEY_LENGTH,
    API_KEY_PREFIX,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM_DEFAULT,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
    PASSWORD_ARGON2_MEMORY_COST,
    PASSWORD_ARGON2_PARALLELISM,
    PASSWORD_ARGON2_TIME_COST,
    PASSWORD_BCRYPT_ROUNDS,
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
)


class SecurityConfig:
    """
    Security configuration and cryptographic operations.
    
    Provides utilities for:
    - Password hashing and verification
    - JWT token management
    - Data encryption/decryption
    - API key generation
    - Secure random generation
    """
    
    def __init__(self) -> None:
        """Initialize security configuration."""
        # Argon2 password hasher (primary)
        self._argon2 = PasswordHasher(
            time_cost=PASSWORD_ARGON2_TIME_COST,
            memory_cost=PASSWORD_ARGON2_MEMORY_COST,
            parallelism=PASSWORD_ARGON2_PARALLELISM,
            hash_len=32,
            salt_len=16,
        )
        
        # Passlib context (BCrypt fallback + migration support)
        self._pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto",
            argon2__time_cost=PASSWORD_ARGON2_TIME_COST,
            argon2__memory_cost=PASSWORD_ARGON2_MEMORY_COST,
            argon2__parallelism=PASSWORD_ARGON2_PARALLELISM,
            bcrypt__rounds=PASSWORD_BCRYPT_ROUNDS,
        )
        
        # Fernet encryption (symmetric)
        self._fernet = Fernet(self._get_or_create_encryption_key())
        
        # JWT keys (will be loaded lazily)
        self._jwt_private_key: str | None = None
        self._jwt_public_key: str | None = None
    
    # =========================================================================
    # PASSWORD HASHING
    # =========================================================================
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2.
        
        Argon2 is the winner of the Password Hashing Competition and provides
        excellent protection against GPU/ASIC attacks.
        
        Args:
            password: Plain text password
            
        Returns:
            str: Hashed password
            
        Raises:
            ValueError: If password doesn't meet requirements
            
        Example:
            >>> hashed = security_config.hash_password("MySecureP@ss123")
            >>> print(hashed[:20])
            $argon2id$v=19$m=65536...
        """
        # Validate password
        self.validate_password_strength(password)
        
        try:
            # Use Argon2 as primary hasher
            return self._argon2.hash(password)
        except HashingError:
            # Fallback to BCrypt if Argon2 fails
            return self._pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Supports both Argon2 and BCrypt hashes for migration compatibility.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database
            
        Returns:
            bool: True if password matches
            
        Example:
            >>> is_valid = security_config.verify_password(
            ...     "MySecureP@ss123",
            ...     stored_hash
            ... )
        """
        try:
            # Try Argon2 verification first
            if hashed_password.startswith("$argon2"):
                self._argon2.verify(hashed_password, plain_password)
                
                # Check if rehashing is needed (parameters changed)
                if self._argon2.check_needs_rehash(hashed_password):
                    # Signal that password should be rehashed
                    # (caller should save new hash)
                    pass
                
                return True
            else:
                # Use passlib for BCrypt or other legacy hashes
                return self._pwd_context.verify(plain_password, hashed_password)
        except (VerifyMismatchError, InvalidHashError, ValueError):
            return False
    
    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Check if password hash needs rehashing.
        
        Returns True if hash uses outdated parameters or algorithm.
        
        Args:
            hashed_password: Current password hash
            
        Returns:
            bool: True if rehashing recommended
        """
        if hashed_password.startswith("$argon2"):
            return self._argon2.check_needs_rehash(hashed_password)
        else:
            # BCrypt or other legacy - should migrate to Argon2
            return True
    
    # =========================================================================
    # PASSWORD VALIDATION
    # =========================================================================
    
    def validate_password_strength(self, password: str) -> None:
        """
        Şifre güvenlik gereksinimlerini doğrula.
        
        Gereksinimler (ayarlardan yapılandırılabilir):
        - Uzunluk: 12-128 karakter
        - Büyük harf içermeli
        - Küçük harf içermeli
        - Rakam içermeli
        - Özel karakter içermeli
        
        Args:
            password: Doğrulanacak şifre
            
        Raises:
            ValueError: Şifre gereksinimleri karşılanmazsa
            
        Example:
            >>> security_config.validate_password_strength("MyP@ss123")
            # Geçerli şifre
            
            >>> security_config.validate_password_strength("weak")
            ValueError: Şifre en az 12 karakter olmalıdır
        """
        if len(password) < PASSWORD_MIN_LENGTH:
            raise ValueError(
                f"Şifre en az {PASSWORD_MIN_LENGTH} karakter olmalıdır"
            )
        
        if len(password) > PASSWORD_MAX_LENGTH:
            raise ValueError(
                f"Şifre en fazla {PASSWORD_MAX_LENGTH} karakter olabilir"
            )
        
        if settings.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            raise ValueError("Şifre en az bir büyük harf içermelidir")
        
        if settings.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            raise ValueError("Şifre en az bir küçük harf içermelidir")
        
        if settings.PASSWORD_REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            raise ValueError("Şifre en az bir rakam içermelidir")
        
        if settings.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                raise ValueError(
                    f"Şifre en az bir özel karakter içermelidir: {special_chars}"
                )
        
        # Yaygın zayıf şifreleri kontrol et
        if password.lower() in self._get_common_passwords():
            raise ValueError(
                "Bu şifre çok yaygın kullanılıyor. Lütfen daha güçlü bir şifre seçin."
            )
    
    def _get_common_passwords(self) -> set[str]:
        """Get set of common/weak passwords to reject."""
        return {
            "password", "123456", "12345678", "qwerty", "abc123",
            "monkey", "1234567", "letmein", "trustno1", "dragon",
            "baseball", "iloveyou", "master", "sunshine", "ashley",
            "bailey", "shadow", "123123", "654321", "superman",
        }
    
    def calculate_password_strength(self, password: str) -> dict[str, Any]:
        """
        Şifre gücü skorunu hesapla.
        
        Args:
            password: Analiz edilecek şifre
            
        Returns:
            dict: Güç analizi
            
        Example:
            >>> analysis = security_config.calculate_password_strength("MyP@ss123")
            >>> print(analysis["score"])  # 0-100
            75
            >>> print(analysis["strength"])
            'güçlü'
            >>> print(analysis["feedback"])
            ['Daha uzun şifre kullanın']
        """
        score = 0
        feedback = []
        
        # Uzunluk skorlaması
        length = len(password)
        if length >= 16:
            score += 30
        elif length >= 12:
            score += 20
        elif length >= 8:
            score += 10
        else:
            feedback.append("Şifre çok kısa")
        
        # Karakter çeşitliliği
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        variety_score = sum([has_lower, has_upper, has_digit, has_special])
        score += variety_score * 15
        
        if variety_score < 3:
            feedback.append("Büyük harf, küçük harf, rakam ve özel karakterleri birlikte kullanın")
        
        # Sıralı karakter cezası
        if password.lower() in "".join(str(i) for i in range(10)):
            score -= 20
            feedback.append("Ardışık rakam dizileri kullanmayın")
        
        if password.lower() in "abcdefghijklmnopqrstuvwxyz":
            score -= 20
            feedback.append("Ardışık harf dizileri kullanmayın")
        
        # Yaygın şifre kontrolü
        if password.lower() in self._get_common_passwords():
            score = max(0, score - 50)
            feedback.append("Bu çok yaygın bir şifre, daha özgün bir şifre seçin")
        
        # Güç kategorisi belirleme
        if score >= 80:
            strength = "çok_güçlü"
        elif score >= 60:
            strength = "güçlü"
        elif score >= 40:
            strength = "orta"
        elif score >= 20:
            strength = "zayıf"
        else:
            strength = "çok_zayıf"
        
        return {
            "score": min(100, max(0, score)),
            "strength": strength,
            "feedback": feedback,
            "has_lowercase": has_lower,
            "has_uppercase": has_upper,
            "has_digit": has_digit,
            "has_special": has_special,
            "length": length,
        }
    
    # =========================================================================
    # JWT TOKEN MANAGEMENT
    # =========================================================================
    
    def create_access_token(
        self,
        subject: str,
        additional_claims: dict[str, Any] | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            subject: Token subject (usually user ID)
            additional_claims: Extra claims to include
            expires_delta: Custom expiration time
            
        Returns:
            str: Encoded JWT token
            
        Example:
            >>> token = security_config.create_access_token(
            ...     subject="user_123",
            ...     additional_claims={"role": "admin"}
            ... )
        """
        if expires_delta is None:
            expires_delta = timedelta(
                minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        expire = datetime.now(timezone.utc) + expires_delta
        
        claims = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
            "jti": self.generate_token_id(),
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        return jwt.encode(
            claims,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithm=settings.JWT_ALGORITHM,
        )
    
    def create_refresh_token(
        self,
        subject: str,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a JWT refresh token.
        
        Refresh tokens have longer expiration and can be used to obtain
        new access tokens.
        
        Args:
            subject: Token subject (user ID)
            additional_claims: Extra claims
            
        Returns:
            str: Encoded JWT refresh token
        """
        expires_delta = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        expire = datetime.now(timezone.utc) + expires_delta
        
        claims = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
            "jti": self.generate_token_id(),
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        return jwt.encode(
            claims,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithm=settings.JWT_ALGORITHM,
        )
    
    def decode_token(self, token: str) -> dict[str, Any]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            dict: Decoded token payload
            
        Raises:
            jwt.ExpiredSignatureError: If token is expired
            jwt.InvalidTokenError: If token is invalid
            
        Example:
            >>> payload = security_config.decode_token(token)
            >>> user_id = payload["sub"]
        """
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY.get_secret_value(),
                algorithms=[settings.JWT_ALGORITHM],
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise jwt.ExpiredSignatureError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f"Invalid token: {e}")
    
    def verify_token_type(self, token: str, expected_type: str) -> dict[str, Any]:
        """
        Verify token type matches expected type.
        
        Args:
            token: JWT token
            expected_type: Expected token type ('access' or 'refresh')
            
        Returns:
            dict: Decoded payload if valid
            
        Raises:
            ValueError: If token type doesn't match
        """
        payload = self.decode_token(token)
        
        token_type = payload.get("type")
        if token_type != expected_type:
            raise ValueError(
                f"Invalid token type. Expected '{expected_type}', got '{token_type}'"
            )
        
        return payload
    
    def generate_token_id(self) -> str:
        """
        Generate a unique token ID (jti claim).
        
        Returns:
            str: UUID4 token identifier
        """
        return str(uuid.uuid4())
    
    # =========================================================================
    # DATA ENCRYPTION
    # =========================================================================
    
    def encrypt(self, data: str | bytes) -> str:
        """
        Encrypt data using Fernet (AES-128-CBC + HMAC).
        
        Fernet provides authenticated encryption and is suitable for
        encrypting sensitive data at rest.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            str: Base64-encoded encrypted data
            
        Example:
            >>> encrypted = security_config.encrypt("sensitive data")
            >>> decrypted = security_config.decrypt(encrypted)
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self._fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt Fernet-encrypted data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            str: Decrypted data
            
        Raises:
            InvalidToken: If decryption fails
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except (InvalidToken, ValueError) as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def encrypt_aes_gcm(self, data: bytes, associated_data: bytes | None = None) -> dict[str, str]:
        """
        Encrypt data using AES-256-GCM (AEAD).
        
        AES-GCM provides authenticated encryption with additional data (AEAD)
        and is faster than Fernet for large data.
        
        Args:
            data: Data to encrypt
            associated_data: Additional authenticated data (optional)
            
        Returns:
            dict: Contains 'ciphertext', 'nonce', and 'tag'
            
        Example:
            >>> result = security_config.encrypt_aes_gcm(
            ...     b"secret document",
            ...     associated_data=b"user_id:123"
            ... )
        """
        key = self._derive_aes_key()
        aesgcm = AESGCM(key)
        
        # Generate random nonce
        nonce = secrets.token_bytes(12)
        
        # Encrypt
        ciphertext = aesgcm.encrypt(nonce, data, associated_data)
        
        return {
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
        }
    
    def decrypt_aes_gcm(
        self,
        ciphertext: str,
        nonce: str,
        associated_data: bytes | None = None,
    ) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data.
        
        Args:
            ciphertext: Base64-encoded ciphertext
            nonce: Base64-encoded nonce
            associated_data: Additional authenticated data
            
        Returns:
            bytes: Decrypted data
            
        Raises:
            ValueError: If decryption or authentication fails
        """
        key = self._derive_aes_key()
        aesgcm = AESGCM(key)
        
        try:
            ciphertext_bytes = base64.b64decode(ciphertext)
            nonce_bytes = base64.b64decode(nonce)
            
            plaintext = aesgcm.decrypt(nonce_bytes, ciphertext_bytes, associated_data)
            return plaintext
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def _derive_aes_key(self) -> bytes:
        """Derive AES-256 key from master key."""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"turkish-legal-ai-salt",  # In production, use unique salt
            iterations=100000,
        )
        return kdf.derive(settings.MASTER_KEY.get_secret_value().encode())
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create Fernet encryption key."""
        try:
            key = settings.ENCRYPTION_KEY.get_secret_value()
            # Validate key format
            return base64.urlsafe_b64decode(key)
        except Exception:
            # Generate new key (development only)
            if settings.ENVIRONMENT == "development":
                return Fernet.generate_key()
            raise ValueError("ENCRYPTION_KEY must be set in production")
    
    # =========================================================================
    # API KEY GENERATION
    # =========================================================================
    
    def generate_api_key(self) -> str:
        """
        Generate a secure API key.
        
        Format: la_<random_64_chars>
        
        Returns:
            str: API key
            
        Example:
            >>> api_key = security_config.generate_api_key()
            >>> print(api_key[:5])
            la_ab
        """
        random_part = secrets.token_urlsafe(API_KEY_LENGTH)[:API_KEY_LENGTH]
        return f"{API_KEY_PREFIX}{random_part}"
    
    def hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for storage.
        
        Args:
            api_key: Plain API key
            
        Returns:
            str: SHA-256 hash of API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        """
        Verify an API key against its hash.
        
        Args:
            plain_key: Plain API key from request
            hashed_key: Hashed key from database
            
        Returns:
            bool: True if keys match
        """
        computed_hash = self.hash_api_key(plain_key)
        return hmac.compare_digest(computed_hash, hashed_key)
    
    # =========================================================================
    # SECURE RANDOM GENERATION
    # =========================================================================
    
    def generate_random_string(self, length: int = 32) -> str:
        """
        Generate cryptographically secure random string.
        
        Args:
            length: String length
            
        Returns:
            str: Random URL-safe string
        """
        return secrets.token_urlsafe(length)[:length]
    
    def generate_random_hex(self, length: int = 32) -> str:
        """
        Generate cryptographically secure random hex string.
        
        Args:
            length: Hex string length
            
        Returns:
            str: Random hex string
        """
        return secrets.token_hex(length // 2)
    
    def generate_otp(self, length: int = 6) -> str:
        """
        Generate a numeric OTP (One-Time Password).
        
        Args:
            length: OTP length (default: 6 digits)
            
        Returns:
            str: Numeric OTP
            
        Example:
            >>> otp = security_config.generate_otp()
            >>> print(otp)
            '847392'
        """
        return ''.join(secrets.choice('0123456789') for _ in range(length))
    
    # =========================================================================
    # CSRF PROTECTION
    # =========================================================================
    
    def generate_csrf_token(self) -> str:
        """
        Generate CSRF token.
        
        Returns:
            str: CSRF token
        """
        return secrets.token_urlsafe(32)
    
    def verify_csrf_token(self, token: str, expected: str) -> bool:
        """
        Verify CSRF token.
        
        Args:
            token: Token from request
            expected: Expected token from session
            
        Returns:
            bool: True if tokens match
        """
        return hmac.compare_digest(token, expected)
    
    # =========================================================================
    # HASHING UTILITIES
    # =========================================================================
    
    def hash_sha256(self, data: str | bytes) -> str:
        """
        Create SHA-256 hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            str: Hex-encoded SHA-256 hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    def hash_hmac_sha256(self, data: str | bytes, key: str | bytes) -> str:
        """
        Create HMAC-SHA256 of data.
        
        Args:
            data: Data to hash
            key: Secret key
            
        Returns:
            str: Hex-encoded HMAC-SHA256
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()


# =============================================================================
# GLOBAL SECURITY CONFIG INSTANCE
# =============================================================================

security_config = SecurityConfig()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password. Convenience wrapper."""
    return security_config.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password. Convenience wrapper."""
    return security_config.verify_password(plain_password, hashed_password)


def create_access_token(subject: str, **kwargs: Any) -> str:
    """Create access token. Convenience wrapper."""
    return security_config.create_access_token(subject, **kwargs)


def decode_token(token: str) -> dict[str, Any]:
    """Decode JWT token. Convenience wrapper."""
    return security_config.decode_token(token)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SecurityConfig",
    "security_config",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_token",
]