"""Common Regex Patterns - Harvey/Legora CTO-Level"""
import re

# Numbers
TURKISH_NUMBER = r'(?:bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on|yirmi|otuz|kırk|elli|altmış|yetmiş|seksen|doksan|yüz|bin)'
ROMAN_NUMERAL = r'(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)'

# Dates
TURKISH_DATE = r'\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}'
ISO_DATE = r'\d{4}-\d{2}-\d{2}'
SLASH_DATE = r'\d{1,2}/\d{1,2}/\d{2,4}'
DOT_DATE = r'\d{1,2}\.\d{1,2}\.\d{2,4}'

# Legal identifiers
TC_KIMLIK_NO = r'\d{11}'
VERGI_NO = r'\d{10}'
IBAN = r'TR\d{24}'

# URLs
URL = r'https?://[^\s]+'
EMAIL = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

COMPILED = {
    'turkish_date': re.compile(TURKISH_DATE, re.IGNORECASE),
    'iso_date': re.compile(ISO_DATE),
    'url': re.compile(URL),
    'email': re.compile(EMAIL),
}

__all__ = ['COMPILED', 'TURKISH_DATE', 'ISO_DATE', 'URL']
