"""Turkish Legal Clause Structure Patterns - Harvey/Legora CTO-Level"""
import re

# Clause hierarchy patterns
MADDE_HEADER = r'^(?:MADDE|Madde)\s+(\d+)\s*[-–—]?\s*(.*?)$'
FIKRA_MARKER = r'^\((\d+)\)'
BENT_MARKER = r'^([a-z])\)'
PARAGRAF_MARKER = r'^(?:Paragraf|§)\s+(\d+)'

# Section patterns
BOLUM = r'^(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ)?\s*BÖLÜM'
KISIM = r'^(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ)?\s*KISIM'

# Special articles
GECICI_MADDE = r'^(?:GEÇİCİ|Geçici)\s+(?:MADDE|Madde)\s+(\d+)'
EK_MADDE = r'^(?:EK|Ek)\s+(?:MADDE|Madde)\s+(\d+)'

PATTERNS = {
    'madde_header': re.compile(MADDE_HEADER, re.MULTILINE | re.IGNORECASE),
    'fikra': re.compile(FIKRA_MARKER, re.MULTILINE),
    'bent': re.compile(BENT_MARKER, re.MULTILINE),
    'bolum': re.compile(BOLUM, re.MULTILINE | re.IGNORECASE),
    'kisim': re.compile(KISIM, re.MULTILINE | re.IGNORECASE),
    'gecici_madde': re.compile(GECICI_MADDE, re.MULTILINE | re.IGNORECASE),
    'ek_madde': re.compile(EK_MADDE, re.MULTILINE | re.IGNORECASE),
}

__all__ = ['PATTERNS', 'MADDE_HEADER', 'FIKRA_MARKER', 'BOLUM']
