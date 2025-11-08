"""Turkish Legal Citation Patterns - Harvey/Legora CTO-Level
Comprehensive regex patterns for Turkish legal citations.
"""
import re

# Law citations
LAW_FULL = r'(\d{3,5})\s+[Ss]ayılı\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:Kanunu|Kanun))'
LAW_SHORT = r'(\d{3,5})\s+[Ss]ayılı'
LAW_WITH_ARTICLE = r'(\d{3,5})\s+[Ss]ayılı.*?[Mm]adde\s+(\d+)'

# Article citations
MADDE = r'[Mm]adde\s+(\d+)'
FIKRA = r'[Ff]ıkra\s+(\d+)'
BENT = r'[Bb]ent\s+([a-z])\)'

# Court citations
YARGITAY_FULL = r'Yargıtay\s+(\d+)\.\s*(Hukuk|Ceza)\s+Dairesi?\s+(?:E:\s*)?(\d{4}/\d+)?\s*(?:K:\s*)?(\d{4}/\d+)?'
DANISTAY_FULL = r'Danıştay\s+(\d+)\.\s*Dairesi?\s+(?:E:\s*)?(\d{4}/\d+)?\s*(?:K:\s*)?(\d{4}/\d+)?'
AYM_FULL = r'(?:T\.?C\.?)?\s*Anayasa\s+Mahkemesi\s+(?:E:\s*)?(\d{4}/\d+)?\s*(?:K:\s*)?(\d{4}/\d+)?'

# Resmi Gazete
RG_FULL = r'(?:RG|R\.G\.|Resmi\s+Gazete)\s+(?:Sayı|No)?:?\s*(\d{5})?\s+(?:Tarih)?:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})?'

# Compiled patterns
PATTERNS = {
    'law_full': re.compile(LAW_FULL, re.IGNORECASE),
    'law_short': re.compile(LAW_SHORT, re.IGNORECASE),
    'law_article': re.compile(LAW_WITH_ARTICLE, re.IGNORECASE),
    'madde': re.compile(MADDE, re.IGNORECASE),
    'fikra': re.compile(FIKRA, re.IGNORECASE),
    'bent': re.compile(BENT, re.IGNORECASE),
    'yargitay': re.compile(YARGITAY_FULL, re.IGNORECASE),
    'danistay': re.compile(DANISTAY_FULL, re.IGNORECASE),
    'aym': re.compile(AYM_FULL, re.IGNORECASE),
    'rg': re.compile(RG_FULL, re.IGNORECASE),
}

__all__ = ['PATTERNS', 'LAW_FULL', 'MADDE', 'YARGITAY_FULL', 'DANISTAY_FULL', 'RG_FULL']
