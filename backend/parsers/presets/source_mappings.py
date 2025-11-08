"""Turkish Legal Source URL Mappings - Harvey/Legora CTO-Level"""

SOURCE_URLS = {
    'resmi_gazete': 'https://www.resmigazete.gov.tr',
    'mevzuat': 'https://www.mevzuat.gov.tr',
    'yargitay': 'https://www.yargitay.gov.tr',
    'danistay': 'https://www.danistay.gov.tr',
    'aym': 'https://www.anayasa.gov.tr',
    'kvkk': 'https://www.kvkk.gov.tr',
    'spk': 'https://www.spk.gov.tr',
    'bddk': 'https://www.bddk.org.tr',
    'rekabet': 'https://www.rekabet.gov.tr',
    'gib': 'https://www.gib.gov.tr',
    'sgk': 'https://www.sgk.gov.tr',
    'tbmm': 'https://www.tbmm.gov.tr',
}

API_ENDPOINTS = {
    'mevzuat': 'https://www.mevzuat.gov.tr/api',
    'echr': 'https://hudoc.echr.coe.int/api',
}

RATE_LIMITS = {
    'resmi_gazete': 30,  # requests per minute
    'yargitay': 30,
    'mevzuat': 60,
    'echr': 60,
}

__all__ = ['SOURCE_URLS', 'API_ENDPOINTS', 'RATE_LIMITS']
