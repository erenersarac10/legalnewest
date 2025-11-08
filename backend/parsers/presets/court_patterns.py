"""Turkish Court Name Patterns - Harvey/Legora CTO-Level"""
import re

COURTS = {
    'yargitay': {
        'pattern': r'Yargıtay\s+(\d+)\.\s*(Hukuk|Ceza)\s+Dairesi?',
        'type': 'cassation',
        'level': 'supreme'
    },
    'danistay': {
        'pattern': r'Danıştay\s+(\d+)\.\s*Dairesi?',
        'type': 'administrative',
        'level': 'supreme'
    },
    'aym': {
        'pattern': r'(?:T\.?C\.?)?\s*Anayasa\s+Mahkemesi',
        'type': 'constitutional',
        'level': 'supreme'
    },
    'idare_mahkemesi': {
        'pattern': r'(\w+)\s+(?:\d+\.?\s*)?İdare\s+Mahkemesi',
        'type': 'administrative',
        'level': 'first_instance'
    },
    'asliye': {
        'pattern': r'(\w+)\s+(?:\d+\.?\s*)?(Hukuk|Ceza)\s+Mahkemesi',
        'type': 'civil_criminal',
        'level': 'first_instance'
    },
}

COMPILED = {k: re.compile(v['pattern'], re.IGNORECASE) for k, v in COURTS.items()}

__all__ = ['COURTS', 'COMPILED']
