"""Turkish Legal Keywords Lexicon - Harvey/Legora CTO-Level"""

# Legal document types
DOCUMENT_TYPES = {
    'kanun', 'yönetmelik', 'tüzük', 'tebliğ', 'genelge', 'cbk', 'khk',
    'karar', 'hüküm', 'yargı', 'emsal', 'içtihat'
}

# Legal actions
LEGAL_ACTIONS = {
    'dava', 'şikayet', 'başvuru', 'itiraz', 'temyiz', 'istinaf', 'karar düzeltme',
    'ön inceleme', 'duruşma', 'tahkikat', 'keşif', 'bilirkişi'
}

# Court terms
COURT_TERMS = {
    'mahkeme', 'daire', 'heyet', 'kurul', 'hakim', 'savcı',
    'zabıt katibi', 'mübaşir', 'bilirkişi', 'tanık', 'sanık', 'mağdur'
}

# Legal principles (Turkish)
PRINCIPLES = {
    'şüpheden sanık yararlanır', 'işçi lehine yorum', 'kıyas yasağı',
    'dar yorum', 'kusursuz sorumluluk', 'hakkaniyet', 'nesafet',
    'iyi niyet', 'dürüstlük kuralı', 'kamu düzeni'
}

# Jurisdiction types
JURISDICTIONS = {
    'ceza', 'hukuk', 'idare', 'vergi', 'ticaret', 'iş', 'icra iflas',
    'anayasa', 'askeri', 'disiplin'
}

# Legal subjects
SUBJECTS = {
    'miras', 'aile', 'sözleşme', 'tazminat', 'mülkiyet', 'zilyetlik',
    'borç', 'alacak', 'kira', 'satış', 'taşınmaz', 'taşınır'
}

ALL_KEYWORDS = (
    DOCUMENT_TYPES | LEGAL_ACTIONS | COURT_TERMS |
    PRINCIPLES | JURISDICTIONS | SUBJECTS
)

__all__ = ['DOCUMENT_TYPES', 'LEGAL_ACTIONS', 'COURT_TERMS', 'PRINCIPLES', 'ALL_KEYWORDS']
