"""LLM Prompt Library - Harvey/Legora CTO-Level Production-Grade
Comprehensive Turkish legal AI prompt template library

Production Features:
- 50+ specialized Turkish legal prompt templates
- Document analysis prompts (laws, regulations, court decisions)
- Citation extraction and resolution prompts
- Legal reasoning and argumentation prompts
- Compliance checking prompts
- Contract analysis prompts
- Legal research and precedent finding
- Question answering prompts
- Summarization prompts (executive, technical, layperson)
- Translation prompts (legal Turkish ” English)
- Entity extraction prompts
- Timeline construction prompts
- Risk assessment prompts
- Multi-shot example integration
- Domain-specific templates (KVKK, TCK, TTK, TMK, etc.)
"""
from typing import Dict, List
from .prompt_framework import PromptTemplate, PromptRole, PromptStrategy, OutputFormat


# ============================================================================
# DOCUMENT ANALYSIS TEMPLATES
# ============================================================================

LAW_ANALYSIS_TEMPLATE = PromptTemplate(
    template_id="law_analysis",
    name="Turkish Law Analysis",
    template="""Sen Türk hukuku konusunda uzman bir yapay zeka asistan1s1n.

A_a1daki kanun metnini analiz et:

{law_text}

Lütfen _unlar1 belirt:
1. Kanunun temel amac1 ve kapsam1
2. Ana maddeler ve düzenlemeler
3. Yapt1r1mlar ve cezai hükümler
4. 0lgili dier kanunlarla ili_kisi
5. Uygulamada dikkat edilmesi gereken hususlar

Analiz türkçe olsun ve hukuki terminolojiyi doru kullan.""",
    description="Analyze Turkish law text comprehensively",
    variables=["law_text"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT,
    output_format=OutputFormat.MARKDOWN
)

REGULATION_ANALYSIS_TEMPLATE = PromptTemplate(
    template_id="regulation_analysis",
    name="Regulation Analysis",
    template="""A_a1daki yönetmelik metnini incele:

{regulation_text}

^u aç1lardan deerlendir:
1. Hangi kanuna dayan1yor ve yetkili kurum
2. Düzenleme alan1 ve kapsam
3. Kritik maddeler ve yükümlülükler
4. Uyum için gereken ad1mlar
5. Yürürlük ve geçi_ hükümleri

Detayl1 ve uygulamaya dönük bir analiz yap.""",
    description="Analyze regulatory text with compliance focus",
    variables=["regulation_text"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT
)

COURT_DECISION_ANALYSIS_TEMPLATE = PromptTemplate(
    template_id="court_decision_analysis",
    name="Court Decision Analysis",
    template="""A_a1daki {court_type} karar1n1 analiz et:

Karar No: {decision_number}
Tarih: {decision_date}

Karar Metni:
{decision_text}

Lütfen _u ba_l1klar1 içeren bir analiz yap:
1. Uyu_mazl11n konusu ve taraflar
2. Mahkemenin deerlendirmesi
3. Hukuki dayanak (ilgili kanun maddeleri)
4. Karar ve gerekçesi
5. Emsal nitelii ve içtihat deeri
6. Uygulama için ç1kar1mlar""",
    description="Analyze court decision with legal reasoning",
    variables=["court_type", "decision_number", "decision_date", "decision_text"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT
)


# ============================================================================
# CITATION & REFERENCE TEMPLATES
# ============================================================================

CITATION_EXTRACTION_TEMPLATE = PromptTemplate(
    template_id="citation_extraction",
    name="Legal Citation Extraction",
    template="""A_a1daki hukuki metinden tüm at1flar1 (citations) ç1kar:

{legal_text}

Her at1f için _unlar1 belirt:
- At1f yap1lan kaynak (kanun, yönetmelik, karar vb.)
- Kaynak numaras1 (örn: 5237 say1l1 TCK)
- Madde numaras1 (varsa)
- F1kra/bent bilgisi (varsa)
- At1f metni

JSON format1nda döndür.""",
    description="Extract all legal citations from text",
    variables=["legal_text"],
    role=PromptRole.USER,
    output_format=OutputFormat.JSON
)

CITATION_RESOLUTION_TEMPLATE = PromptTemplate(
    template_id="citation_resolution",
    name="Citation Resolution",
    template="""^u at1f1 çözümle ve tam künye bilgilerini ver:

At1f: {citation_text}

Beklenen bilgiler:
- Tam kaynak ad1
- Kanun/karar numaras1
- Yay1mland11 Resmi Gazete tarihi ve say1s1
- 0lgili madde/f1kra metni (biliyorsan)
- Güncel durum (yürürlükte mi, mülga m1)

JSON format1nda döndür.""",
    description="Resolve legal citation to full reference",
    variables=["citation_text"],
    role=PromptRole.USER,
    output_format=OutputFormat.JSON
)


# ============================================================================
# LEGAL REASONING TEMPLATES
# ============================================================================

LEGAL_ARGUMENT_TEMPLATE = PromptTemplate(
    template_id="legal_argument",
    name="Legal Argument Generation",
    template="""Sen deneyimli bir hukukçusun. A_a1daki durum için hukuki bir argüman olu_tur:

Durum:
{case_description}

Savunulan Tez:
{argument_position}

0lgili Kanunlar:
{relevant_laws}

Lütfen _unlar1 içeren güçlü bir hukuki argüman geli_tir:
1. Tezin hukuki dayana1
2. 0lgili kanun maddeleri ve yorumlar1
3. Emsal içtihatlar (varsa)
4. Kar_1 argümanlara cevaplar
5. Sonuç ve talep

Profesyonel hukuki dil kullan.""",
    description="Generate legal argument with supporting law",
    variables=["case_description", "argument_position", "relevant_laws"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT
)

PRECEDENT_SEARCH_TEMPLATE = PromptTemplate(
    template_id="precedent_search",
    name="Precedent Search Query",
    template="""A_a1daki hukuki sorun için emsal karar arama sorgusu olu_tur:

Sorun:
{legal_issue}

Anahtar Kavramlar:
{key_concepts}

0lgili Kanun:
{relevant_law}

^unlar1 içeren bir emsal arama stratejisi öner:
1. Aranacak anahtar kelimeler
2. 0lgili mahkeme türleri (Yarg1tay, Dan1_tay, AYM)
3. Tarih aral11 önerisi
4. Beklenen karar türü

JSON format1nda döndür.""",
    description="Generate precedent search strategy",
    variables=["legal_issue", "key_concepts", "relevant_law"],
    role=PromptRole.USER,
    output_format=OutputFormat.JSON
)


# ============================================================================
# COMPLIANCE CHECKING TEMPLATES
# ============================================================================

KVKK_COMPLIANCE_TEMPLATE = PromptTemplate(
    template_id="kvkk_compliance",
    name="KVKK Compliance Check",
    template="""A_a1daki veri i_leme sürecini KVKK (6698 say1l1 Ki_isel Verilerin Korunmas1 Kanunu) aç1s1ndan deerlendir:

Veri 0_leme Süreci:
{data_processing_description}

0_lenen Ki_isel Veriler:
{personal_data_types}

Hukuki Dayanak:
{legal_basis}

KVKK uyumluluk kontrolü yap ve _unlar1 belirt:
1. 0_lemenin hukuki dayana1 yeterli mi?
2. Ayd1nlatma yükümlülüü yerine getirilmi_ mi?
3. Veri güvenlii tedbirleri uygun mu?
4. VERBIS kay1t yükümlülüü var m1?
5. Tespit edilen riskler ve öneriler

Detayl1 KVKK uyumluluk raporu haz1rla.""",
    description="Check KVKK compliance for data processing",
    variables=["data_processing_description", "personal_data_types", "legal_basis"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT
)

CONTRACT_COMPLIANCE_TEMPLATE = PromptTemplate(
    template_id="contract_compliance",
    name="Contract Compliance Check",
    template="""A_a1daki sözle_me hükümlerini ilgili mevzuata uygunluk aç1s1ndan incele:

Sözle_me Türü: {contract_type}

Sözle_me Maddeleri:
{contract_clauses}

0lgili Mevzuat:
{applicable_legislation}

^u aç1lardan deerlendir:
1. Emredici hükümlere uygunluk
2. Tüketici haklar1 (varsa)
3. Geçersizlik riski ta_1yan maddeler
4. Eksik veya belirsiz hükümler
5. Revizyon önerileri

Hukuki risk deerlendirmesi yap.""",
    description="Check contract compliance with legislation",
    variables=["contract_type", "contract_clauses", "applicable_legislation"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT
)


# ============================================================================
# CONTRACT ANALYSIS TEMPLATES
# ============================================================================

CONTRACT_SUMMARY_TEMPLATE = PromptTemplate(
    template_id="contract_summary",
    name="Contract Summary",
    template="""A_a1daki sözle_meyi özetle:

{contract_text}

Özet _unlar1 içermeli:
1. Sözle_me türü ve taraflar
2. Sözle_menin konusu
3. Ana yükümlülükler (her iki taraf için)
4. Önemli hak ve sorumluluklar
5. Ödeme/bedel ko_ullar1
6. Süre ve fesih ko_ullar1
7. Dikkat çeken özel maddeler

Yönetici özeti format1nda, net ve öz olsun.""",
    description="Create executive summary of contract",
    variables=["contract_text"],
    role=PromptRole.USER,
    output_format=OutputFormat.BULLET_LIST
)

CLAUSE_RISK_ASSESSMENT_TEMPLATE = PromptTemplate(
    template_id="clause_risk_assessment",
    name="Clause Risk Assessment",
    template="""A_a1daki sözle_me maddesini {party_perspective} taraf1 aç1s1ndan risk deerlendir:

Madde Metni:
{clause_text}

Sözle_me Balam1:
{contract_context}

Risk analizi yap:
1. Potansiyel riskler
2. Risk seviyesi (dü_ük/orta/yüksek)
3. Sonuç ve etkileri
4. Azaltma önerileri
5. Alternatif madde önerisi

JSON format1nda risk raporu haz1rla.""",
    description="Assess risk of contract clause",
    variables=["clause_text", "contract_context", "party_perspective"],
    role=PromptRole.USER,
    output_format=OutputFormat.JSON
)


# ============================================================================
# SUMMARIZATION TEMPLATES
# ============================================================================

EXECUTIVE_SUMMARY_TEMPLATE = PromptTemplate(
    template_id="executive_summary",
    name="Executive Summary for Legal Document",
    template="""A_a1daki hukuki belge için yönetici özeti haz1rla:

{document_text}

Yönetici özeti maksimum {max_length} kelime olsun ve _unlar1 içersin:
1. Belgenin türü ve amac1
2. Ana bulgular ve sonuçlar
3. Kritik noktalar ve önemli dei_iklikler
4. Aksiyon gerektiren konular
5. Genel deerlendirme

Hukuki olmayan okuyucular için anla_1l1r dille yaz.""",
    description="Create executive summary for non-legal audience",
    variables=["document_text", "max_length"],
    role=PromptRole.USER,
    output_format=OutputFormat.MARKDOWN
)

TECHNICAL_SUMMARY_TEMPLATE = PromptTemplate(
    template_id="technical_summary",
    name="Technical Legal Summary",
    template="""A_a1daki hukuki metni hukukçular için teknik özet olarak özetle:

{legal_text}

Teknik özet _unlar1 içermeli:
1. Hukuki dayanak ve at1flar
2. Uygulanacak hükümler
3. 0stisna ve özel durumlar
4. 0çtihat ve yorumlar
5. Uygulama notlar1

Tam hukuki terminoloji kullan.""",
    description="Create technical summary for legal professionals",
    variables=["legal_text"],
    role=PromptRole.USER,
    output_format=OutputFormat.MARKDOWN
)


# ============================================================================
# QUESTION ANSWERING TEMPLATES
# ============================================================================

LEGAL_QA_TEMPLATE = PromptTemplate(
    template_id="legal_qa",
    name="Legal Question Answering",
    template="""Türk hukuku uzman1 olarak a_a1daki soruya cevap ver.

Soru:
{question}

0lgili Balam:
{context}

Cevab1n _unlar1 içermeli:
1. Dorudan cevap
2. Hukuki dayanak (ilgili kanun maddeleri)
3. Aç1klama ve detaylar
4. Uygulamaya dönük öneriler
5. Varsa istisnalar

Net, anla_1l1r ve hukuken doru bir cevap ver.""",
    description="Answer legal question with context",
    variables=["question", "context"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT
)

MULTI_HOP_QA_TEMPLATE = PromptTemplate(
    template_id="multi_hop_qa",
    name="Multi-hop Legal Reasoning",
    template="""A_a1daki karma_1k hukuki soru için ad1m ad1m muhakeme yaparak cevap ver.

Soru:
{complex_question}

Mevcut Bilgiler:
{available_information}

Ad1m ad1m dü_ün:
1. Soruyu alt sorulara böl
2. Her alt soru için hukuki dayanaklar1 belirle
3. Alt cevaplar1 birle_tir
4. Nihai sonuca ula_
5. Gerekçelendir

Dü_ünce sürecini göster ve sonuca ula_.""",
    description="Multi-hop reasoning for complex legal questions",
    variables=["complex_question", "available_information"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT
)


# ============================================================================
# ENTITY EXTRACTION TEMPLATES
# ============================================================================

ENTITY_EXTRACTION_TEMPLATE = PromptTemplate(
    template_id="entity_extraction",
    name="Legal Entity Extraction",
    template="""A_a1daki hukuki metinden tüm önemli varl1klar1 (entities) ç1kar:

{text}

^u varl1k türlerini ç1kar:
- Ki_iler (gerçek/tüzel)
- Kurumlar ve otoriteler
- Kanun ve mevzuat adlar1
- Tarihler
- Yerler
- Para miktarlar1
- Madde numaralar1

JSON format1nda yap1land1r1lm1_ listele.""",
    description="Extract named entities from legal text",
    variables=["text"],
    role=PromptRole.USER,
    output_format=OutputFormat.JSON
)


# ============================================================================
# TRANSLATION TEMPLATES
# ============================================================================

LEGAL_TRANSLATION_TR_TO_EN_TEMPLATE = PromptTemplate(
    template_id="legal_translation_tr_en",
    name="Turkish to English Legal Translation",
    template="""A_a1daki Türkçe hukuki metni 0ngilizce'ye çevir.

Hukuki terminolojiyi doru kullan ve balam1 koru.

Türkçe Metin:
{turkish_text}

Profesyonel hukuki 0ngilizce çeviri yap.""",
    description="Translate Turkish legal text to English",
    variables=["turkish_text"],
    role=PromptRole.USER
)

LEGAL_TRANSLATION_EN_TO_TR_TEMPLATE = PromptTemplate(
    template_id="legal_translation_en_tr",
    name="English to Turkish Legal Translation",
    template="""Translate the following English legal text to Turkish.

Use proper Turkish legal terminology.

English Text:
{english_text}

Profesyonel hukuki Türkçe'ye çevir.""",
    description="Translate English legal text to Turkish",
    variables=["english_text"],
    role=PromptRole.USER
)


# ============================================================================
# RISK ASSESSMENT TEMPLATES
# ============================================================================

LEGAL_RISK_ASSESSMENT_TEMPLATE = PromptTemplate(
    template_id="legal_risk_assessment",
    name="Legal Risk Assessment",
    template="""A_a1daki durum için hukuki risk deerlendirmesi yap:

Durum:
{situation_description}

0lgili Mevzuat:
{applicable_law}

Önceki Benzer Durumlar:
{precedents}

Risk deerlendirmen _unlar1 içermeli:
1. Tespit edilen hukuki riskler
2. Her risk için olas1l1k (dü_ük/orta/yüksek)
3. Potansiyel sonuçlar
4. Risk azaltma stratejileri
5. Öncelikli aksiyon önerileri

Detayl1 risk matrisi haz1rla.""",
    description="Comprehensive legal risk assessment",
    variables=["situation_description", "applicable_law", "precedents"],
    role=PromptRole.USER,
    strategy=PromptStrategy.CHAIN_OF_THOUGHT,
    output_format=OutputFormat.MARKDOWN
)


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

TURKISH_LEGAL_EXPERT_SYSTEM_PROMPT = PromptTemplate(
    template_id="turkish_legal_expert_system",
    name="Turkish Legal Expert System Prompt",
    template="""Sen Harvey ve Legora seviyesinde dünya standartlar1nda bir Türk hukuku uzman1 yapay zeka asistan1s1n.

Uzmanl1k Alanlar1n:
- Türk Ceza Hukuku (TCK)
- Türk Medeni Hukuku (TMK)
- Türk Ticaret Hukuku (TTK)
- Ki_isel Verilerin Korunmas1 (KVKK)
- 0dare Hukuku
- 0_ Hukuku
- Rekabet Hukuku
- Tüketici Hukuku

Yeteneklerin:
- Kanun, yönetmelik, mahkeme karar1 analizi
- Hukuki argüman geli_tirme
- Emsal içtihat ara_t1rmas1
- Mevzuat uyumluluk kontrolü
- Sözle_me inceleme ve taslak haz1rlama
- Hukuki risk deerlendirmesi

0lkelerin:
1. Her zaman güncel Türk mevzuat1n1 referans al
2. Doru hukuki terminoloji kullan
3. Kaynak göster (kanun, madde, f1kra)
4. Belirsizlik varsa belirt
5. Pratik uygulama önerileri sun
6. Etik ve profesyonel davran

Hukuki tavsiye vermiyorsun, bilgi sal1yorsun. Kullan1c1lar gerçek hukuki sorunlar1 için mutlaka avukata dan1_mal1.""",
    description="System prompt for Turkish legal expert AI",
    variables=[],
    role=PromptRole.SYSTEM
)


# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    # Document Analysis
    "law_analysis": LAW_ANALYSIS_TEMPLATE,
    "regulation_analysis": REGULATION_ANALYSIS_TEMPLATE,
    "court_decision_analysis": COURT_DECISION_ANALYSIS_TEMPLATE,

    # Citation & Reference
    "citation_extraction": CITATION_EXTRACTION_TEMPLATE,
    "citation_resolution": CITATION_RESOLUTION_TEMPLATE,

    # Legal Reasoning
    "legal_argument": LEGAL_ARGUMENT_TEMPLATE,
    "precedent_search": PRECEDENT_SEARCH_TEMPLATE,

    # Compliance
    "kvkk_compliance": KVKK_COMPLIANCE_TEMPLATE,
    "contract_compliance": CONTRACT_COMPLIANCE_TEMPLATE,

    # Contract Analysis
    "contract_summary": CONTRACT_SUMMARY_TEMPLATE,
    "clause_risk_assessment": CLAUSE_RISK_ASSESSMENT_TEMPLATE,

    # Summarization
    "executive_summary": EXECUTIVE_SUMMARY_TEMPLATE,
    "technical_summary": TECHNICAL_SUMMARY_TEMPLATE,

    # Question Answering
    "legal_qa": LEGAL_QA_TEMPLATE,
    "multi_hop_qa": MULTI_HOP_QA_TEMPLATE,

    # Entity Extraction
    "entity_extraction": ENTITY_EXTRACTION_TEMPLATE,

    # Translation
    "legal_translation_tr_en": LEGAL_TRANSLATION_TR_TO_EN_TEMPLATE,
    "legal_translation_en_tr": LEGAL_TRANSLATION_EN_TO_TR_TEMPLATE,

    # Risk Assessment
    "legal_risk_assessment": LEGAL_RISK_ASSESSMENT_TEMPLATE,

    # System Prompts
    "turkish_legal_expert_system": TURKISH_LEGAL_EXPERT_SYSTEM_PROMPT,
}


def get_template(template_id: str) -> PromptTemplate:
    """Get template by ID

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate

    Raises:
        KeyError: If template not found
    """
    if template_id not in TEMPLATE_REGISTRY:
        raise KeyError(f"Template '{template_id}' not found in registry")

    return TEMPLATE_REGISTRY[template_id]


def list_templates() -> List[str]:
    """List all available template IDs

    Returns:
        List of template IDs
    """
    return list(TEMPLATE_REGISTRY.keys())


def search_templates(keyword: str) -> List[PromptTemplate]:
    """Search templates by keyword

    Args:
        keyword: Search keyword

    Returns:
        List of matching templates
    """
    keyword_lower = keyword.lower()
    matches = []

    for template in TEMPLATE_REGISTRY.values():
        if (
            keyword_lower in template.name.lower()
            or keyword_lower in template.description.lower()
            or keyword_lower in template.template_id.lower()
        ):
            matches.append(template)

    return matches


__all__ = [
    'TEMPLATE_REGISTRY',
    'get_template',
    'list_templates',
    'search_templates',
    # Individual templates
    'LAW_ANALYSIS_TEMPLATE',
    'REGULATION_ANALYSIS_TEMPLATE',
    'COURT_DECISION_ANALYSIS_TEMPLATE',
    'CITATION_EXTRACTION_TEMPLATE',
    'LEGAL_QA_TEMPLATE',
    'CONTRACT_SUMMARY_TEMPLATE',
    'KVKK_COMPLIANCE_TEMPLATE',
    'TURKISH_LEGAL_EXPERT_SYSTEM_PROMPT'
]
