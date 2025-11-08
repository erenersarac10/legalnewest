# Harvey/Legora CTO-Level Integration Guide

## 🎯 Overview

This guide demonstrates how to use the **4 new Harvey/Legora quality components** together to create a production-grade Turkish Legal AI system with:

- ✅ **Mandatory Risk Scoring** (no optional fields)
- ✅ **Multi-source Risk Assessment** (Hallucination + RAG + Reasoning)
- ✅ **Config-based Jurisdiction Profiles** (Tenant overrides)
- ✅ **Multi-level Explainability** (Summary/Standard/Full/Technical)
- ✅ **Response Validation Guardrails** (Invalid responses BLOCKED)

---

## 📦 New Components

### 1. `legal_opinion_validator.py`
**Purpose:** Response validation guardrails - ensures every legal opinion meets Harvey/Legora standards.

**Key Features:**
- MANDATORY field enforcement (risk_score, citations, sources)
- Citation quality validation (min citations by risk level)
- Risk assessment completeness
- Compliance checks (jurisdiction-specific warnings)
- Auto-fix suggestions

**Hard Requirements:**
```python
# These fields are MANDATORY - no exceptions
required_fields = {
    "risk_level": RiskLevel,  # Cannot be None
    "citations": List[str],  # Min 1 citation
    "confidence_score": float,  # 0-100 range
    "disclaimers": List[str],  # Required for compliance
}
```

### 2. `legal_risk_scorer.py`
**Purpose:** Multi-source risk aggregation - combines Hallucination + RAG + Reasoning into single risk score.

**Risk Model:**
```python
risk_score = (
    0.40 × hallucination_risk +  # 40% - Fake citations
    0.30 × rag_quality_risk +    # 30% - Source coverage
    0.30 × reasoning_risk        # 30% - Analysis uncertainty
)
```

**Risk Levels:**
- 🟢 **LOW** (0.00-0.10): 90-100% confidence
- 🟡 **MEDIUM** (0.11-0.30): 70-89% confidence
- 🟠 **HIGH** (0.31-0.50): 50-69% confidence
- 🔴 **CRITICAL** (0.51-1.00): <50% confidence

### 3. `jurisdiction_config.py`
**Purpose:** Config-based jurisdiction profiles - flexible, tenant-specific customization.

**Features:**
- Load profiles from JSON/YAML/Database
- Tenant-specific overrides (e.g., stricter evidence threshold)
- Practice area profiles (Banking, IP, Insurance, etc.)
- Hot reloading (no restart needed)
- Profile versioning

**Example Tenant Override:**
```json
{
  "type": "tenant_override",
  "tenant_id": "law_firm_xyz",
  "jurisdiction": "criminal",
  "overrides": {
    "evidence_threshold": 0.9,
    "strict_construction": true
  }
}
```

### 4. `explainability_engine.py`
**Purpose:** Multi-level explanations - tailored for different audiences.

**Explanation Levels:**
- **SUMMARY** (3-4 sentences): Quick review
- **STANDARD** (3-5 paragraphs): Most users
- **FULL** (9 sections): Legal professionals
- **TECHNICAL** (Full + metrics): Developers/auditors

---

## 🔧 Integration Example

Here's how to use all components together:

```python
from backend.services.legal_reasoning_service import LegalReasoningService, LegalJurisdiction
from backend.services.hallucination_detector import HallucinationDetector
from backend.services.legal_risk_scorer import LegalRiskScorer
from backend.services.legal_opinion_validator import LegalOpinionValidator
from backend.services.explainability_engine import ExplainabilityEngine
from backend.services.jurisdiction_config import get_profile_manager

# =============================================================================
# STEP 1: Setup
# =============================================================================

# Initialize services
reasoning_service = LegalReasoningService()
hallucination_detector = HallucinationDetector()
risk_scorer = LegalRiskScorer()
validator = LegalOpinionValidator(strict_mode=True)
explainability = ExplainabilityEngine()
profile_manager = get_profile_manager()

# =============================================================================
# STEP 2: Get Tenant-Specific Profile
# =============================================================================

# Load tenant-specific profile (with overrides)
tenant_id = "law_firm_xyz"
jurisdiction = LegalJurisdiction.CRIMINAL

profile = profile_manager.get_profile(
    jurisdiction=jurisdiction,
    tenant_id=tenant_id,  # Applies tenant overrides
)

print(f"Profile for {tenant_id}: {profile.evidence_threshold}")
# Example: 0.9 (stricter than default 0.8 for criminal law)

# =============================================================================
# STEP 3: Generate Legal Opinion
# =============================================================================

question = "TCK 86'ya göre devleti alenen aşağılama suçu nedir?"
statutes = ["TCK Madde 86: Devleti ve milletini alenen aşağılama..."]
retrieval_context = [
    "TCK Madde 86 metni...",
    "Yargıtay 4. HD, 2020/1234 kararı...",
]

# Generate opinion with explainability enabled
opinion = await reasoning_service.generate_opinion(
    question=question,
    statutes=statutes,
    jurisdiction=jurisdiction,
    enable_explainability=True,  # CRITICAL for HIGH/CRITICAL risk
)

# =============================================================================
# STEP 4: Hallucination Detection
# =============================================================================

hallucination_result = await hallucination_detector.detect(
    response_text=opinion.legal_analysis,
    retrieval_context=retrieval_context,
)

print(f"Hallucination detected: {hallucination_result.is_hallucination}")
print(f"Confidence: {hallucination_result.confidence_score}%")
print(f"Unverified citations: {len(hallucination_result.unverified_citations)}")

# =============================================================================
# STEP 5: Comprehensive Risk Scoring
# =============================================================================

# Calculate final risk score (Hallucination + RAG + Reasoning)
risk_assessment = risk_scorer.score(
    opinion=opinion,
    hallucination_result=hallucination_result,
    retrieval_context=retrieval_context,
)

# Update opinion with final risk score
opinion.risk_level = risk_assessment.risk_level
opinion.confidence_score = risk_assessment.confidence_score

print(f"\n🎯 Final Risk Assessment:")
print(f"   Risk Score: {risk_assessment.risk_score:.3f}")
print(f"   Risk Level: {risk_assessment.risk_level.value}")
print(f"   Confidence: {risk_assessment.confidence_score:.1f}%")

# Print risk breakdown
print(f"\n📊 Risk Breakdown:")
for factor in risk_assessment.risk_factors:
    print(f"   - {factor.name}: {factor.value:.3f} (weight: {factor.weight})")

# Recommendations
print(f"\n💡 Recommendations:")
for rec in risk_assessment.recommendations:
    print(f"   - {rec}")

# =============================================================================
# STEP 6: Response Validation (GUARDRAILS)
# =============================================================================

# Validate opinion against Harvey/Legora standards
validation_result = validator.validate(opinion)

if not validation_result.is_valid:
    print(f"\n❌ VALIDATION FAILED! ({len(validation_result.errors)} errors)")

    for error in validation_result.errors:
        print(f"   - [{error.severity.value}] {error.field}: {error.message}")

    # Attempt auto-fix
    print(f"\n🔧 Attempting auto-fix...")
    opinion = validator.auto_fix(opinion)

    # Re-validate
    validation_result = validator.validate(opinion)

    if validation_result.is_valid:
        print(f"✅ Auto-fix successful!")
    else:
        # BLOCK RESPONSE - cannot publish
        raise ValueError(
            f"Invalid legal opinion - cannot publish!\n"
            f"Errors: {[e.message for e in validation_result.errors]}"
        )
else:
    print(f"\n✅ Validation passed!")

# Print warnings (non-blocking)
if validation_result.has_warnings:
    print(f"\n⚠️  Warnings ({len(validation_result.warnings)}):")
    for warning in validation_result.warnings:
        print(f"   - {warning.field}: {warning.message}")

# =============================================================================
# STEP 7: Multi-Level Explainability
# =============================================================================

print(f"\n" + "=" * 80)
print(f"EXPLANATIONS")
print(f"=" * 80)

# Summary explanation (for quick review)
print(f"\n📝 SUMMARY:")
summary = explainability.explain(opinion, level="summary")
print(summary)

# Standard explanation (for most users)
print(f"\n📄 STANDARD:")
standard = explainability.explain(opinion, level="standard")
print(standard[:500] + "...")  # Truncated for display

# Full explanation (for legal professionals)
print(f"\n📚 FULL EXPLANATION:")
full = explainability.explain(opinion, level="full")
print(full[:500] + "...")  # Truncated for display

# Technical explanation (for developers/auditors)
technical = explainability.explain(opinion, level="technical")
# (Not printed - too long)

# =============================================================================
# STEP 8: Audit Trail (KVKK-Compliant)
# =============================================================================

# Generate audit trail for compliance logging
audit = explainability.generate_audit_trail(
    opinion=opinion,
    tenant_id=tenant_id,
    user_id="user_12345_anonymized",  # Anonymized
    session_id="session_abc123",
)

print(f"\n📋 Audit Trail (KVKK-compliant):")
print(f"   Tenant: {audit.tenant_id}")
print(f"   Jurisdiction: {audit.jurisdiction}")
print(f"   Risk Level: {audit.risk_level}")
print(f"   Confidence: {audit.confidence_score}%")
print(f"   Sources: {len(audit.statutes_used)} statutes, {len(audit.cases_used)} cases")
print(f"   Timestamp: {audit.timestamp}")

# NOTE: Question/answer text NOT included in audit trail (KVKK compliance)

# =============================================================================
# STEP 9: Final Response (API Format)
# =============================================================================

# Build final API response
api_response = {
    # Core answer
    "question": opinion.question,
    "answer": opinion.short_answer,

    # MANDATORY FIELDS (Harvey/Legora requirement)
    "risk_score": risk_assessment.risk_score,
    "risk_level": opinion.risk_level.value,
    "confidence_score": opinion.confidence_score,
    "citations": opinion.citations,

    # Compliance
    "disclaimers": opinion.disclaimers,
    "compliance_warnings": opinion.compliance_warnings,

    # Explainability
    "explanation": {
        "summary": summary,
        "standard": standard,
        "full_available": True,
    },

    # Metadata
    "metadata": {
        "jurisdiction": opinion.jurisdiction.value,
        "reasoning_method": opinion.reasoning_method.value,
        "validation_passed": validation_result.is_valid,
        "audit_trail_id": audit.session_id,
    },
}

print(f"\n✅ FINAL API RESPONSE:")
import json
print(json.dumps(api_response, ensure_ascii=False, indent=2)[:1000] + "...")

```

---

## 🚨 Critical Enforcement Points

### 1. **No Optional Risk/Citations**
```python
# ❌ BAD - Optional fields
if opinion.citations:  # Can be None or empty
    process_citations()

# ✅ GOOD - Validator enforces
validation_result = validator.validate(opinion)
if not validation_result.is_valid:
    raise ValueError("Invalid opinion - citations required!")
```

### 2. **Multi-Source Risk Scoring**
```python
# ❌ BAD - Only reasoning confidence
risk_score = 1.0 - (opinion.confidence_score / 100)

# ✅ GOOD - Combined sources
risk_assessment = risk_scorer.score(
    opinion=opinion,
    hallucination_result=hallucination_result,
    retrieval_context=retrieval_context,
)
```

### 3. **Tenant-Specific Profiles**
```python
# ❌ BAD - Hard-coded profiles
profile = CRIMINAL_LAW_PROFILE

# ✅ GOOD - Tenant overrides
profile = profile_manager.get_profile(
    jurisdiction=LegalJurisdiction.CRIMINAL,
    tenant_id=tenant_id,  # Applies custom thresholds
)
```

### 4. **Explainability Enforcement**
```python
# ❌ BAD - Optional explainability
explanation = opinion.explanation_trace or "No trace available"

# ✅ GOOD - Validator enforces for HIGH/CRITICAL
if opinion.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
    if not opinion.explanation_trace:
        raise ValueError("ExplanationTrace REQUIRED for HIGH/CRITICAL risk")
```

---

## 📊 Metrics & Monitoring

All components track metrics for production monitoring:

```python
# Validation metrics
print(f"Validation time: {validation_result.validation_time_ms:.1f}ms")
print(f"Errors: {len(validation_result.errors)}")
print(f"Warnings: {len(validation_result.warnings)}")

# Risk scoring metrics
print(f"Risk calculation breakdown:")
for key, value in risk_assessment.breakdown.items():
    print(f"   {key}: {value:.3f}")

# Audit trail for compliance
logger.info(
    f"Legal opinion generated: "
    f"tenant={audit.tenant_id}, "
    f"jurisdiction={audit.jurisdiction}, "
    f"risk={audit.risk_level}, "
    f"confidence={audit.confidence_score}%"
)
```

---

## 🎯 Harvey/Legora Quality Checklist

✅ **Response Validation:**
- [ ] Every response passes `LegalOpinionValidator`
- [ ] Invalid responses are BLOCKED (not published)
- [ ] Auto-fix applied when possible

✅ **Risk Scoring:**
- [ ] Multi-source risk (Hallucination + RAG + Reasoning)
- [ ] Risk score is MANDATORY (not optional)
- [ ] Risk recommendations generated

✅ **Jurisdiction Profiles:**
- [ ] Tenant-specific overrides supported
- [ ] Profiles loaded from config (not hard-coded)
- [ ] Criminal law uses stricter thresholds

✅ **Explainability:**
- [ ] Multi-level explanations (summary/standard/full)
- [ ] HIGH/CRITICAL risk requires full trace
- [ ] Audit trail generated (KVKK-compliant)

✅ **Citations:**
- [ ] Minimum citations enforced by risk level
- [ ] Hallucination detection validates citations
- [ ] Citation-less responses REJECTED

---

## 🚀 Production Deployment

### Environment Setup
```bash
# Set config directory for jurisdiction profiles
export JURISDICTION_CONFIG_DIR=/etc/legal_ai/profiles

# Enable database for tenant overrides
export ENABLE_PROFILE_DB=true

# Enable strict validation
export STRICT_VALIDATION=true
```

### Example Config File (`/etc/legal_ai/profiles/tenant_xyz.json`)
```json
{
  "type": "tenant_override",
  "tenant_id": "law_firm_xyz",
  "jurisdiction": "criminal",
  "description": "Stricter evidence threshold for criminal cases",
  "overrides": {
    "evidence_threshold": 0.9,
    "strict_construction": true,
    "requires_disclaimer": true
  }
}
```

---

## 📝 Summary

These 4 components together provide **Harvey/Legora CTO-level** quality:

1. **Validator** → Ensures every response meets standards (no invalid responses)
2. **Risk Scorer** → Multi-source risk (no guessing - real assessment)
3. **Profile Manager** → Tenant customization (enterprise-grade flexibility)
4. **Explainability** → Tailored explanations (transparency for all audiences)

**Result:** Production-ready Turkish Legal AI with mandatory risk scoring, citations, and compliance! 🎯
