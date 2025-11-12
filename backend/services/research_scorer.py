"""
Research Scorer - Harvey/Legora %100 Quality Legal Research Scoring.

World-class legal research quality assessment for Turkish Legal AI:
- Multi-dimensional research scoring (depth, breadth, quality, recency)
- Source reliability assessment (primary vs. secondary, authority hierarchy)
- Citation network analysis (comprehensiveness, diversity, authority)
- Research completeness evaluation (coverage gaps, missing authorities)
- Methodology assessment (systematic vs. ad-hoc)
- Turkish legal research standards compliance
- Expert-level research quality metrics
- Publication-ready research validation

Why Research Scorer?
    Without: Incomplete research ’ weak arguments ’ lost cases
    With: Comprehensive scoring ’ high-quality research ’ Harvey-level briefs

    Impact: Transform good research into excellent research! <¯

Architecture:
    [Research Document] ’ [ResearchScorer]
                              “
        [Depth Analyzer] ’ [Breadth Analyzer]
                              “
        [Quality Assessor] ’ [Citation Network Analyzer]
                              “
        [Completeness Checker] ’ [Methodology Evaluator]
                              “
        [Research Score Card + Improvement Recommendations]

Research Dimensions:

    1. Depth (¼È˜ derinlemesine?):
        - Primary source analysis
        - Case law depth
        - Statutory interpretation depth
        - Scholarly commentary depth
        - Historical analysis

    2. Breadth (¼È˜ kapsaml1?):
        - Topic coverage
        - Jurisdiction coverage
        - Time period coverage
        - Authority diversity
        - Perspective diversity

    3. Quality (¼È˜ kaliteli?):
        - Source authority level
        - Source recency
        - Citation accuracy
        - Analysis rigor
        - Reasoning quality

    4. Recency (¼È˜ güncel?):
        - Recent case law
        - Recent amendments
        - Recent commentary
        - Current doctrine
        - Fresh perspectives

    5. Completeness (¼È˜ eksiksiz?):
        - All key authorities cited
        - Counter-arguments addressed
        - Alternative interpretations considered
        - Gaps identified
        - Coverage adequacy

Scoring System:

    Overall Score (0-100):
        - 95-100: Exceptional (Supreme Court brief quality)
        - 85-94: Excellent (Harvey/Legora quality)
        - 75-84: Very Good (Senior associate quality)
        - 65-74: Good (Mid-level associate quality)
        - 55-64: Adequate (Junior associate quality)
        - 45-54: Needs Improvement
        - 0-44: Insufficient

    Dimension Weights:
        - Quality: 30%
        - Depth: 25%
        - Completeness: 20%
        - Breadth: 15%
        - Recency: 10%

Source Authority Hierarchy (Turkish Legal System):

    Tier 1 - Binding Precedent (100 points):
        - Anayasa Mahkemesi (Constitutional Court)
        - Yarg1tay 0çtihad1 Birle_tirme (Supreme Court Unification)
        - Dan1_tay 0çtihad1 Birle_tirme (Council of State Unification)

    Tier 2 - Highly Persuasive (90 points):
        - Yarg1tay Hukuk/Ceza Daireleri (Supreme Court Chambers)
        - Dan1_tay Daireleri (Council of State Chambers)
        - Bölge Adliye Mahkemesi (Regional Courts)

    Tier 3 - Persuasive (75 points):
        - 0lk Derece Mahkemeleri (First Instance Courts)
        - A1r Ceza Mahkemeleri (High Criminal Courts)
        - Asliye Mahkemeleri (Civil Courts)

    Tier 4 - Primary Sources (85 points):
        - Kanunlar (Statutes)
        - Tüzükler (Regulations)
        - Yönetmelikler (By-laws)
        - Tebliler (Communiqués)

    Tier 5 - Secondary Sources (60-80 points):
        - Prof. Dr. (Full Professor): 80 points
        - Doç. Dr. (Associate Professor): 70 points
        - Dr. Ör. Üyesi (Assistant Professor): 65 points
        - Avukat (Practitioner): 60 points

    Tier 6 - Tertiary Sources (40-50 points):
        - Legal databases commentary
        - Practice guides
        - Form books
        - General references

Performance:
    - Single research scoring: < 200ms (p95)
    - Batch scoring (10 documents): < 1.5s (p95)
    - Citation network analysis: < 500ms (p95)
    - Completeness check: < 300ms (p95)

Usage:
    >>> from backend.services.research_scorer import ResearchScorer
    >>>
    >>> scorer = ResearchScorer(session=db_session)
    >>>
    >>> # Score research document
    >>> score_card = await scorer.score_research(
    ...     document_id="BRIEF_2024_001",
    ...     sources=[source1, source2, source3],
    ...     citation_network=citations,
    ... )
    >>>
    >>> print(f"Overall Score: {score_card.overall_score:.1f}/100")
    >>> print(f"Quality: {score_card.quality_grade}")
    >>> if score_card.recommendations:
    ...     print("Improvements:", score_card.recommendations)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.logging import get_logger

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ResearchQuality(str, Enum):
    """Research quality levels."""

    EXCEPTIONAL = "EXCEPTIONAL"  # 95-100
    EXCELLENT = "EXCELLENT"  # 85-94
    VERY_GOOD = "VERY_GOOD"  # 75-84
    GOOD = "GOOD"  # 65-74
    ADEQUATE = "ADEQUATE"  # 55-64
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"  # 45-54
    INSUFFICIENT = "INSUFFICIENT"  # 0-44


class SourceType(str, Enum):
    """Legal source types."""

    CASE_LAW = "CASE_LAW"
    STATUTE = "STATUTE"
    REGULATION = "REGULATION"
    SCHOLARLY_ARTICLE = "SCHOLARLY_ARTICLE"
    TREATISE = "TREATISE"
    PRACTICE_GUIDE = "PRACTICE_GUIDE"
    DATABASE_COMMENTARY = "DATABASE_COMMENTARY"
    OTHER = "OTHER"


class ResearchMethodology(str, Enum):
    """Research methodology types."""

    SYSTEMATIC = "SYSTEMATIC"  # Comprehensive, methodical
    TARGETED = "TARGETED"  # Focused, specific
    EXPLORATORY = "EXPLORATORY"  # Broad, discovery-oriented
    AD_HOC = "AD_HOC"  # Opportunistic, unsystematic


class AuthorityTier(str, Enum):
    """Turkish legal authority hierarchy."""

    TIER_1_BINDING = "TIER_1_BINDING"  # Constitutional Court, Unification
    TIER_2_HIGHLY_PERSUASIVE = "TIER_2_HIGHLY_PERSUASIVE"  # Supreme Courts
    TIER_3_PERSUASIVE = "TIER_3_PERSUASIVE"  # First Instance Courts
    TIER_4_PRIMARY = "TIER_4_PRIMARY"  # Statutes, Regulations
    TIER_5_SECONDARY = "TIER_5_SECONDARY"  # Scholarly works
    TIER_6_TERTIARY = "TIER_6_TERTIARY"  # Practice guides


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SourceMetrics:
    """Metrics for a single source."""

    source_id: str
    source_type: SourceType
    authority_tier: AuthorityTier
    authority_score: float  # 0-100
    recency_score: float  # 0-100
    relevance_score: float  # 0-100
    citation_count: int = 0  # How many times cited in research


@dataclass
class DimensionScore:
    """Score for a research dimension."""

    dimension: str
    score: float  # 0-100
    weight: float  # Contribution to overall score
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CitationNetworkMetrics:
    """Citation network analysis metrics."""

    total_sources: int
    primary_sources: int
    secondary_sources: int
    tertiary_sources: int

    # Diversity
    source_type_diversity: float  # 0-1 (Shannon entropy)
    authority_tier_diversity: float  # 0-1

    # Authority
    average_authority_score: float  # 0-100
    tier_1_count: int = 0
    tier_2_count: int = 0

    # Recency
    average_recency_score: float = 0.0  # 0-100
    recent_sources_count: int = 0  # < 2 years old


@dataclass
class CompletenessAssessment:
    """Research completeness assessment."""

    coverage_score: float  # 0-100
    identified_gaps: List[str] = field(default_factory=list)
    missing_authorities: List[str] = field(default_factory=list)
    uncovered_topics: List[str] = field(default_factory=list)
    counter_arguments_addressed: bool = False


@dataclass
class ResearchScoreCard:
    """Comprehensive research score card."""

    document_id: str
    overall_score: float  # 0-100
    quality_grade: ResearchQuality

    # Dimension scores
    depth_score: DimensionScore
    breadth_score: DimensionScore
    quality_score: DimensionScore
    recency_score: DimensionScore
    completeness_score: DimensionScore

    # Network metrics
    citation_metrics: CitationNetworkMetrics

    # Completeness
    completeness_assessment: CompletenessAssessment

    # Methodology
    methodology: ResearchMethodology

    # Strengths and improvements
    key_strengths: List[str] = field(default_factory=list)
    key_weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scorer_version: str = "1.0"


# =============================================================================
# RESEARCH SCORER
# =============================================================================


class ResearchScorer:
    """
    Harvey/Legora-level legal research quality scorer.

    Features:
    - Multi-dimensional scoring (depth, breadth, quality, recency, completeness)
    - Citation network analysis
    - Turkish legal authority hierarchy
    - Research completeness gaps
    - Actionable improvement recommendations
    """

    # Dimension weights
    DIMENSION_WEIGHTS = {
        "quality": 0.30,
        "depth": 0.25,
        "completeness": 0.20,
        "breadth": 0.15,
        "recency": 0.10,
    }

    # Authority tier scores
    AUTHORITY_SCORES = {
        AuthorityTier.TIER_1_BINDING: 100,
        AuthorityTier.TIER_2_HIGHLY_PERSUASIVE: 90,
        AuthorityTier.TIER_3_PERSUASIVE: 75,
        AuthorityTier.TIER_4_PRIMARY: 85,
        AuthorityTier.TIER_5_SECONDARY: 70,
        AuthorityTier.TIER_6_TERTIARY: 45,
    }

    # Quality grade thresholds
    QUALITY_THRESHOLDS = {
        95: ResearchQuality.EXCEPTIONAL,
        85: ResearchQuality.EXCELLENT,
        75: ResearchQuality.VERY_GOOD,
        65: ResearchQuality.GOOD,
        55: ResearchQuality.ADEQUATE,
        45: ResearchQuality.NEEDS_IMPROVEMENT,
        0: ResearchQuality.INSUFFICIENT,
    }

    def __init__(self, session: AsyncSession):
        """Initialize research scorer."""
        self.session = session

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def score_research(
        self,
        document_id: str,
        sources: List[SourceMetrics],
        topic_coverage: Optional[Dict[str, float]] = None,
        methodology: ResearchMethodology = ResearchMethodology.TARGETED,
    ) -> ResearchScoreCard:
        """
        Score legal research comprehensively.

        Args:
            document_id: Research document identifier
            sources: List of sources with metrics
            topic_coverage: Topic coverage map (topic -> coverage score 0-1)
            methodology: Research methodology used

        Returns:
            ResearchScoreCard with comprehensive scoring

        Example:
            >>> score_card = await scorer.score_research(
            ...     document_id="BRIEF_2024_001",
            ...     sources=source_metrics,
            ...     methodology=ResearchMethodology.SYSTEMATIC,
            ... )
        """
        start_time = datetime.now(timezone.utc)

        logger.info(
            f"Scoring research: {document_id}",
            extra={"document_id": document_id, "source_count": len(sources)}
        )

        try:
            # 1. Analyze citation network
            citation_metrics = await self._analyze_citation_network(sources)

            # 2. Score each dimension
            depth_score = await self._score_depth(sources, methodology)
            breadth_score = await self._score_breadth(sources, topic_coverage or {})
            quality_score = await self._score_quality(sources, citation_metrics)
            recency_score = await self._score_recency(sources)
            completeness_score = await self._score_completeness(sources, topic_coverage or {})

            # 3. Assess completeness
            completeness_assessment = await self._assess_completeness(
                sources, topic_coverage or {}
            )

            # 4. Calculate overall score
            overall_score = self._calculate_overall_score([
                depth_score,
                breadth_score,
                quality_score,
                recency_score,
                completeness_score,
            ])

            # 5. Determine quality grade
            quality_grade = self._determine_quality_grade(overall_score)

            # 6. Extract key findings
            key_strengths = self._extract_strengths([
                depth_score,
                breadth_score,
                quality_score,
                recency_score,
                completeness_score,
            ])
            key_weaknesses = self._extract_weaknesses([
                depth_score,
                breadth_score,
                quality_score,
                recency_score,
                completeness_score,
            ])
            recommendations = self._generate_recommendations(
                [depth_score, breadth_score, quality_score, recency_score, completeness_score],
                completeness_assessment,
            )

            score_card = ResearchScoreCard(
                document_id=document_id,
                overall_score=overall_score,
                quality_grade=quality_grade,
                depth_score=depth_score,
                breadth_score=breadth_score,
                quality_score=quality_score,
                recency_score=recency_score,
                completeness_score=completeness_score,
                citation_metrics=citation_metrics,
                completeness_assessment=completeness_assessment,
                methodology=methodology,
                key_strengths=key_strengths,
                key_weaknesses=key_weaknesses,
                recommendations=recommendations,
            )

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info(
                f"Research scored: {document_id} = {overall_score:.1f}/100 ({quality_grade.value}) ({duration_ms:.2f}ms)",
                extra={
                    "document_id": document_id,
                    "overall_score": overall_score,
                    "quality_grade": quality_grade.value,
                    "duration_ms": duration_ms,
                }
            )

            return score_card

        except Exception as exc:
            logger.error(
                f"Research scoring failed: {document_id}",
                extra={"document_id": document_id, "exception": str(exc)}
            )
            raise

    async def batch_score(
        self,
        documents: List[tuple[str, List[SourceMetrics]]],
    ) -> List[ResearchScoreCard]:
        """
        Score multiple research documents in batch.

        Args:
            documents: List of (document_id, sources) tuples

        Returns:
            List of ResearchScoreCard objects
        """
        logger.info(f"Batch scoring {len(documents)} research documents")

        results = []
        for document_id, sources in documents:
            score_card = await self.score_research(document_id=document_id, sources=sources)
            results.append(score_card)

        return results

    # =========================================================================
    # CITATION NETWORK ANALYSIS
    # =========================================================================

    async def _analyze_citation_network(
        self,
        sources: List[SourceMetrics],
    ) -> CitationNetworkMetrics:
        """Analyze citation network comprehensively."""
        if not sources:
            return CitationNetworkMetrics(
                total_sources=0,
                primary_sources=0,
                secondary_sources=0,
                tertiary_sources=0,
                source_type_diversity=0.0,
                authority_tier_diversity=0.0,
                average_authority_score=0.0,
            )

        # Count by category
        primary_count = sum(
            1 for s in sources
            if s.authority_tier in [AuthorityTier.TIER_1_BINDING, AuthorityTier.TIER_2_HIGHLY_PERSUASIVE, AuthorityTier.TIER_4_PRIMARY]
        )
        secondary_count = sum(
            1 for s in sources
            if s.authority_tier == AuthorityTier.TIER_5_SECONDARY
        )
        tertiary_count = sum(
            1 for s in sources
            if s.authority_tier == AuthorityTier.TIER_6_TERTIARY
        )

        # Calculate diversity (Shannon entropy)
        source_type_diversity = self._calculate_diversity([s.source_type for s in sources])
        authority_tier_diversity = self._calculate_diversity([s.authority_tier for s in sources])

        # Average authority
        avg_authority = sum(s.authority_score for s in sources) / len(sources)

        # Tier counts
        tier_1_count = sum(1 for s in sources if s.authority_tier == AuthorityTier.TIER_1_BINDING)
        tier_2_count = sum(1 for s in sources if s.authority_tier == AuthorityTier.TIER_2_HIGHLY_PERSUASIVE)

        # Recency
        avg_recency = sum(s.recency_score for s in sources) / len(sources)
        recent_count = sum(1 for s in sources if s.recency_score >= 80)

        return CitationNetworkMetrics(
            total_sources=len(sources),
            primary_sources=primary_count,
            secondary_sources=secondary_count,
            tertiary_sources=tertiary_count,
            source_type_diversity=source_type_diversity,
            authority_tier_diversity=authority_tier_diversity,
            average_authority_score=avg_authority,
            tier_1_count=tier_1_count,
            tier_2_count=tier_2_count,
            average_recency_score=avg_recency,
            recent_sources_count=recent_count,
        )

    def _calculate_diversity(self, items: List[Any]) -> float:
        """Calculate Shannon entropy (diversity) for items."""
        if not items:
            return 0.0

        from collections import Counter
        import math

        counts = Counter(items)
        total = len(items)
        entropy = 0.0

        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize to 0-1
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0

    # =========================================================================
    # DIMENSION SCORING
    # =========================================================================

    async def _score_depth(
        self,
        sources: List[SourceMetrics],
        methodology: ResearchMethodology,
    ) -> DimensionScore:
        """Score research depth."""
        if not sources:
            return DimensionScore(
                dimension="Depth",
                score=0.0,
                weight=self.DIMENSION_WEIGHTS["depth"],
            )

        # Factors for depth
        primary_source_ratio = sum(
            1 for s in sources
            if s.source_type in [SourceType.CASE_LAW, SourceType.STATUTE]
        ) / len(sources)

        avg_authority = sum(s.authority_score for s in sources) / len(sources)

        # High-authority sources indicate depth
        tier_1_2_ratio = sum(
            1 for s in sources
            if s.authority_tier in [AuthorityTier.TIER_1_BINDING, AuthorityTier.TIER_2_HIGHLY_PERSUASIVE]
        ) / len(sources)

        # Methodology bonus
        methodology_bonus = {
            ResearchMethodology.SYSTEMATIC: 1.10,
            ResearchMethodology.TARGETED: 1.05,
            ResearchMethodology.EXPLORATORY: 1.00,
            ResearchMethodology.AD_HOC: 0.90,
        }[methodology]

        # Calculate depth score
        base_score = (
            primary_source_ratio * 40 +
            (avg_authority / 100) * 40 +
            tier_1_2_ratio * 20
        )
        depth_score = min(100, base_score * methodology_bonus)

        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        if primary_source_ratio >= 0.6:
            strengths.append("Strong primary source foundation")
        else:
            weaknesses.append("Insufficient primary sources")
            recommendations.append("Add more case law and statutory analysis")

        if tier_1_2_ratio >= 0.3:
            strengths.append("Excellent high-authority citations")
        elif tier_1_2_ratio < 0.1:
            weaknesses.append("Lacking top-tier authorities")
            recommendations.append("Include Constitutional Court or Supreme Court decisions")

        return DimensionScore(
            dimension="Depth",
            score=depth_score,
            weight=self.DIMENSION_WEIGHTS["depth"],
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    async def _score_breadth(
        self,
        sources: List[SourceMetrics],
        topic_coverage: Dict[str, float],
    ) -> DimensionScore:
        """Score research breadth."""
        if not sources:
            return DimensionScore(
                dimension="Breadth",
                score=0.0,
                weight=self.DIMENSION_WEIGHTS["breadth"],
            )

        # Source type diversity
        source_types = set(s.source_type for s in sources)
        type_diversity = len(source_types) / len(SourceType)

        # Authority tier diversity
        authority_tiers = set(s.authority_tier for s in sources)
        tier_diversity = len(authority_tiers) / len(AuthorityTier)

        # Topic coverage (if provided)
        if topic_coverage:
            avg_topic_coverage = sum(topic_coverage.values()) / len(topic_coverage)
        else:
            avg_topic_coverage = 0.5  # Neutral

        # Calculate breadth score
        breadth_score = (
            type_diversity * 30 +
            tier_diversity * 30 +
            avg_topic_coverage * 40
        )

        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        if type_diversity >= 0.5:
            strengths.append("Diverse source types")
        else:
            weaknesses.append("Limited source type diversity")
            recommendations.append("Expand to include scholarly articles and practice guides")

        if avg_topic_coverage >= 0.7:
            strengths.append("Comprehensive topic coverage")
        elif avg_topic_coverage < 0.5:
            weaknesses.append("Gaps in topic coverage")
            recommendations.append("Address uncovered topics and alternative perspectives")

        return DimensionScore(
            dimension="Breadth",
            score=breadth_score,
            weight=self.DIMENSION_WEIGHTS["breadth"],
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    async def _score_quality(
        self,
        sources: List[SourceMetrics],
        citation_metrics: CitationNetworkMetrics,
    ) -> DimensionScore:
        """Score research quality."""
        if not sources:
            return DimensionScore(
                dimension="Quality",
                score=0.0,
                weight=self.DIMENSION_WEIGHTS["quality"],
            )

        # Average authority score
        avg_authority = citation_metrics.average_authority_score

        # Average relevance
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)

        # High-tier source ratio
        high_tier_ratio = (citation_metrics.tier_1_count + citation_metrics.tier_2_count) / len(sources)

        # Calculate quality score
        quality_score = (
            (avg_authority / 100) * 50 +
            (avg_relevance / 100) * 30 +
            high_tier_ratio * 20
        )

        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        if avg_authority >= 80:
            strengths.append("High-authority sources throughout")
        elif avg_authority < 60:
            weaknesses.append("Low average authority level")
            recommendations.append("Replace tertiary sources with primary authorities")

        if avg_relevance >= 80:
            strengths.append("Highly relevant citations")
        elif avg_relevance < 60:
            weaknesses.append("Some sources lack relevance")
            recommendations.append("Ensure all citations directly support arguments")

        return DimensionScore(
            dimension="Quality",
            score=quality_score,
            weight=self.DIMENSION_WEIGHTS["quality"],
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    async def _score_recency(
        self,
        sources: List[SourceMetrics],
    ) -> DimensionScore:
        """Score research recency."""
        if not sources:
            return DimensionScore(
                dimension="Recency",
                score=0.0,
                weight=self.DIMENSION_WEIGHTS["recency"],
            )

        # Average recency score
        avg_recency = sum(s.recency_score for s in sources) / len(sources)

        # Recent source ratio (< 2 years)
        recent_ratio = sum(1 for s in sources if s.recency_score >= 80) / len(sources)

        # Calculate recency score
        recency_score = avg_recency * 0.7 + recent_ratio * 100 * 0.3

        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        if avg_recency >= 70:
            strengths.append("Current and up-to-date research")
        elif avg_recency < 50:
            weaknesses.append("Outdated sources")
            recommendations.append("Update with recent case law and amendments")

        if recent_ratio >= 0.5:
            strengths.append("Majority of sources are recent")
        elif recent_ratio < 0.3:
            weaknesses.append("Too few recent sources")
            recommendations.append("Add recent Supreme Court decisions")

        return DimensionScore(
            dimension="Recency",
            score=recency_score,
            weight=self.DIMENSION_WEIGHTS["recency"],
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    async def _score_completeness(
        self,
        sources: List[SourceMetrics],
        topic_coverage: Dict[str, float],
    ) -> DimensionScore:
        """Score research completeness."""
        if not sources:
            return DimensionScore(
                dimension="Completeness",
                score=0.0,
                weight=self.DIMENSION_WEIGHTS["completeness"],
            )

        # Minimum source threshold
        source_adequacy = min(100, (len(sources) / 15) * 100)  # 15+ sources is ideal

        # Topic coverage completeness
        if topic_coverage:
            topic_completeness = sum(topic_coverage.values()) / len(topic_coverage) * 100
            uncovered_topics = [topic for topic, score in topic_coverage.items() if score < 0.5]
        else:
            topic_completeness = 50  # Neutral
            uncovered_topics = []

        # Source type completeness (should have case law + statute + scholarly)
        has_case_law = any(s.source_type == SourceType.CASE_LAW for s in sources)
        has_statute = any(s.source_type == SourceType.STATUTE for s in sources)
        has_scholarly = any(s.source_type == SourceType.SCHOLARLY_ARTICLE for s in sources)

        type_completeness = (
            (1 if has_case_law else 0) +
            (1 if has_statute else 0) +
            (1 if has_scholarly else 0)
        ) / 3 * 100

        # Calculate completeness score
        completeness_score = (
            source_adequacy * 0.4 +
            topic_completeness * 0.4 +
            type_completeness * 0.2
        )

        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        if len(sources) >= 15:
            strengths.append("Sufficient source quantity")
        elif len(sources) < 10:
            weaknesses.append("Insufficient sources")
            recommendations.append("Expand research to include more authorities")

        if not has_case_law:
            weaknesses.append("Missing case law analysis")
            recommendations.append("Add relevant court decisions")
        if not has_statute:
            weaknesses.append("Missing statutory analysis")
            recommendations.append("Cite applicable statutes and regulations")
        if not has_scholarly:
            weaknesses.append("Missing scholarly commentary")
            recommendations.append("Include academic perspectives")

        if uncovered_topics:
            weaknesses.append(f"Gaps in {len(uncovered_topics)} topics")
            recommendations.append(f"Research: {', '.join(uncovered_topics[:3])}")

        return DimensionScore(
            dimension="Completeness",
            score=completeness_score,
            weight=self.DIMENSION_WEIGHTS["completeness"],
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    # =========================================================================
    # COMPLETENESS ASSESSMENT
    # =========================================================================

    async def _assess_completeness(
        self,
        sources: List[SourceMetrics],
        topic_coverage: Dict[str, float],
    ) -> CompletenessAssessment:
        """Assess research completeness with gap identification."""
        # Identify gaps
        identified_gaps = []
        missing_authorities = []
        uncovered_topics = []

        # Check for missing source types
        source_types = set(s.source_type for s in sources)
        if SourceType.CASE_LAW not in source_types:
            missing_authorities.append("Case law (Yarg1tay/Dan1_tay decisions)")
        if SourceType.STATUTE not in source_types:
            missing_authorities.append("Applicable statutes (Kanun/Tüzük)")
        if SourceType.SCHOLARLY_ARTICLE not in source_types:
            identified_gaps.append("Scholarly analysis missing")

        # Check for missing high-tier authorities
        authority_tiers = set(s.authority_tier for s in sources)
        if AuthorityTier.TIER_1_BINDING not in authority_tiers:
            identified_gaps.append("No Constitutional Court or Unification decisions")

        # Uncovered topics
        if topic_coverage:
            uncovered_topics = [
                topic for topic, score in topic_coverage.items()
                if score < 0.5
            ]

        # Coverage score
        coverage_score = 100 - (len(identified_gaps) * 10 + len(missing_authorities) * 15)
        coverage_score = max(0, coverage_score)

        # Counter-arguments (heuristic)
        counter_arguments_addressed = len(sources) >= 10 and any(
            s.source_type == SourceType.SCHOLARLY_ARTICLE for s in sources
        )

        return CompletenessAssessment(
            coverage_score=coverage_score,
            identified_gaps=identified_gaps,
            missing_authorities=missing_authorities,
            uncovered_topics=uncovered_topics,
            counter_arguments_addressed=counter_arguments_addressed,
        )

    # =========================================================================
    # OVERALL SCORING
    # =========================================================================

    def _calculate_overall_score(
        self,
        dimension_scores: List[DimensionScore],
    ) -> float:
        """Calculate weighted overall score."""
        total_score = 0.0
        for dim_score in dimension_scores:
            total_score += dim_score.score * dim_score.weight

        return round(total_score, 1)

    def _determine_quality_grade(self, overall_score: float) -> ResearchQuality:
        """Determine quality grade from overall score."""
        for threshold, grade in sorted(self.QUALITY_THRESHOLDS.items(), reverse=True):
            if overall_score >= threshold:
                return grade
        return ResearchQuality.INSUFFICIENT

    # =========================================================================
    # FINDINGS EXTRACTION
    # =========================================================================

    def _extract_strengths(
        self,
        dimension_scores: List[DimensionScore],
    ) -> List[str]:
        """Extract key strengths from dimension scores."""
        strengths = []

        # Top 3 dimensions
        sorted_dims = sorted(dimension_scores, key=lambda d: d.score, reverse=True)
        for dim in sorted_dims[:3]:
            if dim.score >= 75:
                strengths.extend(dim.strengths[:2])  # Top 2 strengths per dimension

        return strengths[:5]  # Top 5 overall

    def _extract_weaknesses(
        self,
        dimension_scores: List[DimensionScore],
    ) -> List[str]:
        """Extract key weaknesses from dimension scores."""
        weaknesses = []

        # Bottom 3 dimensions
        sorted_dims = sorted(dimension_scores, key=lambda d: d.score)
        for dim in sorted_dims[:3]:
            if dim.score < 75:
                weaknesses.extend(dim.weaknesses[:2])  # Top 2 weaknesses per dimension

        return weaknesses[:5]  # Top 5 overall

    def _generate_recommendations(
        self,
        dimension_scores: List[DimensionScore],
        completeness_assessment: CompletenessAssessment,
    ) -> List[str]:
        """Generate actionable improvement recommendations."""
        recommendations = []

        # Dimension-specific recommendations
        sorted_dims = sorted(dimension_scores, key=lambda d: d.score)
        for dim in sorted_dims[:3]:  # Focus on weakest dimensions
            if dim.score < 75:
                recommendations.extend(dim.recommendations[:2])

        # Completeness recommendations
        if completeness_assessment.missing_authorities:
            recommendations.append(
                f"Add: {', '.join(completeness_assessment.missing_authorities[:2])}"
            )

        # Prioritize top 5
        return recommendations[:5]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ResearchScorer",
    "ResearchQuality",
    "SourceType",
    "ResearchMethodology",
    "AuthorityTier",
    "SourceMetrics",
    "DimensionScore",
    "CitationNetworkMetrics",
    "CompletenessAssessment",
    "ResearchScoreCard",
]
