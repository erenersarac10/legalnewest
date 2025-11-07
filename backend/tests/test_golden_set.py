"""
Golden Test Set - Pytest Regression Test Suite.

Harvey/Legora %100: Automated regression prevention with %99 accuracy guarantee.

This module provides comprehensive pytest test suite for validating all legal
document parsers against the golden test set:
- Automated CI/CD integration
- Per-adapter test suites
- Regression detection
- Performance benchmarking
- %99 accuracy enforcement

Usage:
    # Run all tests
    pytest backend/tests/test_golden_set.py -v

    # Run specific adapter
    pytest backend/tests/test_golden_set.py -k "test_yargitay" -v

    # Run with coverage
    pytest backend/tests/test_golden_set.py --cov=backend.parsers.adapters

    # Generate HTML report
    pytest backend/tests/test_golden_set.py --html=report.html
"""

import pytest
import asyncio
from pathlib import Path
from typing import List

from backend.tests.golden_test_set import GoldenTestSet, GroundTruth
from backend.tests.golden_test_validator import GoldenTestValidator


# =============================================================================
# TEST CONFIGURATION
# =============================================================================


# Minimum accuracy threshold for Harvey/Legora %100 parity
MIN_ACCURACY_THRESHOLD = 0.99

# Maximum documents to test per adapter (for faster CI/CD)
MAX_DOCUMENTS_PER_ADAPTER = None  # None = all documents

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data" / "golden_test_set"


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def golden_test_set():
    """
    Load golden test set (session scope for performance).

    Returns:
        GoldenTestSet instance with loaded ground truth
    """
    test_set = GoldenTestSet(data_dir=TEST_DATA_DIR)

    # Try to load existing ground truth
    try:
        test_set.load_ground_truth()
    except FileNotFoundError:
        # Generate if not exists
        pytest.skip("Golden test set not generated. Run: python backend/tests/generate_golden_test_set.py")

    return test_set


@pytest.fixture(scope="session")
def validator():
    """
    Create validator instance (session scope).

    Returns:
        GoldenTestValidator configured for Harvey/Legora standards
    """
    return GoldenTestValidator(
        strict_mode=True,
        min_score_threshold=0.90,  # Individual test threshold (90%)
    )


@pytest.fixture
def resmi_gazete_ground_truth(golden_test_set):
    """Get Resmi Gazete ground truth."""
    return golden_test_set.get_adapter_ground_truth("resmi_gazete")


@pytest.fixture
def mevzuat_ground_truth(golden_test_set):
    """Get Mevzuat.gov.tr ground truth."""
    return golden_test_set.get_adapter_ground_truth("mevzuat_gov")


@pytest.fixture
def yargitay_ground_truth(golden_test_set):
    """Get Yargıtay ground truth."""
    return golden_test_set.get_adapter_ground_truth("yargitay")


@pytest.fixture
def danistay_ground_truth(golden_test_set):
    """Get Danıştay ground truth."""
    return golden_test_set.get_adapter_ground_truth("danistay")


@pytest.fixture
def aym_ground_truth(golden_test_set):
    """Get AYM ground truth."""
    return golden_test_set.get_adapter_ground_truth("aym")


# =============================================================================
# TEST SUITE - RESMI GAZETE
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
class TestResmiGazeteAdapter:
    """
    Test suite for Resmi Gazete adapter.

    Harvey/Legora %100: 60 test cases across 3 time periods.
    """

    async def test_resmi_gazete_accuracy(
        self,
        resmi_gazete_ground_truth,
        validator,
    ):
        """
        Test Resmi Gazete adapter overall accuracy.

        Requirement: %99 accuracy across all 60 test documents.
        """
        from backend.parsers.adapters.adapter_factory import get_factory

        factory = get_factory()
        adapter = factory.create("resmi_gazete")

        # Validate adapter
        results = validator.validate_adapter(
            adapter,
            resmi_gazete_ground_truth,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        # Generate report
        report = validator.generate_report(results, output_format="text")
        print("\n" + report)

        # Assert accuracy
        assert results.accuracy >= MIN_ACCURACY_THRESHOLD, (
            f"Resmi Gazete accuracy {results.accuracy:.1%} < {MIN_ACCURACY_THRESHOLD:.1%}\n"
            f"Failed tests: {results.failed_tests}/{results.total_tests}"
        )

    @pytest.mark.parametrize("category", ["historical", "modern", "contemporary"])
    async def test_resmi_gazete_by_period(
        self,
        resmi_gazete_ground_truth,
        validator,
        category,
    ):
        """
        Test Resmi Gazete adapter by time period.

        Ensures consistent accuracy across all historical periods.
        """
        # Filter by category
        category_ground_truth = [
            gt for gt in resmi_gazete_ground_truth
            if gt.category.value == category
        ]

        if not category_ground_truth:
            pytest.skip(f"No {category} test cases found")

        # Validate
        from backend.parsers.adapters.adapter_factory import get_factory
        factory = get_factory()
        adapter = factory.create("resmi_gazete")

        results = validator.validate_adapter(
            adapter,
            category_ground_truth,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        # More lenient for individual periods (95%)
        assert results.accuracy >= 0.95, (
            f"Resmi Gazete {category} accuracy {results.accuracy:.1%} < 95%"
        )


# =============================================================================
# TEST SUITE - MEVZUAT.GOV.TR
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
class TestMevzuatAdapter:
    """
    Test suite for Mevzuat.gov.tr adapter.

    Harvey/Legora %100: 60 test cases for consolidated legislation.
    """

    async def test_mevzuat_accuracy(
        self,
        mevzuat_ground_truth,
        validator,
    ):
        """
        Test Mevzuat.gov.tr adapter overall accuracy.

        Requirement: %99 accuracy across all 60 test documents.
        """
        from backend.parsers.adapters.adapter_factory import get_factory

        factory = get_factory()
        adapter = factory.create("mevzuat_gov")

        results = validator.validate_adapter(
            adapter,
            mevzuat_ground_truth,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        report = validator.generate_report(results, output_format="text")
        print("\n" + report)

        assert results.accuracy >= MIN_ACCURACY_THRESHOLD, (
            f"Mevzuat.gov.tr accuracy {results.accuracy:.1%} < {MIN_ACCURACY_THRESHOLD:.1%}"
        )


# =============================================================================
# TEST SUITE - YARGITAY
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
class TestYargitayAdapter:
    """
    Test suite for Yargıtay adapter.

    Harvey/Legora %100: 60 supreme court decisions.
    """

    async def test_yargitay_accuracy(
        self,
        yargitay_ground_truth,
        validator,
    ):
        """
        Test Yargıtay adapter overall accuracy.

        Requirement: %99 accuracy across all 60 test decisions.
        """
        from backend.parsers.adapters.adapter_factory import get_factory

        factory = get_factory()
        adapter = factory.create("yargitay")

        results = validator.validate_adapter(
            adapter,
            yargitay_ground_truth,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        report = validator.generate_report(results, output_format="text")
        print("\n" + report)

        assert results.accuracy >= MIN_ACCURACY_THRESHOLD, (
            f"Yargıtay accuracy {results.accuracy:.1%} < {MIN_ACCURACY_THRESHOLD:.1%}"
        )

    async def test_yargitay_decision_type_extraction(
        self,
        yargitay_ground_truth,
        validator,
    ):
        """
        Test decision type extraction accuracy.

        Verifies that "bozma", "onanma", etc. are correctly extracted.
        """
        # Filter cases with expected decision type
        test_cases = [
            gt for gt in yargitay_ground_truth
            if gt.expected_decision_type is not None
        ]

        if not test_cases:
            pytest.skip("No test cases with expected decision type")

        from backend.parsers.adapters.adapter_factory import get_factory
        factory = get_factory()
        adapter = factory.create("yargitay")

        results = validator.validate_adapter(
            adapter,
            test_cases,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        # Check semantic score (includes decision type)
        avg_semantic_score = sum(r.semantic_score for r in results.results) / len(results.results)
        assert avg_semantic_score >= 0.90, (
            f"Yargıtay decision type extraction score {avg_semantic_score:.1%} < 90%"
        )


# =============================================================================
# TEST SUITE - DANISTAY
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
class TestDanistayAdapter:
    """
    Test suite for Danıştay adapter.

    Harvey/Legora %100: 60 administrative law decisions with topic classification.
    """

    async def test_danistay_accuracy(
        self,
        danistay_ground_truth,
        validator,
    ):
        """
        Test Danıştay adapter overall accuracy.

        Requirement: %99 accuracy across all 60 test decisions.
        """
        from backend.parsers.adapters.adapter_factory import get_factory

        factory = get_factory()
        adapter = factory.create("danistay")

        results = validator.validate_adapter(
            adapter,
            danistay_ground_truth,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        report = validator.generate_report(results, output_format="text")
        print("\n" + report)

        assert results.accuracy >= MIN_ACCURACY_THRESHOLD, (
            f"Danıştay accuracy {results.accuracy:.1%} < {MIN_ACCURACY_THRESHOLD:.1%}"
        )

    async def test_danistay_topic_classification(
        self,
        danistay_ground_truth,
        validator,
    ):
        """
        Test topic classification accuracy (Harvey/Westlaw %98 parity).

        Verifies that topics (vergi, imar, cevre, etc.) are correctly identified.
        """
        # Filter cases with expected topics
        test_cases = [
            gt for gt in danistay_ground_truth
            if gt.expected_topics
        ]

        if not test_cases:
            pytest.skip("No test cases with expected topics")

        from backend.parsers.adapters.adapter_factory import get_factory
        factory = get_factory()
        adapter = factory.create("danistay")

        results = validator.validate_adapter(
            adapter,
            test_cases,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        # Check semantic score (includes topics)
        avg_semantic_score = sum(r.semantic_score for r in results.results) / len(results.results)
        assert avg_semantic_score >= 0.98, (
            f"Danıştay topic classification score {avg_semantic_score:.1%} < 98%"
        )


# =============================================================================
# TEST SUITE - AYM
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
class TestAYMAdapter:
    """
    Test suite for AYM adapter.

    Harvey/Legora %100: 60 constitutional decisions with ECHR violation tagging.
    """

    async def test_aym_accuracy(
        self,
        aym_ground_truth,
        validator,
    ):
        """
        Test AYM adapter overall accuracy.

        Requirement: %99 accuracy across all 60 test decisions.
        """
        from backend.parsers.adapters.adapter_factory import get_factory

        factory = get_factory()
        adapter = factory.create("aym")

        results = validator.validate_adapter(
            adapter,
            aym_ground_truth,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        report = validator.generate_report(results, output_format="text")
        print("\n" + report)

        assert results.accuracy >= MIN_ACCURACY_THRESHOLD, (
            f"AYM accuracy {results.accuracy:.1%} < {MIN_ACCURACY_THRESHOLD:.1%}"
        )

    async def test_aym_violation_tagging(
        self,
        aym_ground_truth,
        validator,
    ):
        """
        Test ECHR violation tagging accuracy (Westlaw %98 parity).

        Verifies that ECHR violations (ECHR_10, ECHR_6, etc.) are correctly identified.
        """
        # Filter cases with expected violations
        test_cases = [
            gt for gt in aym_ground_truth
            if gt.expected_violations
        ]

        if not test_cases:
            pytest.skip("No test cases with expected violations")

        from backend.parsers.adapters.adapter_factory import get_factory
        factory = get_factory()
        adapter = factory.create("aym")

        results = validator.validate_adapter(
            adapter,
            test_cases,
            max_documents=MAX_DOCUMENTS_PER_ADAPTER,
        )

        # Check semantic score (includes violations)
        avg_semantic_score = sum(r.semantic_score for r in results.results) / len(results.results)
        assert avg_semantic_score >= 0.98, (
            f"AYM violation tagging score {avg_semantic_score:.1%} < 98%"
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestAllAdapters:
    """
    Integration tests for all adapters.

    Harvey/Legora %100: System-wide accuracy guarantee.
    """

    async def test_overall_system_accuracy(
        self,
        golden_test_set,
        validator,
    ):
        """
        Test overall system accuracy across all 300 documents.

        Requirement: %99 accuracy across entire golden test set.
        """
        from backend.parsers.adapters.adapter_factory import get_factory

        factory = get_factory()
        all_results = []

        # Test all adapters
        for adapter_name in ["resmi_gazete", "mevzuat_gov", "yargitay", "danistay", "aym"]:
            ground_truth = golden_test_set.get_adapter_ground_truth(adapter_name)
            if not ground_truth:
                continue

            adapter = factory.create(adapter_name)
            results = validator.validate_adapter(
                adapter,
                ground_truth,
                max_documents=MAX_DOCUMENTS_PER_ADAPTER,
            )

            all_results.append(results)

        # Compute overall accuracy
        total_tests = sum(r.total_tests for r in all_results)
        total_passed = sum(r.passed_tests for r in all_results)
        overall_accuracy = total_passed / total_tests if total_tests > 0 else 0.0

        # Print summary
        print("\n" + "=" * 80)
        print("OVERALL SYSTEM ACCURACY")
        print("=" * 80)
        for results in all_results:
            print(f"{results.adapter_name:20s}: {results.accuracy:.1%} ({results.passed_tests}/{results.total_tests})")
        print("-" * 80)
        print(f"{'TOTAL':20s}: {overall_accuracy:.1%} ({total_passed}/{total_tests})")
        print("=" * 80)

        assert overall_accuracy >= MIN_ACCURACY_THRESHOLD, (
            f"Overall system accuracy {overall_accuracy:.1%} < {MIN_ACCURACY_THRESHOLD:.1%}\n"
            f"Passed: {total_passed}/{total_tests}"
        )


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.performance
class TestPerformance:
    """
    Performance benchmarking tests.

    Ensures parsers meet performance SLAs.
    """

    async def test_parse_time_sla(
        self,
        golden_test_set,
        validator,
    ):
        """
        Test that all parsers meet parse time SLA.

        Requirement: < 5000ms per document (99th percentile).
        """
        from backend.parsers.adapters.adapter_factory import get_factory

        factory = get_factory()
        max_parse_time_sla = 5000  # ms

        for adapter_name in ["resmi_gazete", "mevzuat_gov", "yargitay", "danistay", "aym"]:
            ground_truth = golden_test_set.get_adapter_ground_truth(adapter_name)
            if not ground_truth:
                continue

            adapter = factory.create(adapter_name)
            results = validator.validate_adapter(
                adapter,
                ground_truth[:10],  # Test first 10 only for performance
            )

            assert results.max_parse_time_ms <= max_parse_time_sla, (
                f"{adapter_name} max parse time {results.max_parse_time_ms}ms > {max_parse_time_sla}ms SLA"
            )


# =============================================================================
# MARK CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
