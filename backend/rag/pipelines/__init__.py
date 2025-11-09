"""RAG Pipelines Module - Harvey/Legora CTO-Level Production-Grade
Complete end-to-end RAG pipeline orchestration for Turkish legal AI

Available Pipelines:
- BasePipeline: Abstract base class for all pipelines
- QAPipeline: Question-answering with Turkish legal Q&A
- AnalysisPipeline: Document analysis, risk assessment, compliance
- ChatPipeline: Conversational interface with memory management

Each pipeline orchestrates:
1. Preprocessing (query understanding, entity extraction)
2. Retrieval (hybrid search with BM25 + vector)
3. Reranking (multi-stage reranking for quality)
4. Generation (LLM-based answer generation)
5. Postprocessing (citation extraction, formatting)
"""

from .base import (
    BasePipeline,
    PipelineProtocol,
    PipelineConfig,
    PipelineContext,
    PipelineResult,
    PipelineStageResult,
    Citation,
    PipelineStage,
    PipelineStatus
)

from .qa_pipeline import (
    QAPipeline,
    LegalQuery,
    AnswerCandidate
)

from .analysis_pipeline import (
    AnalysisPipeline,
    AnalysisResult,
    ContractClause,
    PrecedentCase,
    RiskAssessment,
    ComplianceCheck
)

from .chat_pipeline import (
    ChatPipeline,
    ChatConfig,
    ConversationMessage,
    ConversationTurn,
    ConversationMemory
)

__all__ = [
    # Base pipeline
    'BasePipeline',
    'PipelineProtocol',
    'PipelineConfig',
    'PipelineContext',
    'PipelineResult',
    'PipelineStageResult',
    'Citation',
    'PipelineStage',
    'PipelineStatus',

    # QA pipeline
    'QAPipeline',
    'LegalQuery',
    'AnswerCandidate',

    # Analysis pipeline
    'AnalysisPipeline',
    'AnalysisResult',
    'ContractClause',
    'PrecedentCase',
    'RiskAssessment',
    'ComplianceCheck',

    # Chat pipeline
    'ChatPipeline',
    'ChatConfig',
    'ConversationMessage',
    'ConversationTurn',
    'ConversationMemory'
]
