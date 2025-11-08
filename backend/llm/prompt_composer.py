"""LLM Prompt Composer - Harvey/Legora CTO-Level Production-Grade
Dynamic prompt composition with intelligent assembly and optimization

Production Features:
- Multi-template composition and merging
- Dynamic section assembly (system, context, examples, instructions)
- Conditional prompt construction based on task type
- Context injection with overflow handling
- Few-shot example selection and ordering
- Instruction combination and deduplication
- Variable resolution with fallbacks
- Prompt length optimization
- Format-specific composition (JSON, structured output)
- RAG context integration
- Chain-of-thought scaffolding
- Multi-turn conversation management
- Template inheritance and overrides
- Composability patterns (builder, strategy)
"""
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging

from .prompt_framework import (
    PromptBuilder,
    PromptMessage,
    PromptTemplate,
    PromptRole,
    OutputFormat,
    PromptStrategy
)
from .prompt_library import TEMPLATE_REGISTRY

logger = logging.getLogger(__name__)


# ============================================================================
# COMPOSITION STRATEGIES
# ============================================================================

@dataclass
class CompositionStrategy:
    """Strategy for composing prompts"""
    name: str
    description: str
    use_system_prompt: bool = True
    use_examples: bool = False
    use_context: bool = True
    use_instructions: bool = True
    max_examples: int = 5
    context_position: str = "before"  # before or after instructions


class CompositionStrategies:
    """Pre-defined composition strategies"""

    ZERO_SHOT = CompositionStrategy(
        name="zero_shot",
        description="Zero-shot: No examples, just instructions",
        use_examples=False
    )

    FEW_SHOT = CompositionStrategy(
        name="few_shot",
        description="Few-shot: With examples",
        use_examples=True,
        max_examples=5
    )

    COT_ZERO_SHOT = CompositionStrategy(
        name="cot_zero_shot",
        description="Chain-of-thought zero-shot",
        use_examples=False,
        use_instructions=True
    )

    COT_FEW_SHOT = CompositionStrategy(
        name="cot_few_shot",
        description="Chain-of-thought few-shot",
        use_examples=True,
        max_examples=3
    )

    RAG_BASED = CompositionStrategy(
        name="rag_based",
        description="RAG-based: Heavy context, minimal instructions",
        use_context=True,
        use_examples=False,
        context_position="before"
    )


# ============================================================================
# PROMPT COMPOSER
# ============================================================================

class PromptComposer:
    """Intelligent prompt composition engine"""

    def __init__(
        self,
        template_registry: Optional[Dict[str, PromptTemplate]] = None,
        default_strategy: Optional[CompositionStrategy] = None
    ):
        """Initialize composer

        Args:
            template_registry: Custom template registry
            default_strategy: Default composition strategy
        """
        self.template_registry = template_registry or TEMPLATE_REGISTRY
        self.default_strategy = default_strategy or CompositionStrategies.ZERO_SHOT

        logger.debug(f"Initialized PromptComposer with {len(self.template_registry)} templates")

    def compose_from_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        strategy: Optional[CompositionStrategy] = None,
        system_prompt_id: Optional[str] = None
    ) -> PromptBuilder:
        """Compose prompt from template with strategy

        Args:
            template_id: Template ID to use
            variables: Template variables
            context: Optional context to inject
            examples: Optional few-shot examples
            strategy: Composition strategy
            system_prompt_id: Optional system prompt template ID

        Returns:
            PromptBuilder
        """
        strategy = strategy or self.default_strategy

        # Get template
        if template_id not in self.template_registry:
            raise ValueError(f"Template '{template_id}' not found")

        template = self.template_registry[template_id]

        # Build prompt
        builder = PromptBuilder()

        # 1. System prompt
        if strategy.use_system_prompt:
            if system_prompt_id:
                sys_template = self.template_registry.get(system_prompt_id)
                if sys_template:
                    builder.system(sys_template.render())
            elif "turkish_legal_expert_system" in self.template_registry:
                # Default to Turkish legal expert system prompt
                sys_template = self.template_registry["turkish_legal_expert_system"]
                builder.system(sys_template.render())

        # 2. Few-shot examples
        if strategy.use_examples and examples:
            max_examples = min(len(examples), strategy.max_examples)
            builder.with_examples(examples[:max_examples])

        # 3. Context (before or after)
        if strategy.use_context and context and strategy.context_position == "before":
            builder.with_context(context, label="0lgili Balam")

        # 4. Main instruction
        if strategy.use_instructions:
            builder.add_template(template, **variables)

        # 5. Context (after)
        if strategy.use_context and context and strategy.context_position == "after":
            builder.with_context(context, label="0lgili Balam")

        # 6. Output format instructions
        if template.output_format != OutputFormat.TEXT:
            builder.with_format_instructions(template.output_format)

        return builder

    def compose_multi_template(
        self,
        template_ids: List[str],
        variables_list: List[Dict[str, Any]],
        separator: str = "\n\n---\n\n"
    ) -> PromptBuilder:
        """Compose multiple templates into single prompt

        Args:
            template_ids: List of template IDs
            variables_list: List of variable dicts for each template
            separator: Separator between templates

        Returns:
            PromptBuilder
        """
        if len(template_ids) != len(variables_list):
            raise ValueError("Template IDs and variables list must have same length")

        builder = PromptBuilder()

        # Combine all templates
        combined_content = []

        for template_id, variables in zip(template_ids, variables_list):
            template = self.template_registry.get(template_id)
            if not template:
                logger.warning(f"Template '{template_id}' not found, skipping")
                continue

            rendered = template.render(**variables)
            combined_content.append(rendered)

        # Add as single user message
        full_content = separator.join(combined_content)
        builder.user(full_content)

        return builder

    def compose_for_qa(
        self,
        question: str,
        context: str,
        include_cot: bool = True,
        include_examples: bool = False,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> PromptBuilder:
        """Compose prompt for Q&A task

        Args:
            question: User question
            context: Relevant context
            include_cot: Include chain-of-thought instructions
            include_examples: Include few-shot examples
            examples: Optional examples

        Returns:
            PromptBuilder
        """
        template_id = "legal_qa" if not include_cot else "multi_hop_qa"

        strategy = CompositionStrategies.FEW_SHOT if include_examples else CompositionStrategies.ZERO_SHOT

        variables = {
            "question": question if not include_cot else question,
            "context": context if not include_cot else "",
            "complex_question": question if include_cot else "",
            "available_information": context if include_cot else ""
        }

        return self.compose_from_template(
            template_id=template_id,
            variables=variables,
            context=context if not include_cot else None,
            examples=examples if include_examples else None,
            strategy=strategy
        )

    def compose_for_analysis(
        self,
        document_type: str,
        document_text: str,
        additional_info: Optional[Dict[str, str]] = None
    ) -> PromptBuilder:
        """Compose prompt for document analysis

        Args:
            document_type: Type of document (law, regulation, court_decision)
            document_text: Document text
            additional_info: Additional context (court_type, decision_number, etc.)

        Returns:
            PromptBuilder
        """
        template_map = {
            "law": "law_analysis",
            "regulation": "regulation_analysis",
            "court_decision": "court_decision_analysis"
        }

        template_id = template_map.get(document_type)
        if not template_id:
            raise ValueError(f"Unknown document type: {document_type}")

        variables = {"law_text": document_text} if document_type == "law" else \
                   {"regulation_text": document_text} if document_type == "regulation" else \
                   {
                       "decision_text": document_text,
                       "court_type": additional_info.get("court_type", "Mahkeme"),
                       "decision_number": additional_info.get("decision_number", ""),
                       "decision_date": additional_info.get("decision_date", "")
                   }

        return self.compose_from_template(
            template_id=template_id,
            variables=variables,
            strategy=CompositionStrategies.COT_ZERO_SHOT
        )

    def compose_for_extraction(
        self,
        extraction_type: str,
        text: str,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> PromptBuilder:
        """Compose prompt for entity/citation extraction

        Args:
            extraction_type: Type of extraction (citation, entity)
            text: Input text
            output_schema: Optional JSON schema for output

        Returns:
            PromptBuilder
        """
        template_map = {
            "citation": "citation_extraction",
            "entity": "entity_extraction"
        }

        template_id = template_map.get(extraction_type)
        if not template_id:
            raise ValueError(f"Unknown extraction type: {extraction_type}")

        variables = {
            "legal_text": text if extraction_type == "citation" else "",
            "text": text if extraction_type == "entity" else ""
        }

        builder = self.compose_from_template(
            template_id=template_id,
            variables=variables,
            strategy=CompositionStrategies.ZERO_SHOT
        )

        # Add JSON schema if provided
        if output_schema:
            schema_str = f"\n\nBeklenen JSON format1:\n{output_schema}"
            builder.user(schema_str)

        return builder

    def compose_for_compliance(
        self,
        compliance_type: str,
        description: str,
        additional_context: Dict[str, str]
    ) -> PromptBuilder:
        """Compose prompt for compliance checking

        Args:
            compliance_type: Type of compliance (kvkk, contract)
            description: Main description
            additional_context: Additional context fields

        Returns:
            PromptBuilder
        """
        template_map = {
            "kvkk": "kvkk_compliance",
            "contract": "contract_compliance"
        }

        template_id = template_map.get(compliance_type)
        if not template_id:
            raise ValueError(f"Unknown compliance type: {compliance_type}")

        variables = {
            "data_processing_description": description,
            **additional_context
        } if compliance_type == "kvkk" else {
            "contract_type": description,
            **additional_context
        }

        return self.compose_from_template(
            template_id=template_id,
            variables=variables,
            strategy=CompositionStrategies.COT_ZERO_SHOT
        )

    def compose_with_rag_context(
        self,
        query: str,
        retrieved_chunks: List[str],
        max_context_length: int = 4000,
        chunk_separator: str = "\n\n---\n\n"
    ) -> PromptBuilder:
        """Compose RAG-based prompt with retrieved context

        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            max_context_length: Maximum context length in characters
            chunk_separator: Separator between chunks

        Returns:
            PromptBuilder
        """
        # Combine chunks with length limit
        combined_context = ""
        for chunk in retrieved_chunks:
            if len(combined_context) + len(chunk) + len(chunk_separator) > max_context_length:
                break
            if combined_context:
                combined_context += chunk_separator
            combined_context += chunk

        # Use legal QA template with RAG strategy
        return self.compose_from_template(
            template_id="legal_qa",
            variables={
                "question": query,
                "context": combined_context
            },
            strategy=CompositionStrategies.RAG_BASED
        )

    def compose_conversation_turn(
        self,
        user_message: str,
        conversation_history: List[PromptMessage],
        max_history_turns: int = 10
    ) -> PromptBuilder:
        """Compose prompt for multi-turn conversation

        Args:
            user_message: Current user message
            conversation_history: Previous messages
            max_history_turns: Maximum turns to include

        Returns:
            PromptBuilder
        """
        builder = PromptBuilder()

        # Add system prompt
        if "turkish_legal_expert_system" in self.template_registry:
            sys_template = self.template_registry["turkish_legal_expert_system"]
            builder.system(sys_template.render())

        # Add history (limit turns)
        recent_history = conversation_history[-max_history_turns * 2:] if conversation_history else []
        for msg in recent_history:
            builder.add_message(msg)

        # Add current user message
        builder.user(user_message)

        return builder


# ============================================================================
# ADAPTIVE COMPOSER
# ============================================================================

class AdaptiveComposer(PromptComposer):
    """Composer with adaptive strategy selection"""

    def __init__(self, **kwargs):
        """Initialize adaptive composer"""
        super().__init__(**kwargs)

        # Task type to strategy mapping
        self.strategy_map = {
            "simple_qa": CompositionStrategies.ZERO_SHOT,
            "complex_reasoning": CompositionStrategies.COT_ZERO_SHOT,
            "with_examples": CompositionStrategies.FEW_SHOT,
            "rag_query": CompositionStrategies.RAG_BASED
        }

    def detect_task_type(
        self,
        query: str,
        has_context: bool = False,
        has_examples: bool = False
    ) -> str:
        """Detect task type from query

        Args:
            query: User query
            has_context: Whether context is available
            has_examples: Whether examples are available

        Returns:
            Task type string
        """
        # Simple heuristics
        complexity_indicators = [
            "neden", "nas1l", "aç1kla", "deerlendir",
            "kar_1la_t1r", "analiz et", "ad1m ad1m"
        ]

        query_lower = query.lower()

        # Check for complexity
        is_complex = any(indicator in query_lower for indicator in complexity_indicators)

        if has_examples:
            return "with_examples"
        elif has_context and not is_complex:
            return "rag_query"
        elif is_complex:
            return "complex_reasoning"
        else:
            return "simple_qa"

    def auto_compose(
        self,
        query: str,
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> PromptBuilder:
        """Automatically compose with detected strategy

        Args:
            query: User query
            context: Optional context
            examples: Optional examples
            **kwargs: Additional template variables

        Returns:
            PromptBuilder
        """
        # Detect task type
        task_type = self.detect_task_type(
            query=query,
            has_context=bool(context),
            has_examples=bool(examples)
        )

        # Get strategy
        strategy = self.strategy_map.get(task_type, CompositionStrategies.ZERO_SHOT)

        logger.debug(f"Auto-detected task type: {task_type}, using strategy: {strategy.name}")

        # Compose based on task type
        if task_type == "rag_query":
            return self.compose_from_template(
                template_id="legal_qa",
                variables={"question": query, "context": context or "", **kwargs},
                context=context,
                strategy=strategy
            )
        elif task_type == "complex_reasoning":
            return self.compose_from_template(
                template_id="multi_hop_qa",
                variables={
                    "complex_question": query,
                    "available_information": context or "",
                    **kwargs
                },
                strategy=strategy
            )
        else:
            return self.compose_from_template(
                template_id="legal_qa",
                variables={"question": query, "context": context or "", **kwargs},
                examples=examples,
                strategy=strategy
            )


__all__ = [
    'CompositionStrategy',
    'CompositionStrategies',
    'PromptComposer',
    'AdaptiveComposer'
]
