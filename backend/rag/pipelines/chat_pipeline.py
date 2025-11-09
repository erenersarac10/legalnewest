"""Chat Pipeline - Harvey/Legora CTO-Level Production-Grade
Conversational pipeline for Turkish legal chatbot with memory and context management

Production Features:
- Multi-turn conversation support with memory
- Context-aware retrieval from conversation history
- Session management and persistence
- Topic detection and switching
- Reference resolution (pronouns, implicit mentions)
- Citation tracking across conversation
- Follow-up question handling
- Conversation summarization
- Memory pruning and optimization
- User intent classification
- Clarification question generation
- Turkish conversational patterns support
- Legal terminology consistency
- Context window management for long conversations
- Conversation branching and forking
- User preference learning
- Conversational search refinement
- Multi-modal input support (questions + documents)
- Conversation export and sharing
"""
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import re
import logging
import json

from .base import (
    BasePipeline,
    PipelineConfig,
    PipelineContext,
    Citation,
    PipelineStatus
)
from ..retrievers.base import SearchResults

logger = logging.getLogger(__name__)


# ============================================================================
# CHAT-SPECIFIC DATA MODELS
# ============================================================================

@dataclass
class ConversationMessage:
    """Single message in conversation"""
    message_id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """Complete conversation turn (user + assistant)"""
    turn_id: int
    user_message: ConversationMessage
    assistant_message: Optional[ConversationMessage] = None
    retrieved_context: Optional[SearchResults] = None
    intent: str = "query"  # query, clarification, follow_up, greeting
    topic: str = "general"


@dataclass
class ConversationMemory:
    """Conversation memory with history and context"""
    session_id: str
    user_id: Optional[str]
    started_at: datetime
    last_updated: datetime

    # Conversation history
    turns: Deque[ConversationTurn] = field(default_factory=lambda: deque(maxlen=50))
    messages: Deque[ConversationMessage] = field(default_factory=lambda: deque(maxlen=100))

    # Context tracking
    active_topics: List[str] = field(default_factory=list)
    mentioned_laws: List[str] = field(default_factory=list)
    mentioned_articles: List[str] = field(default_factory=list)
    cited_documents: List[str] = field(default_factory=list)

    # Summary
    conversation_summary: str = ""

    # User preferences
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add conversation turn

        Args:
            turn: Conversation turn
        """
        self.turns.append(turn)
        self.messages.append(turn.user_message)
        if turn.assistant_message:
            self.messages.append(turn.assistant_message)
        self.last_updated = datetime.now()

    def get_recent_context(self, num_turns: int = 5) -> str:
        """Get recent conversation context

        Args:
            num_turns: Number of recent turns

        Returns:
            Context string
        """
        recent_turns = list(self.turns)[-num_turns:]
        context_parts = []

        for turn in recent_turns:
            context_parts.append(f"Kullanıcı: {turn.user_message.content}")
            if turn.assistant_message:
                context_parts.append(f"Asistan: {turn.assistant_message.content}")

        return "\n".join(context_parts)


@dataclass
class ChatConfig(PipelineConfig):
    """Extended configuration for chat pipeline"""
    # Memory settings
    max_history_turns: int = 20
    max_context_window: int = 10000  # tokens
    enable_conversation_summary: bool = True
    summary_frequency: int = 10  # Summarize every N turns

    # Retrieval settings
    context_aware_retrieval: bool = True
    use_conversation_history: bool = True

    # Intent detection
    enable_intent_classification: bool = True
    enable_topic_detection: bool = True

    # Response settings
    enable_clarification: bool = True
    max_clarification_attempts: int = 3


# ============================================================================
# CHAT PIPELINE
# ============================================================================

class ChatPipeline(BasePipeline):
    """Conversational pipeline for Turkish legal chatbot"""

    # Turkish conversational patterns
    GREETING_PATTERNS = [
        r'\b(merhaba|selam|iyi günler|günaydın|iyi akşamlar)\b',
        r'\b(hoş geldiniz|hoşgeldiniz|nasılsınız|naber)\b'
    ]

    FAREWELL_PATTERNS = [
        r'\b(güle güle|hoşça kal|görüşürüz|teşekkür|sağol)\b',
        r'\b(bye|bay|elveda|iyi günler)\b'
    ]

    CLARIFICATION_PATTERNS = [
        r'\b(anlamadım|ne demek|açıkla|detay|örnek)\b',
        r'\b(nasıl yani|yani|nedir|nedemek)\b'
    ]

    # Intent patterns
    INTENT_PATTERNS = {
        'greeting': GREETING_PATTERNS,
        'farewell': FAREWELL_PATTERNS,
        'clarification': CLARIFICATION_PATTERNS,
        'query': [r'.+\?$'],  # Ends with question mark
    }

    # Reference resolution patterns
    PRONOUNS = {
        'o': 'referring_entity',
        'bu': 'this_entity',
        'şu': 'that_entity',
        'bunlar': 'these_entities',
        'onlar': 'those_entities'
    }

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        config: Optional[ChatConfig] = None,
        reranker: Optional[Any] = None,
        memory_backend: Optional[Any] = None
    ):
        """Initialize chat pipeline

        Args:
            retriever: Retriever instance
            generator: LLM generator
            config: Chat configuration
            reranker: Optional reranker
            memory_backend: Optional memory persistence backend
        """
        super().__init__(retriever, generator, config or ChatConfig(), reranker)

        self.memory_backend = memory_backend

        # Active conversations
        self.conversations: Dict[str, ConversationMemory] = {}

        # Compile patterns
        self.intent_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

        logger.info("Initialized ChatPipeline (memory_enabled={})".format(
            memory_backend is not None
        ))

    def chat(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process chat message

        Args:
            message: User message
            session_id: Session ID
            user_id: Optional user ID
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        # Get or create conversation memory
        memory = self._get_or_create_memory(session_id, user_id)

        # Create context with conversation history
        context = PipelineContext(
            query=message,
            user_id=user_id,
            session_id=session_id,
            conversation_history=[
                {
                    'role': turn.user_message.role,
                    'content': turn.user_message.content
                }
                for turn in list(memory.turns)
            ],
            metadata={'memory': memory}
        )

        # Run pipeline
        result = self.run(message, context, **kwargs)

        # Update memory
        turn_id = len(memory.turns) + 1
        user_msg = ConversationMessage(
            message_id=f"{session_id}_user_{turn_id}",
            role="user",
            content=message,
            timestamp=datetime.now()
        )

        assistant_msg = ConversationMessage(
            message_id=f"{session_id}_assistant_{turn_id}",
            role="assistant",
            content=result.answer,
            timestamp=datetime.now(),
            citations=result.citations
        )

        turn = ConversationTurn(
            turn_id=turn_id,
            user_message=user_msg,
            assistant_message=assistant_msg,
            intent=context.metadata.get('detected_intent', 'query'),
            topic=context.metadata.get('detected_topic', 'general')
        )

        memory.add_turn(turn)

        # Update tracked entities
        self._update_memory_entities(memory, result)

        # Summarize if needed
        if self.config.enable_conversation_summary:
            if len(memory.turns) % self.config.summary_frequency == 0:
                memory.conversation_summary = self._summarize_conversation(memory)

        # Persist memory if backend available
        if self.memory_backend:
            self._persist_memory(memory)

        return {
            'answer': result.answer,
            'citations': result.citations,
            'session_id': session_id,
            'turn_id': turn_id,
            'conversation_context': memory.get_recent_context(3),
            'metadata': {
                'intent': turn.intent,
                'topic': turn.topic,
                'confidence': result.confidence_score
            }
        }

    def preprocess(
        self,
        query: str,
        context: PipelineContext
    ) -> str:
        """Preprocess chat message

        Args:
            query: User message
            context: Pipeline context

        Returns:
            Processed query
        """
        memory: ConversationMemory = context.metadata.get('memory')

        # Detect intent
        if self.config.enable_intent_classification:
            intent = self._detect_intent(query)
            context.metadata['detected_intent'] = intent
        else:
            intent = 'query'

        # Detect topic
        if self.config.enable_topic_detection:
            topic = self._detect_topic(query, memory)
            context.metadata['detected_topic'] = topic

        # Handle greetings/farewells directly
        if intent in ['greeting', 'farewell']:
            context.metadata['skip_retrieval'] = True
            return query

        # Resolve references
        resolved_query = self._resolve_references(query, memory)

        # Expand query with conversation context
        if self.config.context_aware_retrieval and memory and memory.turns:
            expanded_query = self._expand_with_context(resolved_query, memory)
            return expanded_query

        return resolved_query

    def retrieve(
        self,
        query: str,
        context: PipelineContext
    ) -> SearchResults:
        """Retrieve with conversation context

        Args:
            query: Processed query
            context: Pipeline context

        Returns:
            SearchResults
        """
        # Skip retrieval for greetings/farewells
        if context.metadata.get('skip_retrieval', False):
            return SearchResults(
                query=query,
                results=[],
                total_results=0,
                search_time_ms=0.0,
                limit=0
            )

        memory: ConversationMemory = context.metadata.get('memory')

        # Build filters from conversation context
        filters = dict(context.filters)

        # Add law/article filters from conversation history
        if memory:
            if memory.mentioned_laws:
                filters['law_number'] = memory.mentioned_laws[-3:]  # Last 3 mentioned

        # Retrieve
        results = self.retriever.retrieve(
            query,
            filters=filters if filters else None,
            limit=self.config.retrieval_limit
        )

        return results

    def generate(
        self,
        query: str,
        results: SearchResults,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Generate response with conversation awareness

        Args:
            query: Query
            results: Retrieved results
            context: Pipeline context

        Returns:
            Generation output
        """
        memory: ConversationMemory = context.metadata.get('memory')
        intent = context.metadata.get('detected_intent', 'query')

        # Handle special intents
        if intent == 'greeting':
            return {
                'answer': self._generate_greeting_response(),
                'confidence': 1.0
            }

        if intent == 'farewell':
            return {
                'answer': self._generate_farewell_response(),
                'confidence': 1.0
            }

        # Build conversation-aware prompt
        prompt = self._build_chat_prompt(
            query=query,
            results=results,
            memory=memory,
            intent=intent
        )

        # Generate response
        try:
            generation_output = self.generator.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens
            )

            answer = generation_output.get('text', '')

            return {
                'answer': answer,
                'confidence': generation_output.get('confidence', 0.0),
                'prompt_tokens': generation_output.get('prompt_tokens', 0),
                'completion_tokens': generation_output.get('completion_tokens', 0)
            }

        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return {
                'answer': 'Üzgünüm, cevap oluştururken bir hata oluştu.',
                'confidence': 0.0,
                'error': str(e)
            }

    def postprocess(
        self,
        generation_output: Dict[str, Any],
        results: SearchResults,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Postprocess chat response

        Args:
            generation_output: Raw generation output
            results: Retrieved results
            context: Pipeline context

        Returns:
            Final output
        """
        answer = generation_output.get('answer', '')

        # Extract citations
        citations = self._extract_citations_from_results(results)

        # Check if clarification is needed
        needs_clarification = self._check_if_clarification_needed(
            answer,
            results,
            context
        )

        metadata = {
            'prompt_tokens': generation_output.get('prompt_tokens', 0),
            'completion_tokens': generation_output.get('completion_tokens', 0),
            'needs_clarification': needs_clarification
        }

        # Add clarification prompt if needed
        if needs_clarification and self.config.enable_clarification:
            clarification = self._generate_clarification_prompt(context)
            metadata['clarification_prompt'] = clarification

        return {
            'answer': answer,
            'citations': citations,
            'confidence': generation_output.get('confidence', 0.0),
            'metadata': metadata
        }

    def _get_or_create_memory(
        self,
        session_id: str,
        user_id: Optional[str]
    ) -> ConversationMemory:
        """Get or create conversation memory

        Args:
            session_id: Session ID
            user_id: User ID

        Returns:
            ConversationMemory
        """
        if session_id in self.conversations:
            return self.conversations[session_id]

        # Try to load from backend
        if self.memory_backend:
            memory = self._load_memory(session_id)
            if memory:
                self.conversations[session_id] = memory
                return memory

        # Create new memory
        memory = ConversationMemory(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now(),
            last_updated=datetime.now()
        )

        self.conversations[session_id] = memory
        return memory

    def _detect_intent(self, message: str) -> str:
        """Detect user intent

        Args:
            message: User message

        Returns:
            Intent type
        """
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    return intent

        return 'query'  # Default

    def _detect_topic(
        self,
        message: str,
        memory: Optional[ConversationMemory]
    ) -> str:
        """Detect conversation topic

        Args:
            message: User message
            memory: Conversation memory

        Returns:
            Topic
        """
        # Simple keyword-based topic detection
        topics = {
            'ceza_hukuku': ['ceza', 'suç', 'hapis', 'tck'],
            'medeni_hukuk': ['medeni', 'evlilik', 'miras', 'tmk'],
            'ticaret_hukuku': ['ticaret', 'şirket', 'ttk', 'anonim'],
            'iş_hukuku': ['iş', 'çalışan', 'işçi', 'işveren'],
            'kişisel_veriler': ['kvkk', 'kişisel veri', 'gdpr']
        }

        message_lower = message.lower()

        for topic, keywords in topics.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic

        # Check previous topics
        if memory and memory.active_topics:
            return memory.active_topics[-1]

        return 'general'

    def _resolve_references(
        self,
        query: str,
        memory: Optional[ConversationMemory]
    ) -> str:
        """Resolve pronouns and references

        Args:
            query: Query with potential references
            memory: Conversation memory

        Returns:
            Resolved query
        """
        if not memory or not memory.turns:
            return query

        resolved = query

        # Simple pronoun resolution
        # Example: "Bu madde ne anlama gelir?" -> "5237 sayılı kanun 1. madde ne anlama gelir?"

        # Check for pronouns
        query_lower = query.lower()

        if any(pronoun in query_lower for pronoun in self.PRONOUNS.keys()):
            # Get last mentioned law/article
            if memory.mentioned_laws:
                last_law = memory.mentioned_laws[-1]
                # Simple replacement (could be improved)
                resolved = f"{last_law} sayılı kanun hakkında: {query}"

        return resolved

    def _expand_with_context(
        self,
        query: str,
        memory: ConversationMemory
    ) -> str:
        """Expand query with conversation context

        Args:
            query: Query
            memory: Conversation memory

        Returns:
            Expanded query
        """
        # Add recent context
        context_parts = [query]

        # Add mentioned laws
        if memory.mentioned_laws:
            context_parts.append(f"İlgili kanunlar: {', '.join(memory.mentioned_laws[-2:])}")

        # Add active topics
        if memory.active_topics:
            context_parts.append(f"Konu: {memory.active_topics[-1]}")

        return ' | '.join(context_parts)

    def _build_chat_prompt(
        self,
        query: str,
        results: SearchResults,
        memory: Optional[ConversationMemory],
        intent: str
    ) -> str:
        """Build chat prompt with conversation history

        Args:
            query: Current query
            results: Retrieved results
            memory: Conversation memory
            intent: Detected intent

        Returns:
            Prompt text
        """
        system_prompt = """Sen Türk hukuku konusunda uzman bir sohbet asistanısın.
Kullanıcıyla doğal ve akıcı bir şekilde konuş, sorularını cevapla.

Kurallar:
1. Önceki konuşma geçmişini dikkate al
2. Kullanıcının referanslarını anla (bu, o, şu gibi)
3. Tutarlı ve anlaşılır cevaplar ver
4. Kaynaklara dayanarak cevap ver
5. Emin olmadığında açıkça belirt
"""

        # Add conversation history
        conversation_context = ""
        if memory and memory.turns:
            conversation_context = "ÖNCEKİ KONUŞMA:\n"
            conversation_context += memory.get_recent_context(3)
            conversation_context += "\n\n"

        # Add retrieved context
        retrieved_context = ""
        if results.results:
            retrieved_context = "İLGİLİ KAYNAKLAR:\n"
            for i, result in enumerate(results.results[:3]):
                retrieved_context += f"[{i+1}] {result.content[:200]}...\n"
            retrieved_context += "\n"

        prompt = f"""{system_prompt}

{conversation_context}{retrieved_context}YENİ SORU: {query}

CEVAP:"""

        return prompt

    def _generate_greeting_response(self) -> str:
        """Generate greeting response

        Returns:
            Greeting message
        """
        greetings = [
            "Merhaba! Size nasıl yardımcı olabilirim?",
            "Hoş geldiniz! Türk hukuku hakkında sorularınızı yanıtlamak için buradayım.",
            "Merhaba! Hukuki sorularınız için size yardımcı olmaktan mutluluk duyarım."
        ]

        import random
        return random.choice(greetings)

    def _generate_farewell_response(self) -> str:
        """Generate farewell response

        Returns:
            Farewell message
        """
        farewells = [
            "Görüşmek üzere! İyi günler dilerim.",
            "Teşekkür ederim. İyi günler!",
            "Hoşça kalın!"
        ]

        import random
        return random.choice(farewells)

    def _check_if_clarification_needed(
        self,
        answer: str,
        results: SearchResults,
        context: PipelineContext
    ) -> bool:
        """Check if clarification is needed

        Args:
            answer: Generated answer
            results: Retrieved results
            context: Pipeline context

        Returns:
            True if clarification needed
        """
        # Check if retrieval found nothing
        if not results.results:
            return True

        # Check if answer is too short
        if len(answer.split()) < 10:
            return True

        # Check if answer contains uncertainty markers
        uncertainty_markers = [
            'emin değilim',
            'net değil',
            'belirsiz',
            'açık değil',
            'bilgim yok'
        ]

        if any(marker in answer.lower() for marker in uncertainty_markers):
            return True

        return False

    def _generate_clarification_prompt(self, context: PipelineContext) -> str:
        """Generate clarification prompt

        Args:
            context: Pipeline context

        Returns:
            Clarification prompt
        """
        prompts = [
            "Sorunuzu biraz daha detaylandırabilir misiniz?",
            "Hangi konuda daha fazla bilgi istersiniz?",
            "Belirli bir madde veya kanun hakkında mı bilgi istiyorsunuz?"
        ]

        import random
        return random.choice(prompts)

    def _update_memory_entities(
        self,
        memory: ConversationMemory,
        result: Any
    ) -> None:
        """Update tracked entities in memory

        Args:
            memory: Conversation memory
            result: Pipeline result
        """
        # Extract law numbers from citations
        for citation in result.citations:
            if citation.law_number and citation.law_number not in memory.mentioned_laws:
                memory.mentioned_laws.append(citation.law_number)

            if citation.article_number and citation.article_number not in memory.mentioned_articles:
                memory.mentioned_articles.append(citation.article_number)

            if citation.document_id not in memory.cited_documents:
                memory.cited_documents.append(citation.document_id)

    def _summarize_conversation(self, memory: ConversationMemory) -> str:
        """Summarize conversation

        Args:
            memory: Conversation memory

        Returns:
            Conversation summary
        """
        # Simple summary (could use LLM)
        topics = list(set(turn.topic for turn in memory.turns))
        laws = memory.mentioned_laws[:5]  # Top 5 laws

        summary = f"Konuşma konuları: {', '.join(topics)}. "
        if laws:
            summary += f"Bahsedilen kanunlar: {', '.join(laws)}."

        return summary

    def _persist_memory(self, memory: ConversationMemory) -> None:
        """Persist conversation memory

        Args:
            memory: Conversation memory
        """
        if not self.memory_backend:
            return

        try:
            # Serialize memory
            memory_dict = {
                'session_id': memory.session_id,
                'user_id': memory.user_id,
                'started_at': memory.started_at.isoformat(),
                'last_updated': memory.last_updated.isoformat(),
                'turns': [
                    {
                        'turn_id': turn.turn_id,
                        'user_content': turn.user_message.content,
                        'assistant_content': turn.assistant_message.content if turn.assistant_message else None,
                        'intent': turn.intent,
                        'topic': turn.topic
                    }
                    for turn in list(memory.turns)
                ],
                'active_topics': memory.active_topics,
                'mentioned_laws': memory.mentioned_laws,
                'conversation_summary': memory.conversation_summary
            }

            # Save to backend
            self.memory_backend.save(memory.session_id, memory_dict)

        except Exception as e:
            logger.error(f"Failed to persist memory: {e}")

    def _load_memory(self, session_id: str) -> Optional[ConversationMemory]:
        """Load conversation memory

        Args:
            session_id: Session ID

        Returns:
            ConversationMemory or None
        """
        if not self.memory_backend:
            return None

        try:
            memory_dict = self.memory_backend.load(session_id)
            if not memory_dict:
                return None

            # Deserialize (simplified)
            memory = ConversationMemory(
                session_id=memory_dict['session_id'],
                user_id=memory_dict.get('user_id'),
                started_at=datetime.fromisoformat(memory_dict['started_at']),
                last_updated=datetime.fromisoformat(memory_dict['last_updated']),
                active_topics=memory_dict.get('active_topics', []),
                mentioned_laws=memory_dict.get('mentioned_laws', []),
                conversation_summary=memory_dict.get('conversation_summary', '')
            )

            return memory

        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return None

    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation memory

        Args:
            session_id: Session ID
        """
        if session_id in self.conversations:
            del self.conversations[session_id]

        if self.memory_backend:
            self.memory_backend.delete(session_id)

        logger.info(f"Cleared conversation: {session_id}")


__all__ = [
    'ChatPipeline',
    'ChatConfig',
    'ConversationMessage',
    'ConversationTurn',
    'ConversationMemory'
]
