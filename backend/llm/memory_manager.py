"""LLM Memory Manager - Harvey/Legora CTO-Level
Conversation memory and context management"""
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    user_message: str
    assistant_message: str
    timestamp: datetime
    metadata: Dict = None

class MemoryManager:
    """Manage conversation memory"""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversations: Dict[str, deque] = {}
    
    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str, **metadata):
        """Add conversation turn"""
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_turns)
        
        turn = ConversationTurn(
            user_message=user_msg,
            assistant_message=assistant_msg,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.conversations[session_id].append(turn)
    
    def get_history(self, session_id: str, last_n: Optional[int] = None) -> List[ConversationTurn]:
        """Get conversation history"""
        if session_id not in self.conversations:
            return []
        
        history = list(self.conversations[session_id])
        if last_n:
            return history[-last_n:]
        return history
    
    def clear(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def summarize_history(self, session_id: str) -> str:
        """Summarize conversation history"""
        history = self.get_history(session_id)
        if not history:
            return ""
        
        summary_parts = []
        for turn in history:
            summary_parts.append(f"User: {turn.user_message[:100]}...")
            summary_parts.append(f"Assistant: {turn.assistant_message[:100]}...")
        
        return "\n".join(summary_parts)

__all__ = ['MemoryManager', 'ConversationTurn']
