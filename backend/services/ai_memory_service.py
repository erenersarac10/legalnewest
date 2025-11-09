"""AI Memory Service - Harvey/Legora CTO-Level
Long-term memory and context management for Turkish Legal AI"""
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AIMemoryService:
    """Manage AI conversation memory and context"""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.sessions: Dict[str, List[Dict]] = {}
        self.user_preferences: Dict[str, Dict] = {}
    
    def store_interaction(
        self,
        session_id: str,
        user_msg: str,
        ai_response: str,
        metadata: Optional[Dict] = None
    ):
        """Store interaction in memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_msg,
            'ai_response': ai_response,
            'metadata': metadata or {}
        }
        
        self.sessions[session_id].append(interaction)
        
        # Limit history
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
        
        logger.debug(f"Stored interaction for session {session_id}")
    
    def get_context(self, session_id: str, last_n: int = 5) -> str:
        """Get recent context for session"""
        if session_id not in self.sessions:
            return ""
        
        recent = self.sessions[session_id][-last_n:]
        context_parts = []
        
        for interaction in recent:
            context_parts.append(f"User: {interaction['user_message']}")
            context_parts.append(f"AI: {interaction['ai_response']}")
        
        return "\n".join(context_parts)
    
    def store_user_preference(self, user_id: str, key: str, value: any):
        """Store user preference"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id][key] = value
        logger.info(f"Stored preference {key} for user {user_id}")
    
    def get_user_preference(self, user_id: str, key: str, default=None):
        """Get user preference"""
        return self.user_preferences.get(user_id, {}).get(key, default)
    
    def clear_session(self, session_id: str):
        """Clear session memory"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")

__all__ = ['AIMemoryService']
