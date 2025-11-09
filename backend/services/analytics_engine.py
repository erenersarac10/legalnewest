"""Analytics Engine - Harvey/Legora CTO-Level
Advanced analytics for Turkish legal AI system"""
from typing import Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """Process and analyze system metrics"""
    
    def __init__(self):
        self.metrics: List[Dict] = []
    
    def track_event(self, event_type: str, data: Dict):
        """Track analytics event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        }
        self.metrics.append(event)
        logger.debug(f"Tracked event: {event_type}")
    
    def get_usage_stats(self, days: int = 7) -> Dict:
        """Get usage statistics"""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        recent = [m for m in self.metrics if m['timestamp'] >= cutoff_str]
        
        stats = {
            'total_events': len(recent),
            'event_types': {},
            'daily_counts': {}
        }
        
        for metric in recent:
            event_type = metric['type']
            stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
            
            day = metric['timestamp'][:10]
            stats['daily_counts'][day] = stats['daily_counts'].get(day, 0) + 1
        
        return stats
    
    def get_top_queries(self, limit: int = 10) -> List[Dict]:
        """Get most common queries"""
        query_events = [m for m in self.metrics if m['type'] == 'query']
        
        query_counts = {}
        for event in query_events:
            query = event['data'].get('query', '')
            query_counts[query] = query_counts.get(query, 0) + 1
        
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'query': q, 'count': c}
            for q, c in sorted_queries[:limit]
        ]

__all__ = ['AnalyticsEngine']
