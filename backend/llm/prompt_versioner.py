"""LLM Prompt Versioner - Harvey/Legora CTO-Level
Version control and A/B testing for prompts"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptVersion:
    """Prompt version with metadata"""
    version_id: str
    template_id: str
    content: str
    created_at: datetime
    created_by: str
    metrics: Dict = None
    is_active: bool = True

class PromptVersioner:
    """Manage prompt versions and A/B tests"""
    
    def __init__(self):
        self.versions: Dict[str, List[PromptVersion]] = {}
    
    def create_version(self, template_id: str, content: str, created_by: str) -> PromptVersion:
        """Create new version"""
        version_num = len(self.versions.get(template_id, [])) + 1
        version = PromptVersion(
            version_id=f"{template_id}_v{version_num}",
            template_id=template_id,
            content=content,
            created_at=datetime.now(),
            created_by=created_by
        )
        
        if template_id not in self.versions:
            self.versions[template_id] = []
        self.versions[template_id].append(version)
        
        return version
    
    def get_active_version(self, template_id: str) -> Optional[PromptVersion]:
        """Get active version for template"""
        versions = self.versions.get(template_id, [])
        active = [v for v in versions if v.is_active]
        return active[-1] if active else None
    
    def rollback(self, template_id: str, version_id: str) -> bool:
        """Rollback to specific version"""
        versions = self.versions.get(template_id, [])
        for v in versions:
            v.is_active = (v.version_id == version_id)
        return True

__all__ = ['PromptVersioner', 'PromptVersion']
