from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional


class Conversation(BaseModel):
    user_id: str
    turn_id: int
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserPreferences(BaseModel):
    user_id: str
    preferences: Dict[str, Any]  # e.g., {'favorite_subjects': ['Math', 'Art']}


class Milestone(BaseModel):
    user_id: str
    milestone_id: str
    description: str
    status: str  # e.g., 'completed', 'in-progress', 'planned'
    date_achieved: Optional[datetime] = None


class MemoryWriteRequest(BaseModel):
    memory_type: str  # 'conversation', 'preference', 'milestone'
    data: Dict[str, Any]


class MemoryReadRequest(BaseModel):
    user_id: str
    query_type: str  # e.g., 'last_n_turns'
    params: Dict[str, Any] = Field(default_factory=dict)


class MemoryRetrieveRequest(BaseModel):
    user_id: str
    query_text: str
    top_k: int = 5