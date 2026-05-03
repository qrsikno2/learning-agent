from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel

MessageRole = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", None)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self):
        return f"[{self.role}] {self.content}"
