from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel

MessageRole = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    content: Optional[str] = None
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

    def __init__(self, role: MessageRole, content: Optional[str] = None, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", None),
            tool_calls=kwargs.get("tool_calls", None),
            tool_call_id=kwargs.get("tool_call_id", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    def __str__(self):
        if self.tool_calls:
            return f"[{self.role}] tool_calls: {[tc.get('function', {}).get('name', '?') for tc in self.tool_calls]}"
        return f"[{self.role}] {self.content}"
