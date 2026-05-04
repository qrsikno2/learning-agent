import os
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel

log_levels = Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]

class Config(BaseModel):
    default_model: str = "deepseek-v4-flash"
    default_provider: str = "deepseek"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    debug: bool = False
    log_level: log_levels = "INFO"
    
    max_history_length: int = 2000
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def is_verbose(self) -> bool:
        return self.log_level in ["INFO", "WARN", "ERROR", "CRITICAL"]
