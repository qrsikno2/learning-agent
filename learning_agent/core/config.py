import logging
import os
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel

log_levels = Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

class Config(BaseModel):
    default_model: str = "deepseek-v4-flash"
    default_provider: str = "deepseek"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    log_level: log_levels = "INFO"

    max_history_length: int = 2000

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def setup_logging(self) -> None:
        level = _LEVEL_MAP.get(self.log_level, logging.INFO)
        pkg_logger = logging.getLogger("learning_agent")
        pkg_logger.setLevel(level)
        if not pkg_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            ))
            pkg_logger.addHandler(handler)
            pkg_logger.propagate = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
