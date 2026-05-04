from abc import ABC, abstractmethod
from typing import Optional, Any, Iterator

from .message import Message
from .config import Config
from .base import LLM

class StreamableMixin(ABC):
    @abstractmethod
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        raise NotImplementedError("This agent does not support streaming output.")
    
class RunnableMixin(ABC):
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement the run method.")

class Agent(ABC):
    def __init__(self, name: str, llm: LLM, system_prompt: Optional[str] = None, config: Optional[Config] = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    def _build_prompt(self, input_text: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})
        return messages

    def add_message(self, message: Message):
        self._history.append(message)
        if len(self._history) > self.config.max_history_length:
            self._history = self._history[-self.config.max_history_length:]

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
