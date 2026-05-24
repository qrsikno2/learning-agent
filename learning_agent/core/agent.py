import logging
from abc import ABC
from typing import Optional

from .message import Message
from .config import Config
from .base import LLM
from .console import print_user_input, print_assistant_reply

logger = logging.getLogger(__name__)

class Agent(ABC):
    def __init__(self, name: str, llm: LLM, system_prompt: Optional[str] = None, config: Optional[Config] = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
    
    @staticmethod
    def print_turns(func):
        def wrapper(self, input_text: str, **kwargs) -> str:
            print_user_input(input_text)
            response = func(self, input_text, **kwargs)
            print_assistant_reply(response)
            return response
        return wrapper

    def _build_prompt(self, input_text: str = None, custom_sys_prompt: str = None) -> list[dict]:
        messages = []
        sys_prompt = custom_sys_prompt if custom_sys_prompt is not None else self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        if input_text is not None:
            messages.append({"role": "user", "content": input_text})
        return messages

    def add_message(self, message: Message):
        self._history.append(message)
        logger.debug("%s", message)
        if len(self._history) > self.config.max_history_length:
            self._history = self._history[-self.config.max_history_length:]

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
