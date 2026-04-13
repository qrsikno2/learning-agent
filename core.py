import logging
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


logger = logging.getLogger("LearningAgent")


class LLM:
    def __init__(self, base_url, apikey, model_name):
        self.base_url = base_url
        self.apikey = apikey
        self.model_name = model_name

        self.client = OpenAI(base_url=self.base_url, api_key=self.apikey)

    def think(self, prompt: List[ChatCompletionMessageParam]) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                stream=False,
            )

            result = response.choices[0].message.content
            if not result:
                finish_reason = response.choices[0].finish_reason
                logger.warning(
                    f"LLM response has no content. Finish reason: {finish_reason}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in LLM think: {e}")
            return None


class Toolset:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, func: Callable[..., Any]) -> None:
        self.tools[name] = {"description": description, "func": func}

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        return self.tools.get(name, {}).get("func", None)

    def get_available_tools(self) -> str:
        return "\n".join(
            [f"{name}: {info['description']}" for name, info in self.tools.items()]
        )
