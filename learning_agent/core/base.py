import logging
import os
from typing import List, Optional, Iterator
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class LLMBase:
    def __init__(self, base_url, apikey, model_name, provider: Optional[str] = "auto",
                 temperature: float = None, max_tokens: Optional[int] = None, timeout: int = 30):
        self.base_url = base_url
        self.apikey = apikey
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.client = OpenAI(base_url=self.base_url, api_key=self.apikey, timeout=self.timeout)

    def think(self, prompt: List[ChatCompletionMessageParam], **kwargs) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                stream=False,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            result = response.choices[0].message.content
            return result

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def stream_think(self, prompt: List[ChatCompletionMessageParam], **kwargs) -> Iterator[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                stream=True,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content  

        except Exception as e:
            logger.error(f"LLM stream call failed: {e}")
            return None


class LLM(LLMBase):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ): 
        if provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
            
            if not resolved_api_key or not resolved_base_url:
                raise ValueError("DeepSeek API key and base URL must be provided.")
        
            resolved_model_name = model_name or os.getenv("DEEPSEEK_MODEL_NAME") or "deepseek-v4-flash"
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 2048)
            timeout = kwargs.get("timeout", 60)

            super().__init__(
                base_url=resolved_base_url, apikey=resolved_api_key,
                model_name=resolved_model_name, provider="deepseek",
                temperature=temperature, max_tokens=max_tokens, timeout=timeout
            )
        
        else:
            detected_provider = self._auto_detect_provider(api_key, base_url)
            api_key_tmp, base_url_tmp, model_name_tmp = self._resolve_credentials(
                api_key, base_url, detected_provider
            )
            final_api_key = api_key or api_key_tmp
            final_base_url = base_url or base_url_tmp
            final_model_name = model_name or model_name_tmp
            super().__init__(
                base_url=final_base_url, apikey=final_api_key,
                model_name=final_model_name, provider=detected_provider,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens"),
                timeout=kwargs.get("timeout", 30)
            )
    
    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        if os.getenv("DEEPSEEK_API_KEY"): return "deepseek"
        # for detecting other providers, add more conditions here
        
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            
            if "deepseek" in base_url_lower:
                return "deepseek"
            
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                return "local"
        
        return "auto"
    
    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str], provider: str) -> tuple[str, str, str]:
        if provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
            resolved_model_name = os.getenv("DEEPSEEK_MODEL_NAME") or "deepseek-v4-flash"
            return resolved_api_key, resolved_base_url, resolved_model_name
        