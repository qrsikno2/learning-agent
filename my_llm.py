import os
from pydoc import resolve
from typing import Optional
from aiofiles import base
from dotenv import load_dotenv
from openai import OpenAI
from core import LLM

class MyLLM(LLM):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ): 
        if provider == "deepseek":
            self.provider = "deepseek"
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
            
            if not self.api_key or not self.base_url:
                raise ValueError("DeepSeek API key and base URL must be provided.")
        
            self.model_name = model_name or os.getenv("DEEPSEEK_MODEL_NAME") or "deepseek-v4-flash"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens", 2048)
            self.timeout = kwargs.get("timeout", 60)

            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        
        else:
            provider = self._auto_detect_provider(api_key, base_url)
            api_key_tmp, base_url_tmp, model_name_tmp = self._resolve_credentials(api_key, base_url, provider) 
            api_key = api_key or api_key_tmp
            base_url = base_url or base_url_tmp
            model_name = model_name or model_name_tmp
            super().__init__(base_url=base_url, apikey=api_key, model_name=model_name, provider=provider)
    
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
        


if __name__ == "__main__":
    load_dotenv()
    
    llm = MyLLM()
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': '你是谁? 我是猪猪! '} 
    ]
    
    res = llm.think(messages)
    
    print("LLM Response:", res)

