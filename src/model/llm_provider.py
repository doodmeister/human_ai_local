"""
LLM Provider Abstraction Layer
Supports OpenAI API and Ollama local models
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-nano"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available - API key not configured")
        
        # Store client reference to help type checker narrow the type
        client = self.client
        assert client is not None  # Type guard for type checker
        
        import asyncio
        from openai.types.chat import ChatCompletionMessageParam
        
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            # Cast messages to OpenAI's expected type
            cast_messages: List[ChatCompletionMessageParam] = messages  # type: ignore
            
            response = client.chat.completions.create(
                model=self.model,
                messages=cast_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            return LLMResponse(
                content=content.strip(),
                model=response.model,
                usage=usage,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
        
        return await loop.run_in_executor(None, _sync_call)


class OllamaProvider(LLMProvider):
    """Ollama local model provider"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def is_available(self) -> bool:
        """Check if Ollama server is reachable"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError(f"Ollama provider not available at {self.base_url}")
        
        import asyncio
        import requests
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            # Convert messages to Ollama format
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=90
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("message", {}).get("content", "")
            
            # Extract usage if available
            usage = None
            if "prompt_eval_count" in data or "eval_count" in data:
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                }
            
            return LLMResponse(
                content=content.strip(),
                model=self.model,
                usage=usage,
                metadata={
                    "done": data.get("done"),
                    "total_duration": data.get("total_duration"),
                }
            )
        
        return await loop.run_in_executor(None, _sync_call)


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(
        provider_type: str,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4.1-nano",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
    ) -> LLMProvider:
        """
        Create an LLM provider based on type
        
        Args:
            provider_type: "openai" or "ollama"
            openai_api_key: OpenAI API key (if using OpenAI)
            openai_model: OpenAI model name
            ollama_base_url: Ollama server URL
            ollama_model: Ollama model name
        
        Returns:
            Configured LLMProvider instance
        """
        provider_type = provider_type.lower()
        
        if provider_type == "openai":
            return OpenAIProvider(api_key=openai_api_key, model=openai_model)
        elif provider_type == "ollama":
            return OllamaProvider(base_url=ollama_base_url, model=ollama_model)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}. Use 'openai' or 'ollama'.")
    
    @staticmethod
    def create_from_config(config) -> LLMProvider:
        """Create provider from LLMConfig object"""
        return LLMProviderFactory.create_provider(
            provider_type=config.provider,
            openai_api_key=config.openai_api_key,
            openai_model=config.openai_model,
            ollama_base_url=config.ollama_base_url,
            ollama_model=config.ollama_model,
        )
