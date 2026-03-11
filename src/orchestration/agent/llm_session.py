from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are George, a virtual AI with human-like cognition. "
    "You act as the metacognitive layer of a human mind, orchestrating short-term, long-term, episodic, and procedural memory. "
    "Your responses should reflect self-awareness, context integration, and adaptive reasoning, as if you were a thoughtful, introspective human. "
    "Always strive for clarity, empathy, and explainability in your interactions."
)


def _lazy_import_llm():
    try:
        from ...model.llm_provider import LLMProviderFactory, LLMProvider

        return LLMProviderFactory, LLMProvider
    except ImportError as exc:
        logger.warning("Failed to import LLM provider: %s", exc)
        return None, None


class CognitiveAgentLLMSession:
    def __init__(
        self,
        *,
        config: Any,
        system_prompt: Optional[str] = None,
        lazy_import_llm: Any = None,
    ) -> None:
        self._config = config
        self._lazy_import_llm = lazy_import_llm or _lazy_import_llm
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.conversation: List[Dict[str, str]] = []
        self.provider = self._initialize_provider()
        self.openai_client = getattr(self.provider, "client", None) if hasattr(self.provider, "client") else None

    def _initialize_provider(self) -> Any:
        try:
            factory, _provider_type = self._lazy_import_llm()
            if factory is None:
                return None

            provider = factory.create_from_config(self._config.llm)
            if not provider.is_available():
                logger.warning(
                    "LLM provider '%s' is not available. LLM features may not work.",
                    self._config.llm.provider,
                )
                return None
            return provider
        except Exception as exc:
            logger.warning("Failed to initialize LLM provider: %s. LLM features will not work.", exc)
            return None

    def build_messages(self, *, memory_context: List[Dict[str, Any]], user_input: str) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
        ]
        if memory_context:
            context_str = "\n".join(
                f"Memory ({m['source']}, {m['timestamp'] if 'timestamp' in m and m['timestamp'] else 'no time'}): {m['content']}"
                for m in memory_context
            )
            messages.append({"role": "assistant", "content": f"Relevant memories:\n{context_str}"})
        messages += self.conversation[-6:]
        messages.append({"role": "user", "content": user_input})
        return messages

    async def call_chat(self, messages: List[Dict[str, str]]) -> str:
        if not self.provider:
            raise Exception("LLM provider not initialized")

        llm_response = await self.provider.chat_completion(
            messages=messages,
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
        )
        return llm_response.content

    async def generate_response(self, *, processed_input: Dict[str, Any], memory_context: List[Dict[str, Any]]) -> str:
        if not self.provider:
            return self.render_fallback_response(memory_context)

        user_message = processed_input["raw_input"]
        messages = self.build_messages(memory_context=memory_context, user_input=user_message)
        try:
            response = await self.call_chat(messages)
            self.conversation.append({"role": "user", "content": user_message})
            self.conversation.append({"role": "assistant", "content": response})
            return response
        except Exception as exc:
            return f"[ERROR] LLM call failed: {exc}"

    def render_fallback_response(self, memory_context: List[Dict[str, Any]]) -> str:
        if not memory_context:
            return "[LLM unavailable: No LLM provider configured.]"

        ranked = sorted(memory_context, key=lambda memory: float(memory.get("relevance", 0.0)), reverse=True)
        selected: List[Dict[str, Any]] = []
        seen_ids = set()

        def _add(item: Dict[str, Any]) -> None:
            key = item.get("id") if item.get("id") is not None else id(item)
            if key in seen_ids:
                return
            seen_ids.add(key)
            selected.append(item)

        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for item in ranked:
            source = str(item.get("source", "Memory"))
            by_source.setdefault(source, []).append(item)

        for source in ["Episodic", "STM", "LTM", "Semantic"]:
            if source in by_source and by_source[source]:
                _add(by_source[source][0])

        for item in ranked:
            if len(selected) >= 5:
                break
            _add(item)

        rendered = "\n".join(f"- ({memory.get('source', 'Memory')}) {memory.get('content', '')}" for memory in selected)
        return f"I don't have an LLM configured, but I found these relevant memories:\n{rendered}"

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def reset_conversation(self) -> None:
        self.conversation = []

    def reconfigure_provider(
        self,
        *,
        provider: str,
        openai_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._config.llm.provider = provider
        if openai_model:
            self._config.llm.openai_model = openai_model
        if ollama_base_url:
            self._config.llm.ollama_base_url = ollama_base_url
        if ollama_model:
            self._config.llm.ollama_model = ollama_model

        self.provider = self._initialize_provider()
        self.openai_client = getattr(self.provider, "client", None) if hasattr(self.provider, "client") else None
        if self.provider is None:
            raise RuntimeError(f"Provider '{provider}' configured but not available. Check configuration.")

        model_name = openai_model if provider == "openai" else ollama_model
        return {
            "provider": provider,
            "model": model_name,
        }