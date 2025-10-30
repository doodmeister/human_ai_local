# LLM Provider Selection Feature

## Overview
Added ability to select between ChatGPT API and locally-hosted Ollama models in the George cognitive agent, with UI controls in the Streamlit frontend.

## Components Added

### 1. Configuration (`src/core/config.py`)
Added `LLMConfig` dataclass:
```python
@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai" or "ollama"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-nano"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 512
```

### 2. Provider Abstraction (`src/model/llm_provider.py`)
- **LLMProvider**: Abstract base class
- **OpenAIProvider**: OpenAI API implementation
- **OllamaProvider**: Local Ollama implementation
- **LLMProviderFactory**: Factory for creating providers
- **LLMResponse**: Standardized response format

### 3. CognitiveAgent Updates (`src/core/cognitive_agent.py`)
- Uses `LLMProviderFactory` to create provider from config
- New `_call_llm_chat()` method for unified LLM calls
- Backward compatible with existing code
- Automatic fallback if provider unavailable

### 4. API Endpoint (`src/interfaces/api/chat_endpoints.py`)
```
POST /agent/config/llm
{
    "provider": "openai" | "ollama",
    "openai_model": "gpt-4.1-nano",
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "llama3.2"
}
```
Updates LLM provider configuration at runtime and reinitializes the agent's provider.

### 5. Streamlit UI (`scripts/george_streamlit_chat.py`)
Added to sidebar:
- **Provider Selector**: Dropdown to choose OpenAI or Ollama
- **OpenAI Settings**: Model name input (when OpenAI selected)
- **Ollama Settings**: Base URL and model name inputs (when Ollama selected)
- **Apply Button**: Updates backend configuration

## Environment Variables (.env)
```bash
# LLM Provider Selection
LLM_PROVIDER=openai              # or "ollama"

# OpenAI (when provider=openai)
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4.1-nano

# Ollama (when provider=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

## Usage

### Using OpenAI (Default)
1. Set `OPENAI_API_KEY` in `.env`
2. Select "openai" in UI dropdown
3. Choose model (e.g., gpt-4.1-nano, gpt-4o, gpt-3.5-turbo)
4. Click "Apply LLM Config"

### Using Ollama (Local)
1. Install and start Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Select "ollama" in UI dropdown
4. Set base URL (default: http://localhost:11434)
5. Choose model name
6. Click "Apply LLM Config"

## Benefits
- **Flexibility**: Switch between cloud and local models
- **Cost Control**: Use free local models for development
- **Privacy**: Keep data local with Ollama
- **Testing**: Easy to compare model performance
- **Runtime Updates**: No restart needed to change providers

## Technical Details

### Provider Interface
All providers implement:
- `chat_completion(messages, temperature, max_tokens)` - Generate responses
- `is_available()` - Check if provider is properly configured

### Response Format
```python
@dataclass
class LLMResponse:
    content: str              # Generated text
    model: str               # Model used
    usage: Dict[str, int]    # Token usage stats
    metadata: Dict[str, Any] # Provider-specific data
```

### Error Handling
- Provider unavailable → Warning message in UI
- API errors → Graceful fallback with error message
- Invalid config → Validation and user feedback

## Testing Checklist
- [ ] Start backend with OpenAI provider
- [ ] Send chat message, verify OpenAI response
- [ ] Switch to Ollama in UI (if Ollama installed)
- [ ] Send chat message, verify Ollama response
- [ ] Switch back to OpenAI
- [ ] Verify config persists across page refreshes
