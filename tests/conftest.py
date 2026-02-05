import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# Compatibility aliases for refactored module layout used by legacy tests.
import importlib
import types


def _alias_module(old: str, new: str) -> None:
    sys.modules[old] = importlib.import_module(new)


def _alias_chat_modules() -> None:
    _alias_module("src.chat", "src.orchestration.chat")
    _alias_module("src.chat.factory", "src.orchestration.chat.factory")
    _alias_module("src.chat.context_builder", "src.orchestration.chat.context_builder")
    _alias_module("src.chat.conversation_session", "src.orchestration.chat.conversation_session")
    _alias_module("src.chat.chat_service", "src.orchestration.chat.chat_service")
    _alias_module("src.chat.metrics", "src.orchestration.chat.metrics")
    _alias_module("src.chat.constants", "src.orchestration.chat.constants")
    _alias_module("src.chat.scoring", "src.orchestration.chat.scoring")
    _alias_module("src.chat.models", "src.orchestration.chat.models")
    _alias_module("src.chat.memory_capture", "src.orchestration.chat.memory_capture")
    _alias_module("src.chat.memory_query_interface", "src.orchestration.chat.memory_query_interface")
    _alias_module("src.chat.memory_query_parser", "src.orchestration.chat.memory_query_parser")
    _alias_module("src.chat.plan_summarizer", "src.orchestration.chat.plan_summarizer")
    _alias_module("src.chat.goal_detector", "src.orchestration.chat.goal_detector")
    _alias_module("src.chat.executive_orchestrator", "src.orchestration.chat.executive_orchestrator")
    _alias_module("src.chat.intent_classifier_v2", "src.orchestration.chat.intent_classifier_v2")
    _alias_module("src.chat.provenance", "src.orchestration.chat.provenance")
    _alias_module("src.chat.emotion_salience", "src.orchestration.chat.emotion_salience")


def _alias_attention_modules() -> None:
    _alias_module("src.attention", "src.cognition.attention")
    _alias_module(
        "src.attention.attention_mechanism",
        "src.cognition.attention.attention_mechanism",
    )


def _alias_api_entrypoints() -> None:
    try:
        from main import _build_api_app
    except Exception:
        return

    app_module = types.ModuleType("start_server")
    app_module.app = _build_api_app()
    sys.modules.setdefault("start_server", app_module)

    legacy_module = types.ModuleType("george_api_simple")
    legacy_module.app = app_module.app
    sys.modules.setdefault("george_api_simple", legacy_module)


_alias_chat_modules()
_alias_attention_modules()
_alias_api_entrypoints()
