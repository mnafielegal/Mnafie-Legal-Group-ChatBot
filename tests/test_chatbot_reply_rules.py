import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def load_reply_rules_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]

    class ChatPromptTemplate:
        @classmethod
        def from_messages(cls, messages):
            return messages

    prompts_module = ModuleType("langchain_core.prompts")
    prompts_module.ChatPromptTemplate = ChatPromptTemplate
    prompts_module.MessagesPlaceholder = lambda name: name

    monkeypatch.setitem(sys.modules, "langchain_core", ModuleType("langchain_core"))
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", prompts_module)
    monkeypatch.setitem(
        sys.modules,
        "settings",
        SimpleNamespace(settings=SimpleNamespace(MAX_TRANSFER_REPLY_LENGTH=200)),
    )

    spec = importlib.util.spec_from_file_location(
        "chatbot_reply_rules_under_test",
        repo_root / "tools" / "chatbot_reply_rules.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_punctuation_only_message_is_contextual_when_history_exists(monkeypatch):
    reply_rules = load_reply_rules_module(monkeypatch)

    assert reply_rules.is_contextual_follow_up("?", has_history=True)
    assert reply_rules.is_contextual_follow_up("؟", has_history=True)


def test_punctuation_only_message_is_not_contextual_without_history(monkeypatch):
    reply_rules = load_reply_rules_module(monkeypatch)

    assert not reply_rules.is_contextual_follow_up("?", has_history=False)


def test_short_text_follow_up_is_contextual_when_history_exists(monkeypatch):
    reply_rules = load_reply_rules_module(monkeypatch)

    assert reply_rules.is_contextual_follow_up("ليه؟", has_history=True)


def test_plain_gibberish_is_not_contextual_follow_up(monkeypatch):
    reply_rules = load_reply_rules_module(monkeypatch)

    assert not reply_rules.is_contextual_follow_up("jwkbckewi", has_history=True)


def test_call_request_is_transfer_request(monkeypatch):
    reply_rules = load_reply_rules_module(monkeypatch)

    assert reply_rules.is_call_transfer_request("اتصل علي")
    assert reply_rules.is_call_transfer_request("please call me")


def test_rewrite_repeated_reply_uses_previous_reply_context(monkeypatch):
    reply_rules = load_reply_rules_module(monkeypatch)

    class FakeLLM:
        def __init__(self):
            self.prompt = None

        def invoke(self, prompt):
            self.prompt = prompt
            return SimpleNamespace(content="تم تحويل طلبك إلى الموظف المختص.")

    llm = FakeLLM()
    rewritten = reply_rules.rewrite_repeated_reply(
        llm,
        "سيتم تحويل طلبك للموظف المختص.",
        "سيتم تحويل طلبك للموظف المختص.",
    )

    assert rewritten == "تم تحويل طلبك إلى الموظف المختص."
    assert "reply1" in llm.prompt
    assert "reply2" in llm.prompt
