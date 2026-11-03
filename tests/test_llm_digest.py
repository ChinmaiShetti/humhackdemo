import pytest

from ha import llm
from ha.config import load_config


def test_require_digest(monkeypatch):
    monkeypatch.setattr(llm, "model_digest", lambda tag: "abc" if tag == "m" else None)
    assert llm.require_digest("m", "abc") == "abc"
    with pytest.raises(llm.ModelDigestError):
        llm.require_digest("m", "other")
    with pytest.raises(llm.ModelDigestError):
        llm.require_digest("missing", "abc")


def test_generator_and_judge_pin_the_same_locked_model():
    locked = {m["tag"]: m["digest"] for m in load_config("models.yaml")["ollama"]}
    for cfg in (load_config("generation.yaml"), load_config("judge.yaml")):
        assert cfg["model"]["think"] is False
        assert locked[cfg["model"]["tag"]] == cfg["model"]["digest"]
    assert "qwen3:4b" not in locked  # thinking-only build is excluded from the core (D-028)
