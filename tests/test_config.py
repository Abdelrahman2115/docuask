from pathlib import Path

import pytest

from docuask.config import PROJECT_ROOT, load_settings

NO_ENV = Path("does-not-exist.env")


def test_defaults_load(monkeypatch):
    monkeypatch.delenv("DOCUASK_LLM_PROVIDER", raising=False)
    s = load_settings(env_path=NO_ENV)
    assert s.llm.provider in {"groq", "ollama"}
    assert s.chunking.chunk_overlap < s.chunking.chunk_size
    assert s.retrieval.top_k > 0


def test_paths_are_absolute_inside_project():
    s = load_settings(env_path=NO_ENV)
    assert s.paths.index_dir.is_absolute()
    assert PROJECT_ROOT in s.paths.raw_dir.parents


def test_env_overrides_provider(monkeypatch):
    monkeypatch.setenv("DOCUASK_LLM_PROVIDER", "ollama")
    assert load_settings(env_path=NO_ENV).llm.provider == "ollama"


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("DOCUASK_LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        load_settings(env_path=NO_ENV)
