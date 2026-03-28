import sys
import types

def _install_import_stubs():
    sys.modules.setdefault("gradio", types.ModuleType("gradio"))

    chains_module = types.ModuleType("langchain.chains")
    prompts_module = types.ModuleType("langchain.prompts")
    schema_module = types.ModuleType("langchain.schema")
    lc_embeddings_module = types.ModuleType("langchain_community.embeddings")
    lc_llms_module = types.ModuleType("langchain_community.llms")
    lc_vectorstores_module = types.ModuleType("langchain_community.vectorstores")

    chains_module.RetrievalQA = type("RetrievalQA", (), {})
    prompts_module.PromptTemplate = type("PromptTemplate", (), {"__init__": lambda self, **kwargs: None})
    schema_module.Document = type("Document", (), {})
    lc_embeddings_module.HuggingFaceEmbeddings = type("HuggingFaceEmbeddings", (), {"__init__": lambda self, **kwargs: None})
    lc_llms_module.HuggingFaceHub = type("HuggingFaceHub", (), {"__init__": lambda self, **kwargs: None})
    lc_vectorstores_module.Chroma = type("Chroma", (), {"__init__": lambda self, **kwargs: None})

    sys.modules.setdefault("langchain.chains", chains_module)
    sys.modules.setdefault("langchain.prompts", prompts_module)
    sys.modules.setdefault("langchain.schema", schema_module)
    sys.modules.setdefault("langchain_community.embeddings", lc_embeddings_module)
    sys.modules.setdefault("langchain_community.llms", lc_llms_module)
    sys.modules.setdefault("langchain_community.vectorstores", lc_vectorstores_module)


_install_import_stubs()

import app
from src import rag_pipeline
from src import runtime


def reset_runtime_state():
    runtime.reset_runtime_state()


def test_app_initialize_runtime_reports_missing_token(monkeypatch):
    reset_runtime_state()
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setattr(runtime, "_load_vectorstore", lambda: object())

    error = app.initialize_runtime()

    assert error is not None
    assert "HUGGINGFACEHUB_API_TOKEN" in error


def test_app_initialize_runtime_success_with_mocks(monkeypatch):
    reset_runtime_state()

    class FakeLLM:
        pass

    class FakePrompt:
        pass

    fake_vectordb = object()
    fake_chain = object()
    monkeypatch.setattr(runtime, "_load_vectorstore", lambda: fake_vectordb)
    monkeypatch.setattr(
        runtime,
        "_build_chain",
        lambda db: (fake_chain, FakeLLM(), FakePrompt()),
    )

    error = app.initialize_runtime()

    assert error is None
    assert runtime.vectordb is fake_vectordb
    assert runtime.qa_chain is fake_chain


def test_rag_initialize_runtime_requires_token(monkeypatch):
    reset_runtime_state()
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    error = rag_pipeline.initialize_runtime()

    assert error is not None
    assert "HUGGINGFACEHUB_API_TOKEN" in error


def test_generate_answer_returns_readable_error_when_uninitialized(monkeypatch):
    reset_runtime_state()
    monkeypatch.setattr(rag_pipeline, "initialize_runtime", lambda: "missing runtime")

    result = rag_pipeline.generate_answer("test query")

    assert "RAG chain not initialized" in result["answer"]
    assert result["sources"] == []
