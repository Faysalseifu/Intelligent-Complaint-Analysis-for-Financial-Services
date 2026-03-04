import app
from src import rag_pipeline


def reset_app_runtime_state():
    app.qa_chain = None
    app.llm = None
    app.vectordb = None
    app.prompt = None
    app.runtime_error = None
    app._runtime_initialized = False


def test_app_initialize_runtime_reports_missing_token(monkeypatch):
    reset_app_runtime_state()
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setattr(app, "_load_vectorstore", lambda: object())

    error = app.initialize_runtime()

    assert error is not None
    assert "HUGGINGFACEHUB_API_TOKEN" in error


def test_app_initialize_runtime_success_with_mocks(monkeypatch):
    reset_app_runtime_state()

    class FakeLLM:
        pass

    class FakePrompt:
        pass

    fake_vectordb = object()
    fake_chain = object()
    monkeypatch.setattr(app, "_load_vectorstore", lambda: fake_vectordb)
    monkeypatch.setattr(
        app,
        "_build_chain",
        lambda db: (fake_chain, FakeLLM(), FakePrompt()),
    )

    error = app.initialize_runtime()

    assert error is None
    assert app.vectordb is fake_vectordb
    assert app.qa_chain is fake_chain


def test_rag_initialize_runtime_requires_token(monkeypatch):
    rag_pipeline.vectordb = None
    rag_pipeline.llm = None
    rag_pipeline.rag_chain = None
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    error = rag_pipeline.initialize_runtime()

    assert error is not None
    assert "HUGGINGFACEHUB_API_TOKEN" in error


def test_generate_answer_returns_readable_error_when_uninitialized(monkeypatch):
    rag_pipeline.vectordb = None
    rag_pipeline.llm = None
    rag_pipeline.rag_chain = None
    monkeypatch.setattr(rag_pipeline, "initialize_runtime", lambda: "missing runtime")

    result = rag_pipeline.generate_answer("test query")

    assert "RAG chain not initialized" in result["answer"]
    assert result["sources"] == []
