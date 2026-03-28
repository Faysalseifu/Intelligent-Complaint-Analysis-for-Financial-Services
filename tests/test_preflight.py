from pathlib import Path

from src import preflight


def test_validate_env_reports_missing(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    errors = preflight.validate_env(["HUGGINGFACEHUB_API_TOKEN"])
    assert errors
    assert "HUGGINGFACEHUB_API_TOKEN" in errors[0]


def test_validate_env_passes_when_set(monkeypatch):
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "token")
    errors = preflight.validate_env(["HUGGINGFACEHUB_API_TOKEN"])
    assert errors == []


def test_validate_vector_store_missing(tmp_path: Path):
    missing_dir = tmp_path / "missing_vector_store"
    errors = preflight.validate_vector_store(missing_dir)
    assert errors
    assert "not found" in errors[0]


def test_validate_vector_store_empty(tmp_path: Path):
    empty_dir = tmp_path / "vector_store"
    empty_dir.mkdir(parents=True)
    errors = preflight.validate_vector_store(empty_dir)
    assert errors
    assert "is empty" in errors[0]


def test_validate_vector_store_ok(tmp_path: Path):
    vector_dir = tmp_path / "vector_store"
    vector_dir.mkdir(parents=True)
    (vector_dir / "index.bin").write_text("ok", encoding="utf-8")
    errors = preflight.validate_vector_store(vector_dir)
    assert errors == []


def test_run_preflight_ci(monkeypatch):
    monkeypatch.setattr(preflight, "validate_required_files", lambda: [])
    errors = preflight.run_preflight("ci")
    assert errors == []


def test_run_preflight_app_aggregates(monkeypatch):
    monkeypatch.setattr(preflight, "validate_required_files", lambda: ["missing file"])
    monkeypatch.setattr(preflight, "validate_env", lambda required_env=None: ["missing env"])
    monkeypatch.setattr(preflight, "validate_vector_store", lambda vector_store_dir=None: ["missing store"])
    errors = preflight.run_preflight("app")
    assert errors == ["missing file", "missing env", "missing store"]
