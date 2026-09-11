from pathlib import Path

from src.generation.local_specialist_generator import _resolve_nanochat_repo


def test_resolve_nanochat_repo_uses_explicit_deployment_override(monkeypatch, tmp_path):
    configured = tmp_path / "mounted-nanochat"
    configured.mkdir()
    monkeypatch.setenv("NANOCHAT_REPO", str(configured))

    assert _resolve_nanochat_repo() == configured.resolve()


def test_resolve_nanochat_repo_defaults_to_repository_root(monkeypatch):
    monkeypatch.delenv("NANOCHAT_REPO", raising=False)

    root = _resolve_nanochat_repo()

    assert (root / "nanochat" / "checkpoint_manager.py").is_file()
    assert root == Path(__file__).resolve().parents[3]