"""Unit tests for AI model catalog, manager, and download worker."""

import tempfile
from pathlib import Path

from core.model_downloader import (
    MODEL_CATALOG,
    EngineDownloadWorker,
    ModelDownloadWorker,
    ModelManager,
    PretrainedModelEntry,
)


def test_model_catalog_entries():
    assert len(MODEL_CATALOG) >= 4
    for model in MODEL_CATALOG:
        assert model.id
        assert model.name
        assert model.download_url.startswith("https://")
        assert model.filesize_mb > 0
        assert model.scale in (2, 4)


def test_model_manager_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ModelManager(models_dir=tmpdir)
        catalog = mgr.list_catalog()
        assert len(catalog) == len(MODEL_CATALOG)

        # Initially none installed
        for m in catalog:
            assert m.installed is False

        # Simulate installing a model
        first_m = catalog[0]
        fake_file = Path(tmpdir) / first_m.filename
        fake_file.write_bytes(b"\x00" * 4096)

        refreshed = mgr.list_catalog()
        installed_entry = next(m for m in refreshed if m.id == first_m.id)
        assert installed_entry.installed is True
        assert installed_entry.local_path == str(fake_file)

        # Delete model
        assert mgr.delete_model(first_m.id) is True
        assert not fake_file.exists()
        assert mgr.delete_model("non_existent_id") is False


def test_model_download_worker_init():
    entry = MODEL_CATALOG[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        worker = ModelDownloadWorker(model_entry=entry, target_dir=tmpdir)
        assert worker.model.id == entry.id
        assert not worker._is_cancelled
        worker.cancel()
        assert worker._is_cancelled


def test_engine_download_worker_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = str(Path(tmpdir) / "test.engine")
        worker = EngineDownloadWorker(
            url="https://example.com/fake.engine",
            target_engine_path=target_path
        )
        assert worker.url == "https://example.com/fake.engine"
        assert not worker._is_cancelled
        worker.cancel()
        assert worker._is_cancelled


def test_model_manager_sync_downloads(monkeypatch):
    import io

    class DummyResponse:
        def __init__(self, data=b"dummy content data"):
            self.data = data
            self.stream = io.BytesIO(data)

        def read(self, amt=None):
            return self.stream.read(amt)

        def getheader(self, name):
            if name.lower() == "content-length":
                return str(len(self.data))
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def dummy_urlopen(req, timeout=30):
        return DummyResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", dummy_urlopen)

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ModelManager(models_dir=tmpdir)
        first_m = MODEL_CATALOG[0]
        progress_calls = []

        saved = mgr.download_model_sync(
            first_m.id,
            progress_cb=lambda d, t, p: progress_calls.append(p)
        )
        assert Path(saved).is_file()
        assert Path(saved).read_bytes() == b"dummy content data"
        assert len(progress_calls) > 0

        engine_path = Path(tmpdir) / "test_engine.engine"
        saved_eng = mgr.download_engine_sync(
            "https://example.com/fake.engine",
            str(engine_path)
        )
        assert Path(saved_eng).is_file()
        assert Path(saved_eng).read_bytes() == b"dummy content data"


