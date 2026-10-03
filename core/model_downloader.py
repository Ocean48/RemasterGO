"""Pretrained AI model catalog, manager, and asynchronous downloader for RemasterGO."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QThread, Signal


@dataclass
class PretrainedModelEntry:
    id: str
    name: str
    category: str
    description: str
    scale: int
    filesize_mb: float
    download_url: str
    filename: str
    installed: bool = False
    local_path: Optional[str] = None


MODEL_CATALOG: List[PretrainedModelEntry] = [
    PretrainedModelEntry(
        id="realesrgan_x4plus",
        name="Real-ESRGAN x4 Plus",
        category="General Real-World Video",
        description="High-fidelity 4x super-resolution for live-action footage, home videos, and film.",
        scale=4,
        filesize_mb=67.0,
        download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        filename="RealESRGAN_x4plus.pth"
    ),
    PretrainedModelEntry(
        id="realesr_general_x4v3",
        name="Real-ESRGAN General x4 (ONNX)",
        category="General Video / TensorRT Ready",
        description="Fast general video upscaling model in ONNX format, directly compilable to TensorRT engine.",
        scale=4,
        filesize_mb=4.7,
        download_url="https://huggingface.co/JoPmt/Real_Esrgan_x2_Onnx_Tflite_Tfjs/resolve/main/ano_test/realesr-general-x4v3.onnx",
        filename="realesr-general-x4v3.onnx"
    ),
    PretrainedModelEntry(
        id="realesrgan_x4plus_anime_6b",
        name="Real-ESRGAN x4 Anime (6B)",
        category="Anime & Animation",
        description="Optimized 4x upscaler for animated content with strong line de-blurring and artifact reduction.",
        scale=4,
        filesize_mb=17.9,
        download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        filename="RealESRGAN_x4plus_anime_6B.pth"
    ),
    PretrainedModelEntry(
        id="realesr_animevideov3",
        name="Real-ESRGAN AnimeVideo v3",
        category="Anime & Fast Video",
        description="Video-specific anime super-resolution with temporal stability and low noise generation.",
        scale=4,
        filesize_mb=2.4,
        download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        filename="realesr-animevideov3.pth"
    ),
    PretrainedModelEntry(
        id="realesrgan_x2plus",
        name="Real-ESRGAN x2 Plus",
        category="General Video 2x",
        description="Balanced 2x super-resolution for general real-world video with low memory footprint.",
        scale=2,
        filesize_mb=67.0,
        download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        filename="RealESRGAN_x2plus.pth"
    )
]


class ModelManager:
    """Manages local storage and discovery of AI model files."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent / "cache" / "models"
        else:
            self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_catalog(self) -> List[PretrainedModelEntry]:
        """Return the catalog with installation statuses refreshed against disk."""
        catalog_copy: List[PretrainedModelEntry] = []
        for item in MODEL_CATALOG:
            target_path = self.models_dir / item.filename
            is_present = target_path.is_file() and target_path.stat().st_size > 1024
            catalog_copy.append(PretrainedModelEntry(
                id=item.id,
                name=item.name,
                category=item.category,
                description=item.description,
                scale=item.scale,
                filesize_mb=item.filesize_mb,
                download_url=item.download_url,
                filename=item.filename,
                installed=is_present,
                local_path=str(target_path) if is_present else None
            ))
        return catalog_copy

    def get_model(self, model_id: str) -> Optional[PretrainedModelEntry]:
        for m in self.list_catalog():
            if m.id == model_id:
                return m
        return None

    def delete_model(self, model_id: str) -> bool:
        entry = self.get_model(model_id)
        if entry and entry.local_path and Path(entry.local_path).is_file():
            Path(entry.local_path).unlink(missing_ok=True)
            return True
        return False

    def download_model_sync(
        self,
        model_id: str,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None
    ) -> str:
        """Download model weights synchronously (e.g. for CLI or automated setup)."""
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError(f"Model ID '{model_id}' not found in catalog.")

        dest_file = self.models_dir / entry.filename
        temp_file = self.models_dir / f"{entry.filename}.download"

        if status_cb:
            status_cb(f"Connecting to {entry.download_url}...")

        req = urllib.request.Request(
            entry.download_url,
            headers={"User-Agent": "RemasterGO-Engine/1.0"}
        )

        with urllib.request.urlopen(req, timeout=30) as response, open(temp_file, "wb") as out_f:
            content_len = response.getheader("Content-Length")
            total_bytes = int(content_len) if content_len and content_len.isdigit() else int(entry.filesize_mb * 1024 * 1024)

            downloaded = 0
            chunk_size = 64 * 1024

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total_bytes * 100.0) if total_bytes > 0 else 0.0
                if progress_cb:
                    progress_cb(downloaded, total_bytes, pct)

        if dest_file.is_file():
            dest_file.unlink(missing_ok=True)
        temp_file.rename(dest_file)

        if status_cb:
            status_cb(f"Downloaded {entry.filename} successfully.")

        return str(dest_file)

    def download_engine_sync(
        self,
        url: str,
        target_engine_path: str,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None
    ) -> str:
        """Download precompiled TensorRT engine or ONNX model synchronously."""
        dest_file = Path(target_engine_path)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = dest_file.with_suffix(".engine.download")

        if status_cb:
            status_cb(f"Connecting to {url}...")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "RemasterGO-Engine/1.0"}
        )

        with urllib.request.urlopen(req, timeout=45) as response, open(temp_file, "wb") as out_f:
            content_len = response.getheader("Content-Length")
            total_bytes = int(content_len) if content_len and content_len.isdigit() else 0

            downloaded = 0
            chunk_size = 128 * 1024

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total_bytes * 100.0) if total_bytes > 0 else 0.0
                if progress_cb:
                    progress_cb(downloaded, total_bytes, pct)

        if dest_file.is_file():
            dest_file.unlink(missing_ok=True)
        temp_file.rename(dest_file)

        if status_cb:
            status_cb("Engine downloaded successfully.")

        return str(dest_file)


class ModelDownloadWorker(QThread):
    """Asynchronous worker thread downloading model weights with progress reporting."""

    sig_progress = Signal(str, int, int, float)  # model_id, downloaded_bytes, total_bytes, percent
    sig_status = Signal(str, str)               # model_id, status_text
    sig_finished = Signal(str, str)             # model_id, local_path
    sig_error = Signal(str, str)                # model_id, error_details

    def __init__(self, model_entry: PretrainedModelEntry, target_dir: str, parent=None):
        super().__init__(parent)
        self.model = model_entry
        self.target_dir = Path(target_dir)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)
        dest_file = self.target_dir / self.model.filename
        temp_file = self.target_dir / f"{self.model.filename}.download"

        try:
            self.sig_status.emit(self.model.id, f"Connecting to {self.model.download_url}...")

            req = urllib.request.Request(
                self.model.download_url,
                headers={"User-Agent": "RemasterGO-Engine/1.0"}
            )

            with urllib.request.urlopen(req, timeout=30) as response, open(temp_file, "wb") as out_f:
                content_len = response.getheader("Content-Length")
                total_bytes = int(content_len) if content_len and content_len.isdigit() else int(self.model.filesize_mb * 1024 * 1024)

                downloaded = 0
                chunk_size = 64 * 1024

                while not self._is_cancelled:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    pct = (downloaded / total_bytes * 100.0) if total_bytes > 0 else 0.0
                    self.sig_progress.emit(self.model.id, downloaded, total_bytes, pct)

            if self._is_cancelled:
                if temp_file.is_file():
                    temp_file.unlink(missing_ok=True)
                self.sig_status.emit(self.model.id, "Download cancelled.")
                return

            # Rename .download to final filename
            if dest_file.is_file():
                dest_file.unlink(missing_ok=True)
            temp_file.rename(dest_file)

            self.sig_status.emit(self.model.id, "Download complete.")
            self.sig_finished.emit(self.model.id, str(dest_file))

        except Exception as e:
            if temp_file.is_file():
                temp_file.unlink(missing_ok=True)
            self.sig_error.emit(self.model.id, str(e))


class EngineDownloadWorker(QThread):
    """Asynchronous worker thread downloading precompiled TensorRT .engine files with progress."""

    sig_progress = Signal(int, int, float)  # downloaded_bytes, total_bytes, percent
    sig_status = Signal(str)
    sig_finished = Signal(str)              # target_engine_path
    sig_error = Signal(str)

    def __init__(self, url: str, target_engine_path: str, parent=None):
        super().__init__(parent)
        self.url = url
        self.target_path = Path(target_engine_path)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        dest_file = self.target_path
        temp_file = self.target_path.with_suffix(".engine.download")

        try:
            self.sig_status.emit(f"Connecting to {self.url}...")
            req = urllib.request.Request(
                self.url,
                headers={"User-Agent": "RemasterGO-Engine/1.0"}
            )

            with urllib.request.urlopen(req, timeout=45) as response, open(temp_file, "wb") as out_f:
                content_len = response.getheader("Content-Length")
                total_bytes = int(content_len) if content_len and content_len.isdigit() else 0

                downloaded = 0
                chunk_size = 128 * 1024

                while not self._is_cancelled:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    pct = (downloaded / total_bytes * 100.0) if total_bytes > 0 else 0.0
                    self.sig_progress.emit(downloaded, total_bytes, pct)

            if self._is_cancelled:
                if temp_file.is_file():
                    temp_file.unlink(missing_ok=True)
                self.sig_status.emit("Engine download cancelled.")
                return

            if dest_file.is_file():
                dest_file.unlink(missing_ok=True)
            temp_file.rename(dest_file)

            self.sig_status.emit("Engine download complete.")
            self.sig_finished.emit(str(dest_file))

        except Exception as e:
            if temp_file.is_file():
                temp_file.unlink(missing_ok=True)
            self.sig_error.emit(str(e))

