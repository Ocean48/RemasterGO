"""Pretrained NCNN Vulkan AI model catalog, manager, and asynchronous downloader for RemasterGO."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

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
    param_url: Optional[str] = None
    param_filename: Optional[str] = None
    best_for: str = ""
    pros: str = ""
    cons: str = ""
    speed_tier: str = "Standard (23-Block RRDB)"
    recommended_denoise: float = 2.2
    installed: bool = False
    local_path: Optional[str] = None


MODEL_CATALOG: List[PretrainedModelEntry] = [
    PretrainedModelEntry(
        id="nomos2_otf_esrgan_x4",
        name="Nomos2 OTF ESRGAN x4",
        category="Degraded / Noisy Old Video",
        description="Trained with On-The-Fly compression and noise degradation to clean and upscale noisy old videos.",
        scale=4,
        filesize_mb=31.9,
        download_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/4xNomos8kSC.bin",
        filename="Nomos2_OTF_ESRGAN_x4.bin",
        param_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/4xNomos8kSC.param",
        param_filename="Nomos2_OTF_ESRGAN_x4.param",
        best_for="Old AVI/MPG/DVD videos, VHS rips, and low-bitrate videos with visible compression blocks and grain.",
        pros="Trained on realistic On-The-Fly (OTF) compression and noise; recovers strong detail without turning blockiness into harsh artifacts.",
        cons="Slightly softer on already-clean HD footage than UltraSharp; standard 23-block RRDB processing speed.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=2.4
    ),
    PretrainedModelEntry(
        id="uniscale_restore",
        name="UniScale Restore x4",
        category="Degraded / Noisy Old Video",
        description="Dedicated restoration upscaler that suppresses old video compression blocks and noise while recovering sharp detail.",
        scale=4,
        filesize_mb=31.9,
        download_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/uniscale_restore.bin",
        filename="uniscale_restore.bin",
        param_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/uniscale_restore.param",
        param_filename="uniscale_restore.param",
        best_for="Heavily degraded or blocky old videos where other models leave crunchy mosquito noise around edges.",
        pros="Actively suppresses ringing, mosquito noise, and MPEG block edges while reconstructing clean subject outlines.",
        cons="Can smooth over ultra-fine micro-textures (like distant grass or pores) if Denoise Strength is set above 3.0.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=2.2
    ),
    PretrainedModelEntry(
        id="ultramix_balanced_4x",
        name="UltraMix Balanced x4",
        category="Balanced Natural Detail & Low Noise",
        description="Balanced blend of Real-ESRGAN, ESRNet, and UltraSharp for natural textures without noise hallucination.",
        scale=4,
        filesize_mb=31.9,
        download_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/ultramix-balanced-4x.bin",
        filename="ultramix-balanced-4x.bin",
        param_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/ultramix-balanced-4x.param",
        param_filename="ultramix-balanced-4x.param",
        best_for="Home videos, family recordings, and general older live-action clips needing a natural, non-synthetic look.",
        pros="Blends non-GAN ESRNet weights to eliminate fake 'wormy' GAN textures on noisy backgrounds while keeping natural sharpness.",
        cons="Less aggressive edge sharpening on blurry text or signs compared to UltraSharp.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=2.0
    ),
    PretrainedModelEntry(
        id="high_fidelity_4x",
        name="High-Fidelity Live-Action x4",
        category="Photorealistic Video Restoration",
        description="High-fidelity photorealistic super-resolution tuned for live-action footage and natural skin/fabric textures.",
        scale=4,
        filesize_mb=31.9,
        download_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/high-fidelity-4x.bin",
        filename="high-fidelity-4x.bin",
        param_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/high-fidelity-4x.param",
        param_filename="high-fidelity-4x.param",
        best_for="480p/540p/720p movies, TV episodes, and videos with people, faces, skin tones, and fabric.",
        pros="Produces the most camera-like, photorealistic skin tones, hair, and clothing without artificial sharpening halos.",
        cons="Does not aggressively deblur severely out-of-focus shots or heavily damaged analog tape.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=1.6
    ),
    PretrainedModelEntry(
        id="realesrgan_x4plus",
        name="Real-ESRGAN x4 Plus",
        category="General Real-World Video",
        description="High-fidelity 4x super-resolution for live-action footage, home videos, and film.",
        scale=4,
        filesize_mb=64.0,
        download_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/upscayl-standard-4x.bin",
        filename="realesrgan-x4plus.bin",
        param_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/upscayl-standard-4x.param",
        param_filename="realesrgan-x4plus.param",
        best_for="General live-action videos and standard-definition footage with moderate quality.",
        pros="Classic all-around 23-block RRDB upscaler with strong deblurring and edge synthesis.",
        cons="Can turn heavy source noise into GAN artifacts if pre-denoising is too low, and can make low-res faces look slightly plastic.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=2.0
    ),
    PretrainedModelEntry(
        id="nomos8k_sc_4x",
        name="4xNomos8kSC Compression-Resistant",
        category="Film & Compressed Video 4x",
        description="Nomos 8K super-resolution model resilient against compression artifacts in older digital videos.",
        scale=4,
        filesize_mb=31.9,
        download_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/4xNomos8kSC.bin",
        filename="4xNomos8kSC.bin",
        param_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/4xNomos8kSC.param",
        param_filename="4xNomos8kSC.param",
        best_for="Old web downloads (YouTube/streaming rips) and standard-definition digital camcorder videos.",
        pros="Trained on 8K photography with compression resistance; keeps flat backgrounds clean while sharpening subjects.",
        cons="Can make intentional vintage film grain look slightly too smooth.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=1.8
    ),
    PretrainedModelEntry(
        id="remacri_4x",
        name="Foolhardy Remacri x4",
        category="Fine Texture & Film Restoration",
        description="Renowned detail restoration network for film grain, foliage, and fine surface textures.",
        scale=4,
        filesize_mb=31.9,
        download_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/remacri-4x.bin",
        filename="remacri-4x.bin",
        param_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/remacri-4x.param",
        param_filename="remacri-4x.param",
        best_for="Nature, landscapes, outdoor travel footage, foliage, stone, water, and clean DVD/film transfers.",
        pros="Excels at reconstructing fine organic micro-textures (leaves, wood, brick, water) that other models smooth away.",
        cons="Because it amplifies fine surface texture, it will also amplify source noise if used on a very noisy video without strong pre-denoising.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=1.6
    ),
    PretrainedModelEntry(
        id="ultrasharp_4x",
        name="UltraSharp x4 High-Clarity",
        category="Maximum Edge Sharpness",
        description="High-contrast super-resolution model tuned for maximum edge crispness on cleaner video sources.",
        scale=4,
        filesize_mb=31.9,
        download_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/ultrasharp-4x.bin",
        filename="ultrasharp-4x.bin",
        param_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/ultrasharp-4x.param",
        param_filename="ultrasharp-4x.param",
        best_for="Cleaner 480p/540p/720p videos, architecture, text, sports, and clips where maximum crispness is desired.",
        pros="Delivers the highest edge contrast and perceived sharpness; makes soft edges, text, and buildings pop immediately.",
        cons="Not ideal for heavily noisy old videos—its aggressive sharpening can accentuate remaining compression blocks or grain.",
        speed_tier="Standard Quality (23-Block RRDB)",
        recommended_denoise=1.5
    ),
    PretrainedModelEntry(
        id="realesrgan_general_wdn_x4v3",
        name="Real-ESRGAN General WDN x4 v3",
        category="Fast Video with Built-In Denoise",
        description="Compact general video upscaler with integrated Wide Denoise (WDN) weights for fast cleanup.",
        scale=4,
        filesize_mb=2.3,
        download_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/RealESRGAN_General_WDN_x4_v3.bin",
        filename="RealESRGAN_General_WDN_x4_v3.bin",
        param_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/RealESRGAN_General_WDN_x4_v3.param",
        param_filename="RealESRGAN_General_WDN_x4_v3.param",
        best_for="Fast batch processing, long videos, or very noisy clips needing rapid smoothing and upscaling.",
        pros="Runs 5-8x faster than 32 MB models and includes built-in Wide Denoise (WDN) weights to flatten grain automatically.",
        cons="Small 2.3 MB network capacity cannot reconstruct fine realistic textures (hair, pores, foliage) as deeply as 32 MB models.",
        speed_tier="Ultra-Fast (Compact SRVGGNet)",
        recommended_denoise=1.8
    ),
    PretrainedModelEntry(
        id="realesrgan_x4plus_anime_6b",
        name="Real-ESRGAN x4 Anime (6B)",
        category="Anime & Animation",
        description="Optimized 4x upscaler for animated content with strong line de-blurring and artifact reduction.",
        scale=4,
        filesize_mb=17.1,
        download_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/digital-art-4x.bin",
        filename="realesrgan-x4plus-anime.bin",
        param_url="https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/digital-art-4x.param",
        param_filename="realesrgan-x4plus-anime.param",
        best_for="2D anime, cartoons, cel animation, digital art, and CGI with sharp outlines.",
        pros="Uses a 6-block RRDB architecture (~3x faster than 23-block models) tuned for crisp line art and flat color shading.",
        cons="Makes real-world live-action video and human faces look like a cartoon/painting.",
        speed_tier="Fast (6-Block RRDB)",
        recommended_denoise=1.5
    ),
    PretrainedModelEntry(
        id="realesr_animevideov3",
        name="Real-ESRGAN AnimeVideo v3",
        category="Anime & Fast Video",
        description="Video-specific anime super-resolution with temporal stability and low noise generation.",
        scale=4,
        filesize_mb=1.2,
        download_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/realesr-animevideov3-x4.bin",
        filename="realesr-animevideov3-x4.bin",
        param_url="https://raw.githubusercontent.com/upscayl/custom-models/main/models/realesr-animevideov3-x4.param",
        param_filename="realesr-animevideov3-x4.param",
        best_for="Fast 2x/3x/4x upscaling of long anime episodes and animated series.",
        pros="Extremely fast, supports native 2x/3x/4x scales, and minimizes frame-to-frame line flicker on animation.",
        cons="Smallest model capacity (1.2 MB); only suitable for animation and cartoons.",
        speed_tier="Ultra-Fast (Video Compact)",
        recommended_denoise=1.2
    ),
]


class ModelManager:
    """Manages local storage and discovery of NCNN Vulkan AI model files."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent / "bin" / "realesrgan" / "models"
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
                param_url=item.param_url,
                param_filename=item.param_filename,
                best_for=item.best_for,
                pros=item.pros,
                cons=item.cons,
                speed_tier=item.speed_tier,
                recommended_denoise=item.recommended_denoise,
                installed=is_present,
                local_path=str(target_path) if is_present else None
            ))
        return catalog_copy

    def get_model(self, model_id: str) -> Optional[PretrainedModelEntry]:
        for m in self.list_catalog():
            if m.id == model_id:
                return m
        return None

    def find_model_entry(self, query_name: str) -> Optional[PretrainedModelEntry]:
        """Find catalog entry matching UI selection token, filename stem, or ID."""
        token = query_name.strip().split()[0] if query_name.strip() else ""
        clean_q = token.lower().replace("-", "").replace("_", "")
        catalog = self.list_catalog()
        for m in catalog:
            stem = Path(m.filename).stem.lower().replace("-", "").replace("_", "")
            m_id = m.id.lower().replace("-", "").replace("_", "")
            if clean_q in (stem, m_id) or clean_q in stem or stem in clean_q:
                return m
        return catalog[0] if catalog else None

    def delete_model(self, model_id: str) -> bool:
        entry = self.get_model(model_id)
        if entry and entry.local_path and Path(entry.local_path).is_file():
            Path(entry.local_path).unlink(missing_ok=True)
            if entry.param_filename:
                (self.models_dir / entry.param_filename).unlink(missing_ok=True)
            return True
        return False

    def download_model_sync(
        self,
        model_id: str,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None
    ) -> str:
        """Download model weights (and NCNN .param file if applicable) synchronously."""
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError(f"Model ID '{model_id}' not found in catalog.")

        # Download companion .param file first if specified
        if entry.param_url and entry.param_filename:
            param_dest = self.models_dir / entry.param_filename
            if not param_dest.is_file():
                try:
                    req_p = urllib.request.Request(
                        entry.param_url,
                        headers={"User-Agent": "RemasterGO-Engine/1.0"}
                    )
                    with urllib.request.urlopen(req_p, timeout=30) as resp_p, open(param_dest, "wb") as out_p:
                        out_p.write(resp_p.read())
                except Exception:
                    pass

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

            if self.model.param_url and self.model.param_filename:
                param_dest = self.target_dir / self.model.param_filename
                if not param_dest.is_file():
                    req_p = urllib.request.Request(
                        self.model.param_url,
                        headers={"User-Agent": "RemasterGO-Engine/1.0"}
                    )
                    with urllib.request.urlopen(req_p, timeout=30) as resp_p, open(param_dest, "wb") as out_p:
                        out_p.write(resp_p.read())

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

