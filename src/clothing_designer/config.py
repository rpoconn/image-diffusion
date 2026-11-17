"""Settings, read from the environment (and a .env file in the project root)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # Hosted LLM (the "designer")
    claude_model: str = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")

    # Local diffusion model (the "renderer")
    renderer: str = os.environ.get("RENDERER", "mflux")  # "mflux" or "mock"
    image_model: str = os.environ.get("IMAGE_MODEL", "flux2-klein-4b")
    image_quantize: int = _int("IMAGE_QUANTIZE", 8)  # 0 = full precision
    image_steps: int = _int("IMAGE_STEPS", 4)
    image_width: int = _int("IMAGE_WIDTH", 768)
    image_height: int = _int("IMAGE_HEIGHT", 1024)

    # Optional second model used only for edits. "fal-ai/..." is hosted on fal.ai (needs
    # FAL_KEY; the edited image is sent there); "qwen-image-edit-2511" runs locally; blank
    # means the image model above handles edits too. The EDIT_* settings below apply to the
    # local Qwen model only.
    edit_model: str = os.environ.get("EDIT_MODEL", "")
    edit_quantize: int = _int("EDIT_QUANTIZE", 6)
    edit_steps: int = _int("EDIT_STEPS", 8)
    # A few-step add-on (LoRA) as "repo:file"; without it the model needs about 40 steps.
    edit_lora: str = os.environ.get(
        "EDIT_LORA",
        "lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    )
    # Where the compressed copy of the edit model is kept after the first run.
    edit_cache_dir: Path = Path(os.environ.get("EDIT_CACHE_DIR", Path.home() / ".cache" / "clothing-designer"))

    output_dir: Path = Path(os.environ.get("OUTPUT_DIR", "outputs"))
    port: int = _int("PORT", 8000)


settings = Settings()
