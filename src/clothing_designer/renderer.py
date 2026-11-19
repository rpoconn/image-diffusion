"""The renderer: a diffusion model running locally on Apple Silicon via mflux (MLX)."""

from __future__ import annotations

import textwrap
import threading
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from . import hosted
from .config import settings
from .errors import RendererError


class Renderer(Protocol):
    def render(
        self, prompt: str, out_path: Path, seed: int,
        reference: Path | None = None, size: tuple[int, int] | None = None,
    ) -> None: ...


class MfluxRenderer:
    """Runs the local models through mflux. FLUX.2 klein generates from text. Edits go to
    the model named by EDIT_MODEL (Qwen Image Edit) when set, otherwise to klein as well.
    Only one model family is held in memory at a time; switching unloads the other.
    An IMAGE_MODEL or EDIT_MODEL hosted on fal.ai ("fal-ai/...") is called over the network
    instead of loaded here."""

    def __init__(self) -> None:
        self._loaded: str | None = None  # "flux" or "qwen"
        self._generate = None
        self._edit = None
        self._lock = threading.Lock()  # one generation at a time; the GPU is shared

    def _unload(self) -> None:
        import gc

        import mlx.core as mx

        self._generate = self._edit = self._loaded = None
        gc.collect()
        mx.clear_cache()

    def _load_flux(self) -> None:
        if not settings.image_model.startswith("flux2-"):
            raise RendererError(
                f"IMAGE_MODEL={settings.image_model!r} is not supported; use a FLUX.2 model such as "
                "flux2-klein-4b or flux2-klein-9b."
            )
        # Imported here so the web server starts instantly and the mock renderer needs no MLX.
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
        from mlx import nn

        try:
            generate = Flux2Klein(
                model_config=ModelConfig.from_name(settings.image_model),
                quantize=settings.image_quantize or None,
            )
        except Exception as exc:  # download, auth and weight-loading failures all surface here
            raise RendererError(f"Could not load {settings.image_model}: {exc}") from exc

        # mflux has separate classes for generating and editing, but both are set up by the
        # same initializer from the same weights. Building the edit pipeline around the
        # already-loaded networks avoids holding a second copy of the model in memory.
        edit = Flux2KleinEdit.__new__(Flux2KleinEdit)
        nn.Module.__init__(edit)
        dict.update(edit, generate)  # networks (an MLX module keeps these as dict items)
        vars(edit).update(vars(generate))  # config, tokenizers, callbacks
        self._generate, self._edit, self._loaded = generate, edit, "flux"

    def _load_qwen(self) -> None:
        if not settings.edit_model.startswith("qwen-image-edit"):
            raise RendererError(
                f"EDIT_MODEL={settings.edit_model!r} is not supported; use qwen-image-edit-2511 or leave it blank."
            )
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        config = ModelConfig.from_name(settings.edit_model)
        lora = [settings.edit_lora] if settings.edit_lora else None
        try:
            saved = None
            if settings.edit_quantize:
                # The original weights are about 58 GB and compressing them takes minutes, so
                # the compressed copy is written to disk once and loaded from there afterwards.
                saved = settings.edit_cache_dir / f"{settings.edit_model}-q{settings.edit_quantize}"
                if not saved.exists():
                    print(f"Compressing {settings.edit_model} to {settings.edit_quantize}-bit (one-off)...", flush=True)
                    model = QwenImageEdit(model_config=config, quantize=settings.edit_quantize)
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    model.save_model(str(saved))
                    del model
                    self._unload()
            self._edit = QwenImageEdit(
                model_config=config,
                model_path=str(saved) if saved else None,
                lora_paths=lora,
                lora_scales=[1.0] if lora else None,
                # Merging the add-on into the weights makes mflux store them at 8-bit again,
                # which undoes the compression and overflows memory. Keep it separate.
                bake_lora=False,
            )
        except Exception as exc:
            raise RendererError(f"Could not load {settings.edit_model}: {exc}") from exc
        self._loaded = "qwen"

    def render(
        self, prompt: str, out_path: Path, seed: int,
        reference: Path | None = None, size: tuple[int, int] | None = None,
    ) -> None:
        width, height = size or (settings.image_width, settings.image_height)
        model = settings.edit_model or settings.image_model if reference else settings.image_model
        if hosted.is_hosted(model):
            if not reference:
                hosted.generate(prompt, out_path, seed, (width, height))
            elif hosted.is_hosted(settings.edit_model):
                hosted.edit(prompt, out_path, seed, reference, (width, height))
            else:
                raise RendererError(
                    f"IMAGE_MODEL={settings.image_model!r} is hosted and only draws from text. "
                    "Set EDIT_MODEL to a model that can edit an image."
                )
            return
        family = "qwen" if reference and settings.edit_model else "flux"
        with self._lock:
            if self._loaded != family:
                self._unload()
                self._load_qwen() if family == "qwen" else self._load_flux()
            try:
                if family == "qwen":
                    image = self._edit.generate_image(
                        seed=seed,
                        prompt=prompt,
                        image_paths=[str(reference)],
                        image_path=str(reference),
                        num_inference_steps=settings.edit_steps,
                        width=width,
                        height=height,
                        # The few-step add-on is trained to run without guidance.
                        guidance=1.0 if settings.edit_lora else 4.0,
                    )
                else:
                    options = dict(
                        seed=seed,
                        prompt=prompt,
                        num_inference_steps=settings.image_steps,
                        width=width,
                        height=height,
                    )
                    if reference:
                        image = self._edit.generate_image(image_paths=[reference], **options)
                    else:
                        image = self._generate.generate_image(**options)
                image.save(path=str(out_path))
            except Exception as exc:
                raise RendererError(f"Image generation failed: {exc}") from exc


class MockRenderer:
    """Draws a placeholder card instead of running a model. Set RENDERER=mock to try the
    app without downloading any weights."""

    def render(
        self, prompt: str, out_path: Path, seed: int,
        reference: Path | None = None, size: tuple[int, int] | None = None,
    ) -> None:
        width, height = size or (settings.image_width, settings.image_height)
        hue = (seed * 47) % 360
        image = Image.new("HSV", (width, height), (int(hue / 360 * 255), 60, 235)).convert("RGB")
        if reference:
            thumb = Image.open(reference).convert("RGB")
            thumb.thumbnail((width // 3, height // 3))
            image.paste(thumb, (width - thumb.width - 24, 24))
        draw = ImageDraw.Draw(image)
        mode = "EDIT of previous image" if reference else "GENERATE"
        text = f"MOCK RENDER ({mode})\nseed {seed}\n\n" + "\n".join(textwrap.wrap(prompt, 60))
        draw.multiline_text((24, 24), text, fill=(30, 30, 30), spacing=6)
        image.save(out_path)


def make_renderer() -> Renderer:
    if settings.renderer == "mock":
        return MockRenderer()
    if settings.renderer == "mflux":
        return MfluxRenderer()
    raise RendererError(f"RENDERER={settings.renderer!r} is not supported; use 'mflux' or 'mock'.")
