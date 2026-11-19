"""Hosted image models on fal.ai: drawing from a prompt, and editing an existing image.

Unlike the local renderer, this sends the prompt (and any image being edited) to fal.ai."""

from __future__ import annotations

import os
from pathlib import Path

from .config import settings
from .errors import RendererError


def is_hosted(model: str) -> bool:
    """fal model ids look like 'fal-ai/qwen-image-edit-2511'."""
    return model.startswith("fal-ai/")


def _run(model: str, arguments: dict, out_path: Path, seed: int, size: tuple[int, int]) -> None:
    if not os.environ.get("FAL_KEY"):
        raise RendererError("No fal.ai key found. Put FAL_KEY in the .env file and restart.")

    import fal_client
    import httpx

    try:
        result = fal_client.subscribe(
            model,
            arguments={
                **arguments,
                "image_size": {"width": size[0], "height": size[1]},
                "seed": seed,
                "num_images": 1,
                "output_format": "png",
            },
            client_timeout=300,
        )
        image = httpx.get(result["images"][0]["url"], timeout=120, follow_redirects=True)
        image.raise_for_status()
    except fal_client.client.FalClientHTTPError as exc:
        raise RendererError(f"fal.ai rejected the request ({exc.status_code}): {exc.message}") from exc
    except fal_client.client.FalClientTimeoutError as exc:
        raise RendererError("fal.ai did not finish the image within 5 minutes.") from exc
    except (fal_client.client.FalClientError, httpx.HTTPError, KeyError, IndexError) as exc:
        raise RendererError(f"Hosted render failed: {exc}") from exc
    out_path.write_bytes(image.content)


def generate(prompt: str, out_path: Path, seed: int, size: tuple[int, int]) -> None:
    _run(settings.image_model, {"prompt": prompt}, out_path, seed, size)


def edit(prompt: str, out_path: Path, seed: int, reference: Path, size: tuple[int, int]) -> None:
    import fal_client

    # The image is sent inline as a data URI.
    arguments = {"prompt": prompt, "image_urls": [fal_client.encode_file(reference)]}
    _run(settings.edit_model, arguments, out_path, seed, size)
