"""Hosted image editing on fal.ai, for edit models too large to run well on this Mac.

Unlike the local renderer, this sends the image being edited to fal.ai."""

from __future__ import annotations

import os
from pathlib import Path

from .config import settings
from .errors import RendererError


def is_hosted(model: str) -> bool:
    """fal model ids look like 'fal-ai/qwen-image-edit-2511'."""
    return model.startswith("fal-ai/")


def edit(prompt: str, out_path: Path, seed: int, reference: Path, size: tuple[int, int]) -> None:
    if not os.environ.get("FAL_KEY"):
        raise RendererError("No fal.ai key found. Put FAL_KEY in the .env file and restart.")

    import fal_client
    import httpx

    try:
        result = fal_client.subscribe(
            settings.edit_model,
            arguments={
                "prompt": prompt,
                "image_urls": [fal_client.encode_file(reference)],  # sent inline as a data URI
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
        raise RendererError("fal.ai did not finish the edit within 5 minutes.") from exc
    except (fal_client.client.FalClientError, httpx.HTTPError, KeyError, IndexError) as exc:
        raise RendererError(f"Hosted edit failed: {exc}") from exc
    out_path.write_bytes(image.content)
