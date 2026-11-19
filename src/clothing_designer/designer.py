"""The designer: Claude turns a brief (and optional images) into a garment spec
plus a prompt for the local image model."""

from __future__ import annotations

import base64
import io
from typing import Literal

import anthropic
from PIL import Image
from pydantic import BaseModel, Field

from .config import settings
from .pricing import llm_cost


# Models with server-side refusal fallback. Haiku has none, so it is called without it.
FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2")


class DesignerError(Exception):
    """A failure talking to Claude, with a message fit to show the user."""


class Swatch(BaseModel):
    name: str
    hex: str = Field(description="CSS hex colour, e.g. #1F3A5F")


class DesignSpec(BaseModel):
    name: str = Field(description="Short product name for the piece")
    garment_type: str
    summary: str = Field(description="Two or three sentences describing the design")
    silhouette: str = Field(description="Cut, fit, length and proportions")
    fabrics: list[str]
    palette: list[Swatch]
    details: list[str] = Field(description="Construction details: closures, pockets, seams, trims, hardware")


class DesignTurn(BaseModel):
    spec: DesignSpec
    render_mode: Literal["generate", "edit"] = Field(
        description="'generate' renders from the prompt alone; 'edit' redraws an existing image"
    )
    edit_source: Literal["reference", "current"] = Field(
        description="In edit mode, the image to edit: the client's upload or the latest render"
    )
    render_prompt: str = Field(description="Prompt for the image model")
    note: str = Field(description="One or two sentences to the client about the choices made")


SYSTEM_PROMPT = """\
You are the lead designer in a clothing design studio. A client gives you a brief, and you \
produce a concrete garment design plus a prompt for the studio's image model, which renders it.

The design spec should read like something a pattern maker could start from: a real silhouette, \
named fabrics with weights or finishes, a palette with accurate hex values, and specific \
construction details. Make decisions where the brief is open; the client is hiring your taste. \
Keep everything the client asked for.

The image model is FLUX.2 klein, a diffusion model that runs locally. It responds best to \
natural-language prose, not keyword lists, and it has no negative prompt, so describe what should \
be in the picture and never what should not. It can work in two modes:

- generate: it sees only your prompt. Write one dense paragraph that leads with the garment, then \
covers cut, fabric and how it catches light, exact colours by name, the construction details that \
should be visible, and finally the presentation the client chose (how the garment is shown, \
background, lighting, camera angle). The whole garment should be in frame.
- edit: it sees one existing image plus your prompt, and redraws that image with your change. \
This is the only mode in which the image model sees a picture. In generate mode it never sees \
the client's reference, only your words, so a person, garment or setting described from a photo \
comes out as a different person, garment or setting.

The studio edits whenever there is an image to edit: the client's reference, or an earlier \
render. It never redraws from text something that already exists as a picture, because a redraw \
changes everything the client did not ask to change. Generate is only for the first render of a \
brief that came with no image. The request tells you which mode applies. For an edit, write a \
short, direct instruction that states the change first and then names everything that must stay \
identical: the person's face, hair, skin tone, build and body proportions; each garment and \
accessory with its fit, length and colour; the background with its props, colours and lighting; \
the camera distance and framing where they should not move. The image model keeps what you name \
and is free with what you leave out. When editing the client's reference, keep the look of that \
image; the presentation setting applies only if the brief asks to change how it is shown.

In edit mode, edit_source picks the image to edit. "reference" is the client's original upload; \
"current" is the latest render. Each edit drifts a little from its source, so for a new pose or \
view of the subject in the upload, edit the reference again instead of chaining from a render. \
Use current when the client is building on a change that exists only in the render.

On a revision, return the complete updated spec, not just the changed fields, so it always \
matches what the next render should show.
"""


def _image_block(data: bytes) -> dict:
    """Downscale and re-encode an image as a base64 JPEG content block."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((1568, 1568))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}}


def _ask(content: list[dict]) -> tuple[DesignTurn, dict]:
    """The design, plus what the call cost (see pricing.llm_cost)."""
    request = dict(
        model=settings.claude_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        output_format=DesignTurn,
    )
    try:
        client = anthropic.Anthropic()
        if settings.claude_model.startswith(FALLBACK_MODELS):
            # If one of these models declines a request, the API reruns it on a fallback model.
            response = client.beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
            )
        else:
            response = client.messages.parse(**request)
    except anthropic.AuthenticationError as exc:
        raise DesignerError(
            "Anthropic rejected the credentials. Set ANTHROPIC_API_KEY in the .env file and restart."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise DesignerError("Anthropic rate limit reached. Wait a moment and try again.") from exc
    except anthropic.APIStatusError as exc:
        raise DesignerError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise DesignerError("Could not reach the Anthropic API. Check the network connection.") from exc
    except anthropic.AnthropicError as exc:
        raise DesignerError(f"Anthropic client error: {exc}") from exc
    except TypeError as exc:
        # The SDK raises a bare TypeError when it finds no credentials at all.
        if "authentication method" not in str(exc):
            raise
        raise DesignerError(
            "No Anthropic credentials found. Put ANTHROPIC_API_KEY in the .env file and restart."
        ) from exc

    if response.stop_reason == "refusal":
        raise DesignerError("Claude declined this request. Try rewording the brief.")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise DesignerError("Claude's reply was cut off before the design was complete. Try again.")
    usage = response.usage
    # Cached tokens (none today) are counted at the full input rate; response.model is the
    # model that answered, which differs from the setting if a fallback model stepped in.
    input_tokens = (
        usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    )
    return response.parsed_output, llm_cost(response.model, input_tokens, usage.output_tokens)


def start_design(brief: str, presentation: str, reference: bytes | None = None) -> tuple[DesignTurn, dict]:
    """First turn: brief (and optional reference image) to a design."""
    content: list[dict] = []
    if reference:
        content.append(_image_block(reference))
        image_note = (
            "The attached image is the client's reference. The image model will edit it, so "
            "render_mode is edit with edit_source reference: write render_prompt as an edit instruction."
        )
    else:
        image_note = "There is no reference image, so render_mode must be generate."
    content.append(
        {"type": "text", "text": f"Client brief:\n{brief}\n\nPresentation: {presentation}\n\n{image_note}"}
    )
    turn, cost = _ask(content)
    # Not left to the model: an image is always edited, and text alone is always drawn.
    turn.render_mode = "edit" if reference else "generate"
    if reference:
        turn.edit_source = "reference"
    return turn, cost


def revise_design(
    brief: str,
    presentation: str,
    spec: DesignSpec,
    feedback: str,
    render: bytes | None = None,
    reference: bytes | None = None,
) -> tuple[DesignTurn, dict]:
    """Later turns: the client's feedback on the current design and render."""
    content: list[dict] = []
    notes = []
    if reference:
        content += [{"type": "text", "text": "The client's original reference image:"}, _image_block(reference)]
        notes.append("The client's original reference is attached; edit_source reference edits it.")
    if render:
        content += [{"type": "text", "text": "The current render:"}, _image_block(render)]
        notes.append("The current render is attached; edit_source current edits it.")
    if render or reference:
        notes.append(
            "The image model will edit one of these images, so render_mode is edit: write "
            "render_prompt as an edit instruction that states only the client's change and what stays."
        )
    else:
        notes.append("There is no image yet, so render_mode must be generate.")
    image_note = " ".join(notes)
    content.append(
        {
            "type": "text",
            "text": (
                f"Original brief:\n{brief}\n\nPresentation: {presentation}\n\n"
                f"Current design spec:\n{spec.model_dump_json(indent=2)}\n\n"
                f"{image_note}\n\nClient feedback:\n{feedback}"
            ),
        }
    )
    turn, cost = _ask(content)
    # Not left to the model: a revision edits an existing image whenever there is one.
    turn.render_mode = "edit" if (render or reference) else "generate"
    if not (render and reference):
        turn.edit_source = "current" if render else "reference"
    return turn, cost
