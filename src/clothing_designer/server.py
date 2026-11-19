"""Local web app: brief -> Claude designs -> local model renders -> feedback -> repeat."""

from __future__ import annotations

import io
import json
import math
import random
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .config import settings
from .designer import DesignerError, DesignSpec, DesignTurn, revise_design, start_design
from . import pricing
from .errors import RendererError
from .renderer import make_renderer

STATIC_DIR = Path(__file__).parent / "static"
ID_PATTERN = re.compile(r"^[0-9a-f]{10}$")

settings.output_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clothing Designer")
app.mount("/files", StaticFiles(directory=settings.output_dir), name="files")

renderer = make_renderer()
designs: dict[str, dict] = {}


def _dir(design_id: str) -> Path:
    return settings.output_dir / design_id


def _save(design: dict) -> None:
    (_dir(design["id"]) / "design.json").write_text(json.dumps(design, indent=2))


def _get(design_id: str) -> dict:
    if not ID_PATTERN.match(design_id):
        raise HTTPException(404, "Design not found")
    if design_id not in designs:
        path = _dir(design_id) / "design.json"
        if not path.exists():
            raise HTTPException(404, "Design not found")
        designs[design_id] = json.loads(path.read_text())
    return designs[design_id]


def _add_version(design: dict, turn: DesignTurn, llm_cost: dict, feedback: str | None) -> None:
    design["versions"].append(
        {
            "n": len(design["versions"]) + 1,
            "feedback": feedback,
            "spec": turn.spec.model_dump(),
            "render_mode": turn.render_mode,
            "edit_source": turn.edit_source,
            "render_prompt": turn.render_prompt,
            "note": turn.note,
            "image": None,
            "seed": None,
            # What this version cost: the Claude call, then one entry per render of it.
            "llm_cost": llm_cost,
            "render_costs": [],
        }
    )


def _latest_image(design: dict) -> str | None:
    """The newest rendered image, or the client's upload if nothing is rendered yet."""
    for version in reversed(design["versions"]):
        if version["image"]:
            return version["image"]
    return design["reference"]


def _edit_source(design: dict, version: dict) -> str | None:
    """The image an edit starts from: the client's upload or the latest render."""
    if version.get("edit_source") == "reference" and design["reference"]:
        return design["reference"]
    return _latest_image(design)


def _size_like(path: Path) -> tuple[int, int]:
    """The configured pixel count in the source image's aspect ratio, so an edit keeps the
    framing and proportions of what it edits. The model needs multiples of 16."""
    with Image.open(path) as image:
        ratio = image.width / image.height
    area = settings.image_width * settings.image_height
    height = math.sqrt(area / ratio)
    return max(16, round(height * ratio / 16) * 16), max(16, round(height / 16) * 16)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "claude_model": settings.claude_model,
        "renderer": settings.renderer,
        "image_model": settings.image_model,
        "edit_model": settings.edit_model,
        "rates": pricing.rates(),
    }


@app.post("/api/designs")
def create_design(
    brief: str = Form(...),
    presentation: str = Form("Studio product photo on an invisible mannequin, plain light backdrop"),
    reference: UploadFile | None = File(None),
) -> dict:
    brief = brief.strip()
    if not brief:
        raise HTTPException(400, "Describe what you want designed.")

    reference_bytes = None
    if reference is not None and reference.filename:
        try:
            upload = Image.open(io.BytesIO(reference.file.read())).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(400, "The reference file is not an image this app can read.")
        buffer = io.BytesIO()
        upload.save(buffer, format="PNG")
        reference_bytes = buffer.getvalue()

    try:
        turn, llm_cost = start_design(brief, presentation, reference_bytes)
    except DesignerError as exc:
        raise HTTPException(502, str(exc))

    design_id = uuid.uuid4().hex[:10]
    _dir(design_id).mkdir(parents=True)
    design = {
        "id": design_id,
        "brief": brief,
        "presentation": presentation,
        "reference": None,
        "versions": [],
    }
    if reference_bytes:
        (_dir(design_id) / "reference.png").write_bytes(reference_bytes)
        design["reference"] = "reference.png"
    _add_version(design, turn, llm_cost, feedback=None)
    designs[design_id] = design
    _save(design)
    return design


@app.post("/api/designs/{design_id}/revise")
def revise(design_id: str, feedback: str = Form(...)) -> dict:
    design = _get(design_id)
    feedback = feedback.strip()
    if not feedback:
        raise HTTPException(400, "Say what you would like changed.")

    current = design["versions"][-1]
    render_bytes = reference_bytes = None
    if current["image"]:
        render_bytes = (_dir(design_id) / current["image"]).read_bytes()
    if design["reference"]:
        reference_bytes = (_dir(design_id) / design["reference"]).read_bytes()

    try:
        turn, llm_cost = revise_design(
            brief=design["brief"],
            presentation=design["presentation"],
            spec=DesignSpec.model_validate(current["spec"]),
            feedback=feedback,
            render=render_bytes,
            reference=reference_bytes,
        )
    except DesignerError as exc:
        raise HTTPException(502, str(exc))

    _add_version(design, turn, llm_cost, feedback=feedback)
    _save(design)
    return design


@app.post("/api/designs/{design_id}/render")
def render(design_id: str) -> dict:
    """Render the newest version with the image model. Slow when local: seconds to minutes."""
    design = _get(design_id)
    version = design["versions"][-1]

    reference = size = None
    if version["render_mode"] == "edit":
        source = _edit_source(design, version)
        if source:
            reference = _dir(design_id) / source
            size = _size_like(reference)

    seed = random.randrange(2**31)
    filename = f"v{version['n']}-{seed}.png"
    try:
        renderer.render(version["render_prompt"], _dir(design_id) / filename, seed, reference, size)
    except RendererError as exc:
        raise HTTPException(500, str(exc))

    with Image.open(_dir(design_id) / filename) as image:
        cost = pricing.image_cost(pricing.render_model(is_edit=reference is not None), *image.size)
    version.setdefault("render_costs", []).append(cost)  # designs saved before costs existed
    version["image"] = filename
    version["seed"] = seed
    _save(design)
    return design


@app.get("/api/designs/{design_id}")
def get_design(design_id: str) -> dict:
    return _get(design_id)
