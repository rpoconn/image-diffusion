"""API flow with Claude stubbed out and the mock renderer, so no network or model is needed."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from clothing_designer import server
from clothing_designer.config import settings
from clothing_designer.designer import DesignerError, DesignSpec, DesignTurn, Swatch


def make_turn(name: str, mode: str, source: str = "current") -> DesignTurn:
    return DesignTurn(
        spec=DesignSpec(
            name=name,
            garment_type="jacket",
            summary="A test jacket.",
            silhouette="Boxy, hip length.",
            fabrics=["3-layer ripstop nylon"],
            palette=[Swatch(name="Teal", hex="#1F6F78")],
            details=["Two-way front zip"],
        ),
        render_mode=mode,
        edit_source=source,
        render_prompt=f"A {name} on a plain backdrop.",
        note="Kept it simple.",
    )


LLM_COST = {"model": "claude-haiku-5-5", "input_tokens": 2000, "output_tokens": 1000, "usd": 0.0007}


@pytest.fixture
def client(monkeypatch):
    calls = {}

    def fake_start(brief, presentation, reference=None):
        calls["start"] = {"brief": brief, "reference": reference}
        return make_turn("Trail Shell", "edit" if reference else "generate", "reference"), LLM_COST

    def fake_revise(brief, presentation, spec, feedback, render=None, reference=None):
        calls["revise"] = {"spec": spec, "feedback": feedback, "render": render, "reference": reference}
        return make_turn("Trail Shell II", "edit", calls.get("revise_source", "current")), LLM_COST

    monkeypatch.setattr(server, "start_design", fake_start)
    monkeypatch.setattr(server, "revise_design", fake_revise)
    test_client = TestClient(server.app)
    test_client.calls = calls
    return test_client


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_design_render_revise_flow(client):
    design = client.post("/api/designs", data={"brief": "a trail jacket"}).json()
    assert design["versions"][0]["spec"]["name"] == "Trail Shell"
    assert design["versions"][0]["image"] is None

    design = client.post(f"/api/designs/{design['id']}/render").json()
    first_image = design["versions"][0]["image"]
    assert client.get(f"/files/{design['id']}/{first_image}").status_code == 200

    design = client.post(f"/api/designs/{design['id']}/revise", data={"feedback": "taller collar"}).json()
    assert len(design["versions"]) == 2
    assert client.calls["revise"]["render"] is not None  # Claude is shown the current render
    assert client.calls["revise"]["spec"].name == "Trail Shell"

    design = client.post(f"/api/designs/{design['id']}/render").json()
    assert design["versions"][1]["image"] != first_image


def test_reference_image_is_passed_to_claude_and_used_for_edit(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server.renderer, "render",
        lambda prompt, out_path, seed, reference=None, size=None: (
            seen.update(reference=reference, size=size), Image.new("RGB", (8, 8)).save(out_path)
        ),
    )
    design = client.post(
        "/api/designs",
        data={"brief": "make this in corduroy"},
        files={"reference": ("ref.png", png_bytes(), "image/png")},
    ).json()
    assert client.calls["start"]["reference"] is not None
    assert design["reference"] == "reference.png"

    reference_path = settings.output_dir / design["id"] / "reference.png"
    client.post(f"/api/designs/{design['id']}/render")
    assert seen["reference"] == reference_path
    assert seen["size"] == (880, 880)  # square like the upload, about the pixel count of 768x1024

    # A revision shows Claude both images, and can go back to the upload instead of the render.
    client.calls["revise_source"] = "reference"
    client.post(f"/api/designs/{design['id']}/revise", data={"feedback": "now standing"})
    assert client.calls["revise"]["reference"] is not None and client.calls["revise"]["render"] is not None
    client.post(f"/api/designs/{design['id']}/render")
    assert seen["reference"] == reference_path

    client.calls["revise_source"] = "current"
    client.post(f"/api/designs/{design['id']}/revise", data={"feedback": "longer sleeves"})
    client.post(f"/api/designs/{design['id']}/render")
    assert seen["reference"] != reference_path


def test_design_survives_restart(client):
    design = client.post("/api/designs", data={"brief": "a scarf"}).json()
    server.designs.clear()
    assert client.get(f"/api/designs/{design['id']}").json()["brief"] == "a scarf"


def test_errors(client, monkeypatch):
    assert client.post("/api/designs", data={"brief": "   "}).status_code == 400
    assert client.get("/api/designs/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/designs/0123456789").status_code == 404
    bad = client.post("/api/designs", data={"brief": "x"}, files={"reference": ("a.txt", b"not an image", "text/plain")})
    assert bad.status_code == 400

    def boom(*args, **kwargs):
        raise DesignerError("Anthropic rejected the credentials.")

    monkeypatch.setattr(server, "start_design", boom)
    response = client.post("/api/designs", data={"brief": "a hat"})
    assert response.status_code == 502
    assert "credentials" in response.json()["detail"]


def test_hosted_edit_calls_fal_with_the_image_and_saves_the_result(monkeypatch, tmp_path):
    import fal_client
    import httpx

    from clothing_designer import hosted
    from clothing_designer.errors import RendererError

    reference = tmp_path / "ref.png"
    reference.write_bytes(png_bytes())
    monkeypatch.setattr(hosted, "settings", type("S", (), {"edit_model": "fal-ai/qwen-image-edit-2511"})())
    assert hosted.is_hosted("fal-ai/qwen-image-edit-2511") and not hosted.is_hosted("qwen-image-edit-2511")

    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(RendererError, match="FAL_KEY"):
        hosted.edit("p", tmp_path / "out.png", 1, reference, (800, 992))

    sent = {}

    def fake_subscribe(application, arguments, **kwargs):
        sent.update(application=application, **arguments)
        return {"images": [{"url": "https://example.invalid/result.png"}]}

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr(fal_client, "subscribe", fake_subscribe)
    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: httpx.Response(200, content=b"PNGDATA", request=httpx.Request("GET", url)))
    hosted.edit("make it red", tmp_path / "out.png", 7, reference, (800, 992))

    assert sent["application"] == "fal-ai/qwen-image-edit-2511"
    assert sent["prompt"] == "make it red" and sent["seed"] == 7
    assert sent["image_size"] == {"width": 800, "height": 992}
    assert sent["image_urls"][0].startswith("data:image/png;base64,")
    assert (tmp_path / "out.png").read_bytes() == b"PNGDATA"


def test_hosted_generate_calls_fal_without_an_image(monkeypatch, tmp_path):
    import fal_client
    import httpx

    from clothing_designer import hosted

    sent = {}

    def fake_subscribe(application, arguments, **kwargs):
        sent.update(application=application, **arguments)
        return {"images": [{"url": "https://example.invalid/result.png"}]}

    monkeypatch.setattr(hosted, "settings", type("S", (), {"image_model": "fal-ai/flux-2/klein/4b"})())
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr(fal_client, "subscribe", fake_subscribe)
    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: httpx.Response(200, content=b"PNG", request=httpx.Request("GET", url)))
    hosted.generate("a red coat", tmp_path / "out.png", 3, (768, 1024))

    assert sent["application"] == "fal-ai/flux-2/klein/4b" and "image_urls" not in sent
    assert sent["image_size"] == {"width": 768, "height": 1024} and sent["seed"] == 3
    assert (tmp_path / "out.png").read_bytes() == b"PNG"


def test_costs_are_recorded_per_claude_call_and_per_render(client, monkeypatch):
    from clothing_designer import pricing

    design = client.post("/api/designs", data={"brief": "a trail jacket"}).json()
    assert design["versions"][0]["llm_cost"] == LLM_COST and design["versions"][0]["render_costs"] == []

    # The mock renderer is local, so renders are free; drawing again adds a second entry.
    client.post(f"/api/designs/{design['id']}/render")
    design = client.post(f"/api/designs/{design['id']}/render").json()
    renders = design["versions"][0]["render_costs"]
    assert [r["usd"] for r in renders] == [0.0, 0.0] and renders[0]["where"] == "local"

    # A hosted model is charged per megapixel of the image it returned.
    monkeypatch.setattr(pricing, "render_model", lambda is_edit: "fal-ai/flux-2/klein/4b")
    design = client.post(f"/api/designs/{design['id']}/render").json()
    hosted_render = design["versions"][0]["render_costs"][-1]
    assert hosted_render["where"] == "fal.ai"
    assert hosted_render["usd"] == pytest.approx(768 * 1024 / 1e6 * 0.005)

    assert client.get("/api/health").json()["rates"]["llm"]["usd_per_mtok_output"] == 0.50


def test_price_table():
    from clothing_designer import pricing

    assert pricing.llm_cost("claude-haiku-5-5", 2000, 1000)["usd"] == pytest.approx(0.0007)
    assert pricing.claude_rates("claude-sonnet-5-5") == (2.00, 10.00)
    assert pricing.claude_rates("claude-haiku-5-5", input_tokens=150_000) == (0.50, 2.50)
    assert pricing.llm_cost("some-future-model", 10, 10)["usd"] is None
    assert pricing.image_cost("flux2-klein-4b", 768, 1024)["usd"] == 0.0
    assert pricing.image_cost("fal-ai/unlisted", 1000, 1000)["usd"] is None
    assert pricing.image_cost("fal-ai/qwen-image-edit-2511", 1000, 1000)["usd"] == pytest.approx(0.03)


def test_an_existing_image_is_always_edited_never_redrawn(monkeypatch):
    from clothing_designer import designer

    # Claude asks to redraw from text every time; the app only allows that with no image.
    monkeypatch.setattr(designer, "_ask", lambda content: (make_turn("Coat", "generate", "reference"), LLM_COST))
    spec = make_turn("Coat", "generate").spec

    assert designer.start_design("a coat", "studio")[0].render_mode == "generate"
    turn = designer.start_design("this coat in red", "studio", png_bytes())[0]
    assert (turn.render_mode, turn.edit_source) == ("edit", "reference")

    turn = designer.revise_design("a coat", "studio", spec, "longer", render=png_bytes())[0]
    assert (turn.render_mode, turn.edit_source) == ("edit", "current")
    turn = designer.revise_design("a coat", "studio", spec, "longer", render=png_bytes(), reference=png_bytes())[0]
    assert (turn.render_mode, turn.edit_source) == ("edit", "reference")  # Claude's pick when both exist
    assert designer.revise_design("a coat", "studio", spec, "longer")[0].render_mode == "generate"
