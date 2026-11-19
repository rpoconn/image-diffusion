# clothing-designer

A local web app that pairs a hosted LLM with a diffusion model running on your Mac.

- **Claude (Anthropic API)** is the designer. It turns your brief into a garment spec
  (silhouette, fabrics, palette, construction details) and writes the prompt for the image model.
  On revisions it looks at the current render and decides whether to edit it or redraw it.
- **FLUX.2 klein (local, via mflux/MLX)** is the renderer. It draws the design, and edits an
  existing image when Claude asks for a contained change.

## Setup

Needs an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Put your Anthropic API key in `.env`, then:

```bash
uv run clothing-designer
```

Open http://127.0.0.1:8000.

The first render downloads the image model from Hugging Face (about 24 GB for the default
`flux2-klein-4b`) and is slow; later renders reuse the loaded model.

To try the app without downloading a model, set `RENDERER=mock` in `.env`. Claude still
designs; the renderer draws placeholder cards.

## Edit models

By default FLUX.2 klein both draws and edits. For closer likeness when editing a photo, set
`EDIT_MODEL` in `.env`:

- `fal-ai/qwen-image-edit-2511`: Qwen Image Edit hosted on fal.ai. Fast, about 3 cents per
  megapixel, needs `FAL_KEY`. The image being edited is sent to fal.ai.
- `qwen-image-edit-2511`: the same model run locally. Free, but a 58 GB download and very slow
  on a 36 GB Mac.

Designs from a text-only brief are always drawn locally by FLUX.2 klein.

## Drawing on fal.ai instead of this Mac

Set `IMAGE_MODEL=fal-ai/flux-2/klein/4b` (with `FAL_KEY`) to draw new designs on fal.ai: about
two seconds a render and half a cent per megapixel, with no model download. That endpoint only
draws from text, so set `EDIT_MODEL` to a model that edits, such as the hosted Qwen above.

## Cost panel

The panel under the form estimates what the current design has cost so far: Claude tokens plus
hosted renders, updated after every design, revision and redraw. Models that run on this Mac
count as free. The rates live in `src/clothing_designer/pricing.py`; they are list prices, so
update that file when prices change or you add a model, and treat the totals as estimates.

## Settings

All settings live in `.env`; see `.env.example` for the list. The main ones:

| Setting | Default | Notes |
|---|---|---|
| `CLAUDE_MODEL` | `claude-haiku-5-5` | Cheapest. `claude-sonnet-5-5` or `claude-opus-5-5` for stronger design work |
| `IMAGE_MODEL` | `flux2-klein-4b` | `flux2-klein-9b` is larger; non-commercial licence, needs a Hugging Face login |
| `IMAGE_QUANTIZE` | `8` | Lower uses less memory and is a little faster |
| `IMAGE_WIDTH` / `IMAGE_HEIGHT` | `768` / `1024` | Multiples of 16 |

## Layout

```
src/clothing_designer/
  designer.py   Claude: brief + images -> spec + render prompt (structured output)
  renderer.py   local models via mflux (FLUX.2 klein, optional Qwen), plus the mock renderer
  hosted.py     hosted edits on fal.ai
  server.py     FastAPI app and JSON API
  static/       the single-page UI
outputs/        one folder per design: design.json, renders, reference image
```
