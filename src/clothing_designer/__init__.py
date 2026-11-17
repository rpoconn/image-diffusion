"""Clothing Designer: Claude designs garments, a local diffusion model renders them."""


def main() -> None:
    import uvicorn

    from .config import settings

    print(f"Clothing Designer: http://127.0.0.1:{settings.port}")
    uvicorn.run("clothing_designer.server:app", host="127.0.0.1", port=settings.port)
