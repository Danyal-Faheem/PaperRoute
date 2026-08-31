"""Console entry point for ``paperroute`` and ASGI deployments."""

from .app import app, create_app


def run() -> None:
    import uvicorn
    uvicorn.run("paperroute.app:app", host="0.0.0.0", port=8000, reload=False)


__all__ = ["app", "create_app", "run"]
