"""
FastAPI application factory.

Uses the factory pattern so tests can create isolated app instances
with different configurations (e.g. test database).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.router import api_router
from app.config import get_settings

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic.

    Runs before the first request and after the last request.
    Use this for database connections, cache warmup, etc.
    """
    settings = get_settings()
    print(f"[Decisio] Starting {settings.app_name} (debug={settings.debug})")

    # In local dev with SQLite, create tables directly if they don't exist.
    # In production, use Alembic migrations exclusively.
    if settings.is_sqlite:
        from app.database import engine
        from app.models.base import Base
        # Import all models so Base.metadata knows about them
        import app.models  # noqa: F401

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[Decisio] SQLite tables created (dev mode)")

    yield  # App is running and serving requests

    print(f"[Decisio] Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Engineering Decision Capture & Intelligence Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # ── CORS ─────────────────────────────────────────────
    # Allow VS Code extension and local dev frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",   # Dev frontend
            "http://localhost:5173",   # Vite dev server
            "vscode-webview://*",      # VS Code extension webviews
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Interactive Testing UI / Playground ──────────────
    @app.get("/", response_class=HTMLResponse, tags=["Playground"])
    @app.get("/playground", response_class=HTMLResponse, tags=["Playground"])
    async def get_playground():
        html_file = STATIC_DIR / "index.html"
        if html_file.exists():
            return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Decisio Playground</h1><p>Static index.html not found.</p>")

    # ── Health check ─────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": "0.1.0",
        }

    # ── Route registration ───────────────────────────────
    app.include_router(api_router, prefix=settings.api_prefix)

    return app


# The app instance used by `uvicorn app.main:app`
app = create_app()
