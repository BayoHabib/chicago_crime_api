"""FastAPI application factory and main entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chicago_crime_api.api import api_router
from chicago_crime_api.config import get_settings
from chicago_crime_api.services.prediction import get_prediction_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Chicago Crime Prediction API...")
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")

    # Initialize and load prediction service
    try:
        service = get_prediction_service()
        service.initialize()  # Explicit initialization
        logger.info(f"Model loaded: {service.model_version}")
        logger.info(f"Model info: {service.model_info}")
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        # Continue anyway - service will initialize on first request

    yield

    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
# Chicago Crime Prediction API

Production-grade ML API for predicting crime counts in Chicago.

## Features
- **Single Prediction**: Predict crime count for specific location
- **Batch Prediction**: Predict for multiple locations at once
- **Grid Prediction**: Get crime heatmap for entire city
- **Real-time Inference**: Fast predictions with confidence intervals

## Model
Currently using a Poisson regression model trained on historical Chicago crime data.

## Data Source
Data from Chicago Data Portal via `chicago_crime_data_cli`.
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(api_router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Create app instance
app = create_app()


def main() -> None:
    """Run the application with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "chicago_crime_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
    )


if __name__ == "__main__":
    main()
