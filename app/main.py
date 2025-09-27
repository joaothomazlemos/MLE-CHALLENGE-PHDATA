"""
FastAPI application for house price prediction.

This is the main entry point for the Sound Realty house price prediction API.
It provides endpoints for predicting house prices based on property characteristics
and demographic data.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.internal.config import get_config
from app.routers import prediction_router

config = get_config()

app = FastAPI(
    title=config.title,
    description=config.description,
    version=config.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix="/api/v1", tags=["predictions"])


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": config.title,
        "version": config.version,
        "docs": "/docs",
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "house-price-prediction"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)
