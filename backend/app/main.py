"""
Morphometry Analysis API
========================
FastAPI application for scientific cell image analysis.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from app.api.routes import router
from app.core.config import settings, ensure_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("Starting Morphometry Analysis API...")
    ensure_directories()
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Results directory: {settings.results_dir}")

    # Check GPU availability
    try:
        from app.services.segmentation import GPU_AVAILABLE, CELLPOSE_AVAILABLE
        logger.info(f"Cellpose available: {CELLPOSE_AVAILABLE}")
        logger.info(f"GPU available: {GPU_AVAILABLE}")
    except Exception as e:
        logger.warning(f"Could not check GPU status: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Morphometry Analysis API...")


# Create FastAPI app
app = FastAPI(
    title="Morphometry Analysis API",
    description="""
## Scientific Cell Image Analysis

This API provides comprehensive morphometric analysis of microscopy images:

### Features
- **Preprocessing**: Modality-specific image enhancement (brightfield, fluorescence, phase contrast, H&E)
- **Segmentation**: State-of-the-art cell detection using Cellpose, Cellpose-SAM, and classical methods
- **Morphometry**: 30+ quantitative measurements per cell (area, perimeter, shape descriptors, texture)
- **Reports**: Publication-quality PDF, Excel, and JSON reports

### Quick Start
1. Upload an image to `/analyze`
2. Receive cell measurements and report URLs
3. Download reports from `/download/{filename}`

### Supported Image Types
- PNG, JPEG, TIFF (8-bit and 16-bit)
- Grayscale and RGB
- Recommended size: 512x512 to 4096x4096 pixels
    """,
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Analysis"])


# Root endpoint
@app.get("/")
async def root():
    """API root - provides basic information."""
    return {
        "name": "Morphometry Analysis API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/api/v1/health"
    }


# Run with: python -m app.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
