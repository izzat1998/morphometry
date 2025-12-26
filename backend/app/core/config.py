"""
Application Configuration
=========================
Centralized settings for the morphometry analysis pipeline.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Morphometry Analysis"
    debug: bool = False

    # File Storage
    upload_dir: Path = Path("./uploads")
    results_dir: Path = Path("./results")
    max_file_size_mb: int = 100
    allowed_extensions: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # Cellpose Model Settings
    cellpose_model: Literal["cyto3", "nuclei", "cyto2", "cyto"] = "cyto3"
    cellpose_gpu: bool = True
    cellpose_diameter: float | None = None  # None = auto-detect

    # Preprocessing Defaults
    default_denoise_strength: float = 10.0
    default_contrast_clip_limit: float = 2.0

    # Report Generation
    report_format: Literal["pdf", "excel", "both"] = "both"

    class Config:
        env_file = ".env"
        env_prefix = "MORPH_"


settings = Settings()


def ensure_directories():
    """Create necessary directories if they don't exist."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
