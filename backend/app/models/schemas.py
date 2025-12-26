"""
Pydantic Schemas
================
Request and response models for the API.
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


# =========================================================================
# REQUEST MODELS
# =========================================================================

class PreprocessingOptions(BaseModel):
    """Options for image preprocessing."""
    modality: Literal["brightfield", "fluorescence", "phase_contrast", "he_stained"] = "brightfield"
    custom_pipeline: list[str] | None = None
    denoise_strength: float = Field(default=10.0, ge=0, le=50)
    contrast_clip_limit: float = Field(default=2.0, ge=0.5, le=10)


class SegmentationOptions(BaseModel):
    """Options for cell segmentation."""
    model: Literal["cyto3", "nuclei", "cyto2", "cpsam", "watershed", "otsu"] = "cyto3"
    diameter: float | None = Field(default=None, ge=1, le=500)
    flow_threshold: float = Field(default=0.4, ge=0, le=1)
    cellprob_threshold: float = Field(default=0.0, ge=-6, le=6)
    min_size: int = Field(default=15, ge=1, le=1000)


class AnalysisRequest(BaseModel):
    """Complete analysis request with all options."""
    preprocessing: PreprocessingOptions = PreprocessingOptions()
    segmentation: SegmentationOptions = SegmentationOptions()
    pixel_size_um: float = Field(default=1.0, ge=0.001, le=100, description="Pixel size in microns")
    compute_texture: bool = True
    generate_report: bool = True
    report_format: Literal["pdf", "excel", "json", "all"] = "pdf"


# =========================================================================
# RESPONSE MODELS
# =========================================================================

class CellMeasurementResponse(BaseModel):
    """Single cell measurement."""
    cell_id: int
    area: float
    perimeter: float
    circularity: float
    eccentricity: float
    solidity: float
    aspect_ratio: float
    centroid_x: float
    centroid_y: float
    intensity_mean: float | None = None


class SummaryStatistics(BaseModel):
    """Summary statistics for a metric."""
    mean: float
    std: float
    min: float
    max: float
    median: float


class AnalysisSummary(BaseModel):
    """Summary of analysis results."""
    total_cells: int
    cell_density: float  # cells per mm²
    area_stats: SummaryStatistics | None = None
    circularity_stats: SummaryStatistics | None = None
    processing_time_seconds: float


class CalibrationWarning(BaseModel):
    """Warning about calibration uncertainty."""
    level: Literal["info", "warning", "critical"]
    code: str
    message: str
    details: str
    recommendation: str


class CalibrationInfo(BaseModel):
    """Calibration information and uncertainty warnings."""
    status: Literal["ok", "warning", "critical"]
    source: str  # "preset", "stage_micrometer", "user_provided", etc.
    is_calibrated: bool
    pixel_size_um: float
    objective: str | None = None
    uncertainty_linear_percent: float
    uncertainty_area_percent: float
    warnings: list[CalibrationWarning] = []


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    success: bool
    message: str
    analysis_id: str
    summary: AnalysisSummary
    measurements: list[CellMeasurementResponse]
    calibration: CalibrationInfo | None = None  # Calibration info with warnings
    report_urls: dict[str, str] | None = None  # URLs to download reports
    mask_url: str | None = None  # URL to download segmentation mask
    overlay_url: str | None = None  # URL to download overlay image


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    gpu_available: bool
    cellpose_available: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: str | None = None


# =========================================================================
# BATCH PROCESSING
# =========================================================================

class BatchAnalysisRequest(BaseModel):
    """Request for batch processing multiple images."""
    options: AnalysisRequest = AnalysisRequest()
    parallel: bool = True  # Process images in parallel


class BatchAnalysisResponse(BaseModel):
    """Response for batch processing."""
    success: bool
    total_images: int
    processed: int
    failed: int
    results: list[AnalysisResponse]
