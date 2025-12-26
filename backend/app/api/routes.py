"""
API Routes
==========
FastAPI endpoints for morphometry analysis.
"""

import uuid
import io
import base64
from pathlib import Path
from datetime import datetime
import time

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib import cm
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from app.models.schemas import (
    AnalysisRequest, AnalysisResponse, AnalysisSummary,
    CellMeasurementResponse, SummaryStatistics,
    PreprocessingOptions, SegmentationOptions,
    HealthResponse, ErrorResponse, BatchAnalysisResponse,
    CalibrationInfo, CalibrationWarning
)
from app.core.config import settings, ensure_directories
from app.core.preprocessing import ImagePreprocessor, ImageModality
from app.core.calibration import MicroscopeCalibration, get_available_objectives
from app.core.quality import ImageQualityChecker, quick_quality_check
from app.services.segmentation import CellSegmenter, CELLPOSE_AVAILABLE, GPU_AVAILABLE
from app.services.morphometry import MorphometryAnalyzer, RBCFilter
from app.services.report import ReportGenerator, ReportMetadata

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def create_overlay_image(original_image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Create an overlay visualization showing detected cells on the original image.

    Args:
        original_image: Original grayscale or RGB image
        masks: Label array where each cell has a unique integer ID

    Returns:
        RGB image array with colored cell boundaries overlaid
    """
    # Ensure image is RGB
    if len(original_image.shape) == 2:
        rgb_image = np.stack([original_image] * 3, axis=-1)
    elif original_image.shape[-1] == 1:
        rgb_image = np.repeat(original_image, 3, axis=-1)
    else:
        rgb_image = original_image.copy()

    # Normalize to 0-255 if needed
    if rgb_image.max() <= 1.0:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    else:
        rgb_image = rgb_image.astype(np.uint8)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb_image)

    # Create colored mask overlay
    if masks.max() > 0:
        # Use a colormap for different cells
        colored_masks = np.zeros((*masks.shape, 4), dtype=np.float32)
        num_cells = masks.max()
        colors = cm.tab20(np.linspace(0, 1, min(num_cells, 20)))

        for cell_id in range(1, num_cells + 1):
            cell_mask = masks == cell_id
            color_idx = (cell_id - 1) % 20
            colored_masks[cell_mask] = [*colors[color_idx][:3], 0.4]  # 40% opacity

        ax.imshow(colored_masks)

        # Draw cell boundaries
        from skimage import segmentation
        boundaries = segmentation.find_boundaries(masks, mode='outer')
        boundary_overlay = np.zeros((*masks.shape, 4), dtype=np.float32)
        boundary_overlay[boundaries] = [1, 1, 0, 0.8]  # Yellow boundaries
        ax.imshow(boundary_overlay)

    ax.axis('off')
    ax.set_title(f'Cell Detection Overlay ({masks.max()} cells detected)', fontsize=12, pad=10)

    # Save to buffer
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    return buf


# =========================================================================
# HEALTH CHECK
# =========================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and available features."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        gpu_available=GPU_AVAILABLE,
        cellpose_available=CELLPOSE_AVAILABLE,
        timestamp=datetime.now()
    )


# =========================================================================
# IMAGE ANALYSIS
# =========================================================================

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    modality: str = "auto",
    model: str = "auto",
    diameter: float | None = None,
    objective: str | None = None,
    pixel_size_um: float = 1.0,
    compute_texture: bool = True,
    check_quality: bool = True,
    exclude_rbc: bool = False,
    generate_report: bool = True,
    report_format: str = "pdf"
):
    """
    Analyze a single microscopy image with full calibration and quality control.

    This endpoint performs:
    1. Quality assessment (optional) - checks focus, tissue content, artifacts
    2. Image preprocessing based on modality (auto-detected if not specified)
    3. Cell segmentation using the specified model (auto-selected based on modality)
    4. RBC exclusion (optional) - removes red blood cells from analysis
    5. Morphometric measurements with proper calibration (µm, µm²)
    6. Optional report generation

    Args:
        file: Uploaded image file (PNG, JPEG, TIFF)
        modality: Image type (auto, brightfield, fluorescence, phase_contrast, he_stained)
        model: Segmentation model (auto, cyto3, nuclei, cpsam, watershed, otsu)
        diameter: Expected cell diameter in pixels (None = auto-detect)
        objective: Microscope objective (4x, 10x, 20x, 40x, 60x, 100x) - overrides pixel_size_um
        pixel_size_um: Pixel size in microns (used if objective not specified)
        compute_texture: Whether to compute GLCM texture features
        check_quality: Whether to perform quality assessment before analysis
        exclude_rbc: Whether to exclude red blood cells (for H&E histology)
        generate_report: Whether to generate a report
        report_format: Report format (pdf, excel, json, all)

    Returns:
        AnalysisResponse with measurements and report URLs
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)

        logger.info(f"Processing image: {file.filename}, shape: {image_array.shape}")

        # Ensure directories exist
        ensure_directories()

        # =================================================================
        # 0. QUALITY CHECK (optional)
        # =================================================================
        quality_report = None
        if check_quality:
            quality_checker = ImageQualityChecker(is_histology=(modality == "he_stained" or modality == "auto"))
            quality_report = quality_checker.assess(image_array)
            logger.info(f"Quality assessment: {quality_report.quality_level.value} "
                       f"(score: {quality_report.overall_quality:.2f})")
            if not quality_report.is_usable:
                logger.warning(f"Image quality issues: {quality_report.issues}")

        # =================================================================
        # 1. CALIBRATION SETUP
        # =================================================================
        if objective:
            calibration = MicroscopeCalibration.from_objective(objective)
            actual_pixel_size = calibration.pixel_size_um
            logger.info(f"Using objective preset: {objective} ({actual_pixel_size} µm/px)")
        else:
            calibration = MicroscopeCalibration.from_pixel_size(pixel_size_um)
            actual_pixel_size = pixel_size_um
            logger.info(f"Using custom pixel size: {actual_pixel_size} µm/px")

        # Log calibration warnings
        for warning in calibration.get_warnings():
            if warning.level == "warning":
                logger.warning(f"[{warning.code}] {warning.message}")
            elif warning.level == "critical":
                logger.error(f"[{warning.code}] {warning.message}")

        # =================================================================
        # 2. PREPROCESSING (with auto-detection if modality="auto")
        # =================================================================
        preprocessor = ImagePreprocessor()
        preprocess_result = preprocessor.process(
            image_array,
            modality=ImageModality(modality)
        )
        processed_image = preprocess_result.image

        # Get the actual modality used (may have been auto-detected)
        actual_modality = preprocess_result.modality

        # Log detection results
        if preprocess_result.detected_modality:
            logger.info(f"Auto-detected modality: {actual_modality.value} "
                       f"(confidence: {preprocess_result.detection_confidence:.2f})")

        # =================================================================
        # 3. AUTO-SELECT SEGMENTATION MODEL
        # =================================================================
        actual_model = model
        actual_diameter = diameter

        if model == "auto":
            model_by_modality = {
                ImageModality.HE_STAINED: "nuclei",
                ImageModality.FLUORESCENCE: "cyto3",
                ImageModality.BRIGHTFIELD: "cyto3",
                ImageModality.PHASE_CONTRAST: "cyto3",
            }
            actual_model = model_by_modality.get(actual_modality, "cyto3")
            logger.info(f"Auto-selected model: {actual_model} for {actual_modality.value}")

        # Auto-select diameter based on calibration and modality
        if actual_diameter is None:
            if actual_modality == ImageModality.HE_STAINED:
                # Use calibration to get nucleus diameter in pixels
                actual_diameter = calibration.get_nucleus_diameter_pixels()
            else:
                actual_diameter = calibration.get_expected_cell_diameter_pixels()
            logger.info(f"Auto-selected diameter: {actual_diameter:.1f} px")

        # =================================================================
        # 4. SEGMENTATION
        # =================================================================
        segmenter = CellSegmenter(
            model=actual_model,
            diameter=actual_diameter,
            use_gpu=True
        )
        seg_result = segmenter.segment(processed_image)

        if seg_result.cell_count == 0:
            return AnalysisResponse(
                success=True,
                message="Analysis complete, but no cells were detected",
                analysis_id=analysis_id,
                summary=AnalysisSummary(
                    total_cells=0,
                    cell_density=0.0,
                    processing_time_seconds=time.time() - start_time
                ),
                measurements=[]
            )

        # =================================================================
        # 5. MORPHOMETRY with proper calibration
        # =================================================================
        analyzer = MorphometryAnalyzer(
            pixel_size_um=actual_pixel_size,
            objective=objective,
            compute_texture=compute_texture
        )
        morph_result = analyzer.analyze(seg_result.masks, image_array)

        # =================================================================
        # 6. RBC EXCLUSION (optional, for histology)
        # =================================================================
        rbc_stats = None
        if exclude_rbc and actual_modality == ImageModality.HE_STAINED:
            rbc_filter = RBCFilter(calibration=calibration)
            non_rbc_cells, rbc_cells, rbc_stats = rbc_filter.filter_rbcs(
                morph_result.measurements,
                color_image=image_array,
                mask=seg_result.masks
            )
            # Replace measurements with filtered list
            morph_result.measurements = non_rbc_cells
            logger.info(f"RBC exclusion: removed {rbc_stats['rbc_count']} RBCs "
                       f"({rbc_stats['rbc_percentage']:.1f}%)")

        df = morph_result.to_dataframe()

        # 4. Prepare response
        measurements = []
        for _, row in df.iterrows():
            measurements.append(CellMeasurementResponse(
                cell_id=int(row['cell_id']),
                area=float(row['area']),
                perimeter=float(row['perimeter']),
                circularity=float(row['circularity']),
                eccentricity=float(row['eccentricity']),
                solidity=float(row['solidity']),
                aspect_ratio=float(row['aspect_ratio']),
                centroid_x=float(row['centroid_col']),
                centroid_y=float(row['centroid_row']),
                intensity_mean=float(row['intensity_mean']) if 'intensity_mean' in row else None
            ))

        # Build summary statistics
        area_values = df['area'].dropna()
        circ_values = df['circularity'].dropna()

        summary = AnalysisSummary(
            total_cells=morph_result.total_cells,
            cell_density=morph_result.cell_density * 1e6,  # Convert to cells/mm²
            area_stats=SummaryStatistics(
                mean=float(area_values.mean()),
                std=float(area_values.std()),
                min=float(area_values.min()),
                max=float(area_values.max()),
                median=float(area_values.median())
            ) if len(area_values) > 0 else None,
            circularity_stats=SummaryStatistics(
                mean=float(circ_values.mean()),
                std=float(circ_values.std()),
                min=float(circ_values.min()),
                max=float(circ_values.max()),
                median=float(circ_values.median())
            ) if len(circ_values) > 0 else None,
            processing_time_seconds=time.time() - start_time
        )

        # 5. Generate reports
        report_urls = {}
        if generate_report:
            report_gen = ReportGenerator()
            metadata = ReportMetadata(
                title="Morphometry Analysis Report",
                image_filename=file.filename or "uploaded_image",
                segmentation_model=actual_model,
                pixel_size_um=pixel_size_um,
                preprocessing_steps=preprocess_result.applied_filters
            )

            base_path = settings.results_dir / analysis_id

            if report_format in ["pdf", "all"]:
                pdf_path = base_path.with_suffix('.pdf')
                report_gen.generate_pdf(
                    df=df,
                    output_path=pdf_path,
                    metadata=metadata,
                    original_image=image_array,
                    masks=seg_result.masks
                )
                report_urls['pdf'] = f"/download/{analysis_id}.pdf"

            if report_format in ["excel", "all"]:
                excel_path = base_path.with_suffix('.xlsx')
                report_gen.generate_excel(
                    df=df,
                    output_path=excel_path,
                    metadata=metadata,
                    summary_stats=morph_result.summary_stats
                )
                report_urls['excel'] = f"/download/{analysis_id}.xlsx"

            if report_format in ["json", "all"]:
                json_path = base_path.with_suffix('.json')
                report_gen.generate_json(
                    df=df,
                    output_path=json_path,
                    metadata=metadata,
                    summary_stats=morph_result.summary_stats
                )
                report_urls['json'] = f"/download/{analysis_id}.json"

        # Save mask for download
        mask_path = settings.results_dir / f"{analysis_id}_mask.png"
        mask_image = Image.fromarray((seg_result.masks > 0).astype(np.uint8) * 255)
        mask_image.save(mask_path)

        # Generate and save overlay visualization
        overlay_buffer = create_overlay_image(image_array, seg_result.masks)
        overlay_path = settings.results_dir / f"{analysis_id}_overlay.png"
        with open(overlay_path, 'wb') as f:
            f.write(overlay_buffer.getvalue())

        # Build calibration info with warnings
        calib_summary = calibration.get_warning_summary()
        calibration_info = CalibrationInfo(
            status=calib_summary["status"],
            source=calib_summary["source"],
            is_calibrated=calib_summary["is_calibrated"],
            pixel_size_um=calib_summary["pixel_size_um"],
            objective=calibration.objective if calibration.objective != "custom" else None,
            uncertainty_linear_percent=calib_summary["uncertainty_linear_percent"],
            uncertainty_area_percent=calib_summary["uncertainty_area_percent"],
            warnings=[
                CalibrationWarning(
                    level=w["level"],
                    code=w["code"],
                    message=w["message"],
                    details=w["details"],
                    recommendation=w["recommendation"]
                )
                for w in calib_summary["warnings"]
            ]
        )

        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed {morph_result.total_cells} cells",
            analysis_id=analysis_id,
            summary=summary,
            measurements=measurements,
            calibration=calibration_info,
            report_urls=report_urls if report_urls else None,
            mask_url=f"/download/{analysis_id}_mask.png",
            overlay_url=f"/download/{analysis_id}_overlay.png"
        )

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# PREPROCESSING ONLY
# =========================================================================

@router.post("/preprocess")
async def preprocess_image(
    file: UploadFile = File(...),
    modality: str = "brightfield"
):
    """
    Preprocess an image without segmentation.

    Returns the processed image as base64-encoded PNG.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)

        preprocessor = ImagePreprocessor()
        result = preprocessor.process(image_array, modality=ImageModality(modality))

        # Convert to PNG
        processed_image = Image.fromarray(result.image.astype(np.uint8))
        buffer = io.BytesIO()
        processed_image.save(buffer, format='PNG')
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "X-Applied-Filters": ",".join(result.applied_filters)
            }
        )

    except Exception as e:
        logger.exception(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# FILE DOWNLOAD
# =========================================================================

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated report or result file."""
    file_path = settings.results_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        '.pdf': 'application/pdf',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.json': 'application/json',
        '.png': 'image/png',
        '.tiff': 'image/tiff'
    }

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_types.get(suffix, 'application/octet-stream')
    )


# =========================================================================
# MODEL INFORMATION
# =========================================================================

@router.get("/models")
async def list_models():
    """List available segmentation models."""
    models = [
        {
            "id": "auto",
            "name": "Auto-Select",
            "description": "Automatically select best model based on image type (nuclei for H&E, cyto3 for others)",
            "recommended": True
        },
        {
            "id": "cyto3",
            "name": "Cellpose Cyto3",
            "description": "Latest cytoplasm model, best for most cell types",
            "recommended": False
        },
        {
            "id": "nuclei",
            "name": "Cellpose Nuclei",
            "description": "Specialized for nuclei detection (best for H&E histology)",
            "recommended": False
        },
        {
            "id": "cpsam",
            "name": "Cellpose-SAM",
            "description": "Superhuman generalization, zero-shot for new cell types",
            "recommended": False
        },
        {
            "id": "cyto2",
            "name": "Cellpose Cyto2",
            "description": "Legacy cytoplasm model",
            "recommended": False
        },
        {
            "id": "watershed",
            "name": "Watershed",
            "description": "Classical method, no GPU required",
            "recommended": False
        },
        {
            "id": "otsu",
            "name": "Otsu Thresholding",
            "description": "Simple thresholding, fastest option",
            "recommended": False
        }
    ]

    return {
        "cellpose_available": CELLPOSE_AVAILABLE,
        "gpu_available": GPU_AVAILABLE,
        "models": models
    }


@router.get("/objectives")
async def list_objectives():
    """List available microscope objective presets for calibration."""
    return {
        "objectives": get_available_objectives(),
        "note": "Select objective for automatic pixel size calibration. "
                "For custom calibration, use pixel_size_um parameter instead."
    }


@router.get("/modalities")
async def list_modalities():
    """List supported image modalities."""
    return {
        "modalities": [
            {
                "id": "auto",
                "name": "Auto-Detect",
                "description": "Automatically detect image type (H&E, fluorescence, etc.)",
                "recommended": True
            },
            {
                "id": "he_stained",
                "name": "H&E Stained",
                "description": "Histology slides with Hematoxylin & Eosin staining"
            },
            {
                "id": "brightfield",
                "name": "Brightfield",
                "description": "Standard light microscopy with transmitted light"
            },
            {
                "id": "fluorescence",
                "name": "Fluorescence",
                "description": "Labeled cells with fluorescent markers (DAPI, GFP, etc.)"
            },
            {
                "id": "phase_contrast",
                "name": "Phase Contrast",
                "description": "Unstained cells with enhanced contrast"
            }
        ]
    }
