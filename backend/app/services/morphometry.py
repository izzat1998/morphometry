"""
Morphometry Analysis Service
============================
Comprehensive morphometric measurements for cell analysis.

This module calculates quantitative measurements from segmented cells:
- Size metrics: area, perimeter, diameter
- Shape descriptors: circularity, eccentricity, solidity
- Intensity features: mean, std, min, max
- Texture features: entropy, homogeneity
- Spatial metrics: centroid, orientation

Key Insight:
    Morphometric measurements are the scientific output of your analysis.
    The accuracy of these measurements depends on:
    1. Image resolution (pixels per micron) - USE PROPER CALIBRATION!
    2. Segmentation quality
    3. Proper calibration

IMPORTANT: Without proper calibration, all area/perimeter measurements
are in arbitrary pixel units and cannot be compared across magnifications!
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional
from skimage import measure, feature
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage, stats
import logging

from ..core.calibration import MicroscopeCalibration, get_available_objectives, CELL_SIZE_REFERENCE

logger = logging.getLogger(__name__)


# =============================================================================
# RBC (RED BLOOD CELL) DETECTION AND EXCLUSION
# =============================================================================

class RBCFilter:
    """
    Filter to exclude Red Blood Cells from morphometric analysis.

    RBCs in H&E stained histology have characteristic properties:
    - Small size (~7 µm diameter)
    - Circular/disc shape (high circularity)
    - High eosin staining (pink/red color)
    - No visible nucleus

    This filter identifies and excludes RBCs based on these properties.
    """

    # RBC characteristics
    RBC_DIAMETER_UM = 7.0        # Human RBC is ~7 µm
    RBC_DIAMETER_TOLERANCE = 3.0  # ±3 µm tolerance
    RBC_MIN_CIRCULARITY = 0.7    # RBCs are circular
    RBC_MAX_ECCENTRICITY = 0.6   # Low eccentricity (circular)

    def __init__(
        self,
        calibration: MicroscopeCalibration,
        use_color: bool = True,
        use_size: bool = True,
        use_shape: bool = True,
        strictness: float = 0.5
    ):
        """
        Initialize RBC filter.

        Args:
            calibration: Microscope calibration for size calculations
            use_color: Filter based on red/pink color (eosin intensity)
            use_size: Filter based on size (~7 µm)
            use_shape: Filter based on circular shape
            strictness: How strict the filtering (0-1, higher = more strict)
        """
        self.calibration = calibration
        self.use_color = use_color
        self.use_size = use_size
        self.use_shape = use_shape
        self.strictness = strictness

        # Calculate expected RBC size in pixels
        self.rbc_diameter_px = self.RBC_DIAMETER_UM / calibration.pixel_size_um
        self.rbc_area_px = np.pi * (self.rbc_diameter_px / 2) ** 2

        # Size tolerance in pixels
        tolerance_px = self.RBC_DIAMETER_TOLERANCE / calibration.pixel_size_um
        self.min_rbc_diameter_px = self.rbc_diameter_px - tolerance_px
        self.max_rbc_diameter_px = self.rbc_diameter_px + tolerance_px

        logger.info(
            f"RBC Filter initialized: expected diameter = {self.rbc_diameter_px:.1f} px "
            f"({self.RBC_DIAMETER_UM} µm), tolerance = ±{tolerance_px:.1f} px"
        )

    def is_rbc(
        self,
        measurement: 'CellMeasurement',
        color_image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> tuple[bool, float, str]:
        """
        Determine if a cell measurement is likely an RBC.

        Args:
            measurement: CellMeasurement object
            color_image: Original RGB image (for color analysis)
            mask: Segmentation mask

        Returns:
            Tuple of (is_rbc: bool, confidence: float, reason: str)
        """
        scores = []
        reasons = []

        # Size check
        if self.use_size:
            diameter = measurement.equivalent_diameter
            if self.min_rbc_diameter_px <= diameter <= self.max_rbc_diameter_px:
                # Within RBC size range
                # Score based on how close to ideal size
                size_diff = abs(diameter - self.rbc_diameter_px)
                size_score = 1.0 - (size_diff / self.RBC_DIAMETER_TOLERANCE * self.calibration.pixel_size_um)
                scores.append(max(0, size_score))
                reasons.append(f"size={diameter:.1f}px (~{measurement.equivalent_diameter_um:.1f}µm)")
            else:
                scores.append(0.0)

        # Shape check
        if self.use_shape:
            circularity = measurement.circularity
            eccentricity = measurement.eccentricity

            shape_score = 0.0
            if circularity >= self.RBC_MIN_CIRCULARITY:
                shape_score += 0.5
            if eccentricity <= self.RBC_MAX_ECCENTRICITY:
                shape_score += 0.5

            scores.append(shape_score)
            if shape_score > 0.5:
                reasons.append(f"shape(circ={circularity:.2f}, ecc={eccentricity:.2f})")

        # Color check (if color image provided)
        if self.use_color and color_image is not None and mask is not None:
            color_score = self._check_rbc_color(measurement, color_image, mask)
            scores.append(color_score)
            if color_score > 0.5:
                reasons.append("red/pink_color")

        # Calculate overall confidence
        if scores:
            confidence = np.mean(scores)
        else:
            confidence = 0.0

        # Threshold based on strictness
        threshold = 0.3 + (self.strictness * 0.4)  # 0.3 to 0.7
        is_rbc = confidence >= threshold

        reason_str = ", ".join(reasons) if reasons else "no_match"

        return is_rbc, confidence, reason_str

    def _check_rbc_color(
        self,
        measurement: 'CellMeasurement',
        color_image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Check if cell has RBC-like color (high eosin = pink/red).

        RBCs stain strongly with eosin (pink) and have no hematoxylin (purple).
        """
        try:
            # Get cell pixels
            cell_mask = mask == measurement.label
            if not np.any(cell_mask):
                return 0.0

            # Extract RGB values for this cell
            r = color_image[:, :, 0][cell_mask].astype(float)
            g = color_image[:, :, 1][cell_mask].astype(float)
            b = color_image[:, :, 2][cell_mask].astype(float)

            # RBC characteristics in RGB:
            # - High red channel
            # - Red > Blue (eosin not hematoxylin)
            # - Moderate to low green

            mean_r = np.mean(r)
            mean_g = np.mean(g)
            mean_b = np.mean(b)

            score = 0.0

            # Red should be dominant or close to dominant
            if mean_r > mean_b:
                score += 0.4

            # Check for pink/red hue (high R, lower G and B)
            if mean_r > 100 and mean_r > mean_g * 0.9:
                score += 0.3

            # Not too dark (not a nucleus)
            brightness = (mean_r + mean_g + mean_b) / 3
            if brightness > 80:
                score += 0.3

            return min(score, 1.0)

        except Exception as e:
            logger.debug(f"Color check failed: {e}")
            return 0.0

    def filter_rbcs(
        self,
        measurements: list['CellMeasurement'],
        color_image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> tuple[list['CellMeasurement'], list['CellMeasurement'], dict]:
        """
        Filter out RBCs from a list of measurements.

        Args:
            measurements: List of CellMeasurement objects
            color_image: Original RGB image
            mask: Segmentation mask

        Returns:
            Tuple of (non_rbc_cells, rbc_cells, statistics)
        """
        non_rbc_cells = []
        rbc_cells = []
        rbc_confidences = []

        for m in measurements:
            is_rbc, confidence, reason = self.is_rbc(m, color_image, mask)

            if is_rbc:
                rbc_cells.append(m)
                rbc_confidences.append(confidence)
                logger.debug(f"Cell {m.cell_id} identified as RBC: {reason} (conf={confidence:.2f})")
            else:
                non_rbc_cells.append(m)

        stats = {
            "total_cells": len(measurements),
            "non_rbc_count": len(non_rbc_cells),
            "rbc_count": len(rbc_cells),
            "rbc_percentage": (len(rbc_cells) / len(measurements) * 100) if measurements else 0,
            "mean_rbc_confidence": np.mean(rbc_confidences) if rbc_confidences else 0
        }

        logger.info(
            f"RBC filtering: {stats['rbc_count']}/{stats['total_cells']} cells identified as RBCs "
            f"({stats['rbc_percentage']:.1f}%)"
        )

        return non_rbc_cells, rbc_cells, stats


@dataclass
class CellMeasurement:
    """
    Morphometric measurements for a single cell.

    All size measurements are provided in BOTH pixels and calibrated units (µm).
    Shape descriptors are dimensionless and don't need calibration.
    """

    # Identification
    cell_id: int
    label: int

    # =========================================================================
    # SIZE METRICS - PIXELS (raw measurements)
    # =========================================================================
    area: float                  # pixels²
    perimeter: float             # pixels
    equivalent_diameter: float   # pixels
    convex_area: float           # pixels²
    filled_area: float           # pixels²
    major_axis_length: float     # pixels
    minor_axis_length: float     # pixels

    # =========================================================================
    # SIZE METRICS - CALIBRATED (µm, µm²)
    # These are the scientifically meaningful measurements!
    # =========================================================================
    area_um2: float = 0.0                # µm² - USE THIS for publications!
    perimeter_um: float = 0.0            # µm
    equivalent_diameter_um: float = 0.0  # µm
    major_axis_um: float = 0.0           # µm
    minor_axis_um: float = 0.0           # µm

    # =========================================================================
    # SHAPE DESCRIPTORS (dimensionless - no calibration needed)
    # =========================================================================
    circularity: float = 0.0       # 1.0 = perfect circle, <1 = irregular
    eccentricity: float = 0.0      # 0 = circle, 1 = line
    solidity: float = 0.0          # area / convex_area (1 = convex)
    extent: float = 0.0            # area / bounding_box_area
    aspect_ratio: float = 0.0      # major_axis / minor_axis
    roundness: float = 0.0         # 4*area / (pi * major_axis²)
    compactness: float = 0.0       # perimeter² / area

    # =========================================================================
    # BOUNDING BOX (pixels)
    # =========================================================================
    bbox_min_row: int = 0
    bbox_min_col: int = 0
    bbox_max_row: int = 0
    bbox_max_col: int = 0
    bbox_width: int = 0
    bbox_height: int = 0

    # =========================================================================
    # POSITION AND ORIENTATION
    # =========================================================================
    centroid_row: float = 0.0      # pixels
    centroid_col: float = 0.0      # pixels
    orientation: float = 0.0       # radians (-π/2 to π/2)

    # =========================================================================
    # INTENSITY FEATURES (if intensity image provided)
    # =========================================================================
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    intensity_min: float = 0.0
    intensity_max: float = 0.0
    intensity_median: float = 0.0

    # =========================================================================
    # TEXTURE FEATURES - GLCM (if computed)
    # =========================================================================
    texture_entropy: float = 0.0
    texture_contrast: float = 0.0
    texture_homogeneity: float = 0.0
    texture_energy: float = 0.0
    texture_correlation: float = 0.0


@dataclass
class MorphometryResult:
    """Complete morphometry analysis results."""
    measurements: list[CellMeasurement]
    summary_stats: dict
    pixel_size_um: float       # Calibration: microns per pixel
    objective: str             # Objective used (e.g., "40x")
    total_cells: int
    image_area_pixels: int
    image_area_um2: float      # Image area in µm²
    cell_density_per_mm2: float  # cells per mm²

    def to_dataframe(self) -> pd.DataFrame:
        """Convert measurements to pandas DataFrame."""
        return pd.DataFrame([vars(m) for m in self.measurements])

    def get_calibration_info(self) -> dict:
        """Get calibration information for the report."""
        return {
            "objective": self.objective,
            "pixel_size_um": self.pixel_size_um,
            "image_area_um2": self.image_area_um2,
            "image_area_mm2": self.image_area_um2 / 1e6,
            "cell_density_per_mm2": self.cell_density_per_mm2
        }


class MorphometryAnalyzer:
    """
    Comprehensive morphometric analysis of segmented cells.

    Calculates size, shape, intensity, and texture features for each cell.
    All measurements are automatically calibrated to real-world units (µm, µm²).

    Example using objective preset:
        >>> analyzer = MorphometryAnalyzer(objective="40x")
        >>> result = analyzer.analyze(masks, original_image)
        >>> df = result.to_dataframe()
        >>> print(f"Mean cell area: {df['area_um2'].mean():.2f} µm²")

    Example using custom calibration:
        >>> analyzer = MorphometryAnalyzer(pixel_size_um=0.32)
        >>> result = analyzer.analyze(masks, original_image)
    """

    def __init__(
        self,
        pixel_size_um: float = 1.0,
        objective: Optional[str] = None,
        compute_texture: bool = True,
        texture_distances: list[int] = None,
        texture_angles: list[float] = None
    ):
        """
        Initialize the analyzer with calibration.

        Args:
            pixel_size_um: Pixel size in microns (used if objective not specified)
            objective: Microscope objective ("4x", "10x", "20x", "40x", "60x", "100x")
                       If provided, overrides pixel_size_um with preset values.
            compute_texture: Whether to compute GLCM texture features (slower)
            texture_distances: Distances for GLCM texture analysis
            texture_angles: Angles for GLCM texture analysis

        Note:
            For accurate measurements, ALWAYS specify the objective or
            calibrate with a stage micrometer!
        """
        # Set up calibration
        if objective:
            self.calibration = MicroscopeCalibration.from_objective(objective)
            self.pixel_size_um = self.calibration.pixel_size_um
            self.objective = objective
        else:
            self.calibration = MicroscopeCalibration(pixel_size_um=pixel_size_um)
            self.pixel_size_um = pixel_size_um
            self.objective = "custom"

        self.compute_texture = compute_texture
        self.texture_distances = texture_distances or [1, 3, 5]
        self.texture_angles = texture_angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]

        logger.info(f"MorphometryAnalyzer initialized: {self.calibration}")

    def analyze(
        self,
        masks: np.ndarray,
        intensity_image: np.ndarray | None = None
    ) -> MorphometryResult:
        """
        Perform complete morphometric analysis with calibration.

        Args:
            masks: Label image from segmentation (0=background, 1..N=cells)
            intensity_image: Optional grayscale image for intensity features

        Returns:
            MorphometryResult with all measurements in both pixels and µm
        """
        # Ensure masks are integer type for regionprops
        masks = masks.astype(np.int32)

        # Calculate image area
        image_area_pixels = masks.size
        image_area_um2 = self.calibration.pixels_to_um2(image_area_pixels)

        if masks.max() == 0:
            logger.warning("No cells found in mask")
            return MorphometryResult(
                measurements=[],
                summary_stats={},
                pixel_size_um=self.pixel_size_um,
                objective=self.objective,
                total_cells=0,
                image_area_pixels=image_area_pixels,
                image_area_um2=image_area_um2,
                cell_density_per_mm2=0.0
            )

        # Convert intensity image to grayscale if needed
        if intensity_image is not None and intensity_image.ndim == 3:
            intensity_image = np.mean(intensity_image, axis=2).astype(np.float32)

        # Get region properties from scikit-image
        if intensity_image is not None:
            regions = measure.regionprops(masks, intensity_image=intensity_image)
        else:
            regions = measure.regionprops(masks)

        # Calculate measurements for each cell
        measurements = []
        for idx, region in enumerate(regions):
            measurement = self._measure_cell(
                region,
                cell_id=idx + 1,
                intensity_image=intensity_image,
                masks=masks
            )
            measurements.append(measurement)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(measurements)

        # Calculate cell density (cells per mm²)
        total_cells = len(measurements)
        image_area_mm2 = image_area_um2 / 1e6  # Convert µm² to mm²
        cell_density_per_mm2 = total_cells / image_area_mm2 if image_area_mm2 > 0 else 0

        logger.info(
            f"Analysis complete: {total_cells} cells, "
            f"density: {cell_density_per_mm2:.1f} cells/mm², "
            f"calibration: {self.objective} ({self.pixel_size_um} µm/px)"
        )

        return MorphometryResult(
            measurements=measurements,
            summary_stats=summary_stats,
            pixel_size_um=self.pixel_size_um,
            objective=self.objective,
            total_cells=total_cells,
            image_area_pixels=image_area_pixels,
            image_area_um2=image_area_um2,
            cell_density_per_mm2=cell_density_per_mm2
        )

    def _measure_cell(
        self,
        region,
        cell_id: int,
        intensity_image: np.ndarray | None,
        masks: np.ndarray
    ) -> CellMeasurement:
        """Calculate all measurements for a single cell with calibration."""

        # =====================================================================
        # SIZE METRICS IN PIXELS
        # =====================================================================
        area = region.area
        perimeter = region.perimeter
        equivalent_diameter = region.equivalent_diameter
        convex_area = region.convex_area
        filled_area = region.filled_area
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length

        # =====================================================================
        # CALIBRATED SIZE METRICS (µm, µm²)
        # =====================================================================
        area_um2 = self.calibration.pixels_to_um2(area)
        perimeter_um = self.calibration.pixels_to_um(perimeter)
        equivalent_diameter_um = self.calibration.pixels_to_um(equivalent_diameter)
        major_axis_um = self.calibration.pixels_to_um(major_axis)
        minor_axis_um = self.calibration.pixels_to_um(minor_axis)

        # =====================================================================
        # SHAPE DESCRIPTORS (dimensionless)
        # =====================================================================
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        eccentricity = region.eccentricity
        solidity = region.solidity
        extent = region.extent
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
        roundness = (4 * area) / (np.pi * major_axis**2) if major_axis > 0 else 0
        compactness = (perimeter**2) / area if area > 0 else 0

        # Bounding box
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)

        # =====================================================================
        # INTENSITY FEATURES
        # =====================================================================
        intensity_mean = 0.0
        intensity_std = 0.0
        intensity_min = 0.0
        intensity_max = 0.0
        intensity_median = 0.0

        if intensity_image is not None:
            intensity_mean = region.intensity_mean
            cell_pixels = intensity_image[masks == region.label]
            if len(cell_pixels) > 0:
                intensity_std = np.std(cell_pixels)
                intensity_min = np.min(cell_pixels)
                intensity_max = np.max(cell_pixels)
                intensity_median = np.median(cell_pixels)

        # =====================================================================
        # TEXTURE FEATURES (GLCM)
        # =====================================================================
        texture_entropy = 0.0
        texture_contrast = 0.0
        texture_homogeneity = 0.0
        texture_energy = 0.0
        texture_correlation = 0.0

        if self.compute_texture and intensity_image is not None:
            texture_features = self._compute_texture_features(
                region, intensity_image, masks
            )
            texture_entropy = texture_features.get('entropy', 0.0)
            texture_contrast = texture_features.get('contrast', 0.0)
            texture_homogeneity = texture_features.get('homogeneity', 0.0)
            texture_energy = texture_features.get('energy', 0.0)
            texture_correlation = texture_features.get('correlation', 0.0)

        return CellMeasurement(
            # Identification
            cell_id=cell_id,
            label=region.label,
            # Size - pixels
            area=area,
            perimeter=perimeter,
            equivalent_diameter=equivalent_diameter,
            convex_area=convex_area,
            filled_area=filled_area,
            major_axis_length=major_axis,
            minor_axis_length=minor_axis,
            # Size - calibrated (µm)
            area_um2=area_um2,
            perimeter_um=perimeter_um,
            equivalent_diameter_um=equivalent_diameter_um,
            major_axis_um=major_axis_um,
            minor_axis_um=minor_axis_um,
            # Shape (dimensionless)
            circularity=circularity,
            eccentricity=eccentricity,
            solidity=solidity,
            extent=extent,
            aspect_ratio=aspect_ratio,
            roundness=roundness,
            compactness=compactness,
            # Bounding box
            bbox_min_row=bbox[0],
            bbox_min_col=bbox[1],
            bbox_max_row=bbox[2],
            bbox_max_col=bbox[3],
            bbox_width=bbox[3] - bbox[1],
            bbox_height=bbox[2] - bbox[0],
            # Position
            centroid_row=region.centroid[0],
            centroid_col=region.centroid[1],
            orientation=region.orientation,
            # Intensity
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            intensity_median=intensity_median,
            # Texture
            texture_entropy=texture_entropy,
            texture_contrast=texture_contrast,
            texture_homogeneity=texture_homogeneity,
            texture_energy=texture_energy,
            texture_correlation=texture_correlation
        )

    def _compute_texture_features(
        self,
        region,
        intensity_image: np.ndarray,
        masks: np.ndarray
    ) -> dict:
        """
        Compute GLCM texture features for a cell.

        GLCM = Gray-Level Co-occurrence Matrix
        Captures spatial relationships between pixel intensities.
        """
        try:
            # Extract cell region
            min_row, min_col, max_row, max_col = region.bbox
            cell_image = intensity_image[min_row:max_row, min_col:max_col].copy()
            cell_mask = masks[min_row:max_row, min_col:max_col] == region.label

            # Mask out non-cell pixels
            cell_image[~cell_mask] = 0

            # Normalize to 0-255 for GLCM
            if cell_image.max() > 0:
                cell_image = ((cell_image - cell_image.min()) /
                             (cell_image.max() - cell_image.min()) * 255).astype(np.uint8)
            else:
                return {}

            # Skip if cell is too small for texture analysis
            if cell_image.shape[0] < 4 or cell_image.shape[1] < 4:
                return {}

            # Compute GLCM
            glcm = graycomatrix(
                cell_image,
                distances=self.texture_distances,
                angles=self.texture_angles,
                levels=256,
                symmetric=True,
                normed=True
            )

            # Extract texture properties (average over distances and angles)
            contrast = graycoprops(glcm, 'contrast').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()

            # Entropy
            cell_pixels = intensity_image[masks == region.label]
            entropy = stats.entropy(np.histogram(cell_pixels, bins=64)[0] + 1e-10)

            return {
                'contrast': float(contrast),
                'homogeneity': float(homogeneity),
                'energy': float(energy),
                'correlation': float(correlation),
                'entropy': float(entropy)
            }

        except Exception as e:
            logger.debug(f"Texture computation failed for cell {region.label}: {e}")
            return {}

    def _calculate_summary_stats(
        self,
        measurements: list[CellMeasurement]
    ) -> dict:
        """Calculate summary statistics across all cells."""
        if not measurements:
            return {}

        df = pd.DataFrame([vars(m) for m in measurements])

        # Key metrics to summarize
        metrics = [
            'area', 'perimeter', 'circularity', 'eccentricity',
            'solidity', 'aspect_ratio', 'intensity_mean'
        ]

        summary = {}
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    summary[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75))
                    }

        return summary


# =========================================================================
# CALIBRATION HELPERS
# =========================================================================

def calibrate_measurements(
    df: pd.DataFrame,
    pixel_size_um: float
) -> pd.DataFrame:
    """
    Convert pixel measurements to real-world units (microns).

    Args:
        df: DataFrame with measurements in pixels
        pixel_size_um: Pixel size in microns

    Returns:
        DataFrame with calibrated measurements
    """
    calibrated = df.copy()

    # Area: pixels² -> µm²
    if 'area' in calibrated.columns:
        calibrated['area_um2'] = calibrated['area'] * (pixel_size_um ** 2)
        calibrated['convex_area_um2'] = calibrated['convex_area'] * (pixel_size_um ** 2)
        calibrated['filled_area_um2'] = calibrated['filled_area'] * (pixel_size_um ** 2)

    # Length: pixels -> µm
    if 'perimeter' in calibrated.columns:
        calibrated['perimeter_um'] = calibrated['perimeter'] * pixel_size_um
        calibrated['equivalent_diameter_um'] = calibrated['equivalent_diameter'] * pixel_size_um
        calibrated['major_axis_um'] = calibrated['major_axis_length'] * pixel_size_um
        calibrated['minor_axis_um'] = calibrated['minor_axis_length'] * pixel_size_um

    return calibrated


# Convenience function
def analyze_morphometry(
    masks: np.ndarray,
    intensity_image: np.ndarray | None = None,
    pixel_size_um: float = 1.0
) -> pd.DataFrame:
    """
    Quick morphometry analysis function.

    Args:
        masks: Segmentation mask
        intensity_image: Original grayscale image
        pixel_size_um: Pixel size in microns

    Returns:
        DataFrame with all measurements
    """
    analyzer = MorphometryAnalyzer(pixel_size_um=pixel_size_um)
    result = analyzer.analyze(masks, intensity_image)
    return result.to_dataframe()
