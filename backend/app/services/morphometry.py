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
    1. Image resolution (pixels per micron)
    2. Segmentation quality
    3. Proper calibration
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from skimage import measure, feature
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage, stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class CellMeasurement:
    """Morphometric measurements for a single cell."""

    # Identification
    cell_id: int
    label: int

    # Size metrics (in pixels, multiply by calibration for microns)
    area: float
    perimeter: float
    equivalent_diameter: float
    convex_area: float
    filled_area: float

    # Shape descriptors (dimensionless)
    circularity: float          # 1.0 = perfect circle
    eccentricity: float         # 0 = circle, 1 = line
    solidity: float             # area / convex_area
    extent: float               # area / bounding_box_area
    aspect_ratio: float         # major_axis / minor_axis
    roundness: float            # 4*area / (pi * major_axis^2)
    compactness: float          # perimeter^2 / area

    # Bounding box
    bbox_min_row: int
    bbox_min_col: int
    bbox_max_row: int
    bbox_max_col: int
    bbox_width: int
    bbox_height: int

    # Centroid (position)
    centroid_row: float
    centroid_col: float

    # Orientation (radians, -pi/2 to pi/2)
    orientation: float

    # Axes
    major_axis_length: float
    minor_axis_length: float

    # Intensity features (if intensity image provided)
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    intensity_min: float = 0.0
    intensity_max: float = 0.0
    intensity_median: float = 0.0

    # Texture features (if computed)
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
    pixel_size_um: float  # Calibration: microns per pixel
    total_cells: int
    image_area_pixels: int
    cell_density: float   # cells per unit area

    def to_dataframe(self) -> pd.DataFrame:
        """Convert measurements to pandas DataFrame."""
        return pd.DataFrame([vars(m) for m in self.measurements])


class MorphometryAnalyzer:
    """
    Comprehensive morphometric analysis of segmented cells.

    Calculates size, shape, intensity, and texture features for each cell.
    All measurements can be calibrated to real-world units (microns).

    Example:
        >>> analyzer = MorphometryAnalyzer(pixel_size_um=0.5)
        >>> result = analyzer.analyze(masks, original_image)
        >>> df = result.to_dataframe()
        >>> print(f"Mean cell area: {df['area'].mean() * 0.5**2:.2f} um²")
    """

    def __init__(
        self,
        pixel_size_um: float = 1.0,
        compute_texture: bool = True,
        texture_distances: list[int] = [1, 3, 5],
        texture_angles: list[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    ):
        """
        Initialize the analyzer.

        Args:
            pixel_size_um: Pixel size in microns (for calibration)
            compute_texture: Whether to compute texture features (slower)
            texture_distances: Distances for GLCM texture analysis
            texture_angles: Angles for GLCM texture analysis
        """
        self.pixel_size_um = pixel_size_um
        self.compute_texture = compute_texture
        self.texture_distances = texture_distances
        self.texture_angles = texture_angles

    def analyze(
        self,
        masks: np.ndarray,
        intensity_image: np.ndarray | None = None
    ) -> MorphometryResult:
        """
        Perform complete morphometric analysis.

        Args:
            masks: Label image from segmentation (0=background, 1..N=cells)
            intensity_image: Optional grayscale image for intensity features

        Returns:
            MorphometryResult with all measurements and statistics
        """
        # Ensure masks are integer type for regionprops
        masks = masks.astype(np.int32)

        if masks.max() == 0:
            logger.warning("No cells found in mask")
            return MorphometryResult(
                measurements=[],
                summary_stats={},
                pixel_size_um=self.pixel_size_um,
                total_cells=0,
                image_area_pixels=masks.size,
                cell_density=0.0
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

        # Calculate cell density
        total_cells = len(measurements)
        image_area_pixels = masks.size
        cell_density = total_cells / (image_area_pixels * self.pixel_size_um**2)

        return MorphometryResult(
            measurements=measurements,
            summary_stats=summary_stats,
            pixel_size_um=self.pixel_size_um,
            total_cells=total_cells,
            image_area_pixels=image_area_pixels,
            cell_density=cell_density
        )

    def _measure_cell(
        self,
        region,
        cell_id: int,
        intensity_image: np.ndarray | None,
        masks: np.ndarray
    ) -> CellMeasurement:
        """Calculate all measurements for a single cell."""

        # Basic size metrics
        area = region.area
        perimeter = region.perimeter
        equivalent_diameter = region.equivalent_diameter
        convex_area = region.convex_area
        filled_area = region.filled_area

        # Shape descriptors
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        eccentricity = region.eccentricity
        solidity = region.solidity
        extent = region.extent

        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
        roundness = (4 * area) / (np.pi * major_axis**2) if major_axis > 0 else 0
        compactness = (perimeter**2) / area if area > 0 else 0

        # Bounding box
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)

        # Intensity features
        intensity_mean = 0.0
        intensity_std = 0.0
        intensity_min = 0.0
        intensity_max = 0.0
        intensity_median = 0.0

        if intensity_image is not None:
            intensity_mean = region.intensity_mean
            # Calculate additional intensity stats manually
            cell_pixels = intensity_image[masks == region.label]
            if len(cell_pixels) > 0:
                intensity_std = np.std(cell_pixels)
                intensity_min = np.min(cell_pixels)
                intensity_max = np.max(cell_pixels)
                intensity_median = np.median(cell_pixels)

        # Texture features
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
            cell_id=cell_id,
            label=region.label,
            area=area,
            perimeter=perimeter,
            equivalent_diameter=equivalent_diameter,
            convex_area=convex_area,
            filled_area=filled_area,
            circularity=circularity,
            eccentricity=eccentricity,
            solidity=solidity,
            extent=extent,
            aspect_ratio=aspect_ratio,
            roundness=roundness,
            compactness=compactness,
            bbox_min_row=bbox[0],
            bbox_min_col=bbox[1],
            bbox_max_row=bbox[2],
            bbox_max_col=bbox[3],
            bbox_width=bbox[3] - bbox[1],
            bbox_height=bbox[2] - bbox[0],
            centroid_row=region.centroid[0],
            centroid_col=region.centroid[1],
            orientation=region.orientation,
            major_axis_length=major_axis,
            minor_axis_length=minor_axis,
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            intensity_median=intensity_median,
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
