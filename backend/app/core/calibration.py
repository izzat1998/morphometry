"""
Microscope Calibration System
=============================

Provides pixel-to-micron calibration for accurate morphometric measurements.

Without proper calibration, all area/perimeter measurements are meaningless!
This module ensures measurements are in real biological units (µm, µm²).

Key Concepts:
    - Pixel Size: Physical size of one pixel (µm/pixel)
    - Magnification: Total magnification (objective × eyepiece)
    - Calibration Factor: Converts pixels to microns

Usage:
    >>> calib = MicroscopeCalibration.from_objective("40x")
    >>> area_um2 = calib.pixels_to_um2(area_pixels)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class ObjectiveType(str, Enum):
    """Common microscope objective magnifications."""
    OBJ_4X = "4x"
    OBJ_10X = "10x"
    OBJ_20X = "20x"
    OBJ_40X = "40x"
    OBJ_60X = "60x"
    OBJ_100X = "100x"
    CUSTOM = "custom"


class CalibrationSource(str, Enum):
    """How the calibration was obtained."""
    PRESET = "preset"                    # Using built-in preset (uncertain)
    STAGE_MICROMETER = "stage_micrometer"  # Calibrated with stage micrometer (accurate)
    KNOWN_STRUCTURE = "known_structure"   # Calibrated with known structure (good)
    USER_PROVIDED = "user_provided"       # User provided pixel size (unknown accuracy)
    DEFAULT = "default"                   # No calibration specified (very uncertain)


@dataclass
class CalibrationPreset:
    """
    Calibration preset for a specific objective.

    Attributes:
        objective: Objective magnification
        pixel_size_um: Typical/central pixel size in microns
        pixel_size_min: Minimum expected pixel size (high-res cameras)
        pixel_size_max: Maximum expected pixel size (low-res cameras)
        description: Human-readable description
        typical_cell_diameter_um: Expected cell size at this magnification
    """
    objective: str
    pixel_size_um: float           # Central/typical value
    pixel_size_min: float          # Minimum expected
    pixel_size_max: float          # Maximum expected
    description: str
    typical_cell_diameter_um: float

    @property
    def uncertainty_percent(self) -> float:
        """Calculate uncertainty as percentage of central value."""
        range_half = (self.pixel_size_max - self.pixel_size_min) / 2
        return (range_half / self.pixel_size_um) * 100

    @property
    def area_uncertainty_percent(self) -> float:
        """Area uncertainty (squared, so larger than linear)."""
        # Area scales with pixel_size², so uncertainty compounds
        min_area = self.pixel_size_min ** 2
        max_area = self.pixel_size_max ** 2
        central_area = self.pixel_size_um ** 2
        range_half = (max_area - min_area) / 2
        return (range_half / central_area) * 100


# =============================================================================
# CALIBRATION PRESETS
# =============================================================================
# These values are typical for common microscope/camera combinations.
# Users should calibrate with a stage micrometer for precise measurements.

CALIBRATION_PRESETS = {
    # Standard brightfield microscope with typical camera
    # Based on common 1/2" sensor cameras (e.g., 5MP microscope cameras)
    #
    # UNCERTAINTY RANGES are based on variation across:
    # - Different camera sensors (1/3" to 1" sensors)
    # - Different tube lenses (0.5x to 1x adapters)
    # - Different microscope manufacturers

    ObjectiveType.OBJ_4X: CalibrationPreset(
        objective="4x",
        pixel_size_um=2.5,
        pixel_size_min=1.0,    # High-res camera + 1x adapter
        pixel_size_max=4.0,    # Low-res camera + 0.5x adapter
        description="4× objective - low magnification, large field of view",
        typical_cell_diameter_um=50.0  # ~20 pixels
    ),

    ObjectiveType.OBJ_10X: CalibrationPreset(
        objective="10x",
        pixel_size_um=1.0,
        pixel_size_min=0.3,    # High-res scientific camera
        pixel_size_max=2.0,    # Consumer microscope camera
        description="10× objective - standard overview magnification",
        typical_cell_diameter_um=30.0  # ~30 pixels
    ),

    ObjectiveType.OBJ_20X: CalibrationPreset(
        objective="20x",
        pixel_size_um=0.5,
        pixel_size_min=0.2,    # High-res camera
        pixel_size_max=1.0,    # Low-res camera
        description="20× objective - good for cell morphology",
        typical_cell_diameter_um=20.0  # ~40 pixels
    ),

    ObjectiveType.OBJ_40X: CalibrationPreset(
        objective="40x",
        pixel_size_um=0.25,
        pixel_size_min=0.08,   # High-NA oil immersion + scientific camera
        pixel_size_max=0.5,    # Dry objective + basic camera
        description="40× objective - high power, cellular detail",
        typical_cell_diameter_um=15.0  # ~60 pixels
    ),

    ObjectiveType.OBJ_60X: CalibrationPreset(
        objective="60x",
        pixel_size_um=0.17,
        pixel_size_min=0.06,   # High-res oil immersion
        pixel_size_max=0.35,   # Dry objective, lower camera resolution
        description="60× objective - high resolution",
        typical_cell_diameter_um=12.0  # ~70 pixels
    ),

    ObjectiveType.OBJ_100X: CalibrationPreset(
        objective="100x",
        pixel_size_um=0.1,
        pixel_size_min=0.04,   # High-NA oil + scientific camera
        pixel_size_max=0.2,    # Standard oil immersion
        description="100× oil immersion - maximum resolution",
        typical_cell_diameter_um=10.0  # ~100 pixels
    ),
}

# Typical cell sizes for reference (in µm)
CELL_SIZE_REFERENCE = {
    "red_blood_cell": 7.0,          # RBCs are ~7 µm diameter
    "lymphocyte": 7.0,              # Small lymphocytes ~7 µm
    "neutrophil": 12.0,             # Neutrophils ~12-15 µm
    "epithelial_cell": 20.0,        # Epithelial cells ~15-30 µm
    "hepatocyte": 25.0,             # Liver cells ~20-30 µm
    "neuron_soma": 20.0,            # Neuron cell body ~10-30 µm
    "fibroblast": 30.0,             # Fibroblasts ~20-40 µm (elongated)
    "adipocyte": 100.0,             # Fat cells can be very large
    "nucleus_typical": 6.0,         # Typical nucleus ~5-10 µm
    "nucleus_histology": 8.0,       # Nuclei in H&E sections ~6-10 µm
}


@dataclass
class CalibrationWarning:
    """Warning about calibration uncertainty or issues."""
    level: str              # "info", "warning", "critical"
    code: str               # Machine-readable code
    message: str            # Human-readable message
    details: str            # Additional context
    recommendation: str     # Suggested action


class MicroscopeCalibration:
    """
    Microscope calibration for converting pixels to real units.

    This class handles all unit conversions for morphometric measurements.
    It also tracks calibration uncertainty and generates appropriate warnings.

    Example:
        >>> # Using preset (will generate uncertainty warning)
        >>> calib = MicroscopeCalibration.from_objective("40x")
        >>> print(f"Pixel size: {calib.pixel_size_um} µm")
        >>> for warning in calib.get_warnings():
        ...     print(f"[{warning.level}] {warning.message}")

        >>> # Custom calibration (more accurate, fewer warnings)
        >>> calib = MicroscopeCalibration.from_known_distance(100, 50, "40x")

        >>> # Convert measurements
        >>> area_um2 = calib.pixels_to_um2(1500)  # 1500 px² → µm²
        >>> perimeter_um = calib.pixels_to_um(120)  # 120 px → µm
    """

    def __init__(
        self,
        pixel_size_um: float = 1.0,
        objective: str = "custom",
        description: str = "",
        source: CalibrationSource = CalibrationSource.USER_PROVIDED,
        uncertainty_percent: float | None = None,
        pixel_size_min: float | None = None,
        pixel_size_max: float | None = None
    ):
        """
        Initialize calibration.

        Args:
            pixel_size_um: Size of one pixel in microns (µm/pixel)
            objective: Objective magnification (for reference)
            description: Optional description
            source: How the calibration was obtained
            uncertainty_percent: Estimated uncertainty (± %)
            pixel_size_min: Minimum expected pixel size
            pixel_size_max: Maximum expected pixel size
        """
        if pixel_size_um <= 0:
            raise ValueError("Pixel size must be positive")

        self.pixel_size_um = pixel_size_um
        self.objective = objective
        self.description = description
        self.source = source

        # Uncertainty tracking
        self.pixel_size_min = pixel_size_min or pixel_size_um
        self.pixel_size_max = pixel_size_max or pixel_size_um
        self._uncertainty_percent = uncertainty_percent

        # Derived values
        self.pixel_area_um2 = pixel_size_um ** 2

    @property
    def uncertainty_percent(self) -> float:
        """Linear measurement uncertainty (±%)."""
        if self._uncertainty_percent is not None:
            return self._uncertainty_percent
        # Calculate from min/max range
        range_half = (self.pixel_size_max - self.pixel_size_min) / 2
        return (range_half / self.pixel_size_um) * 100

    @property
    def area_uncertainty_percent(self) -> float:
        """Area measurement uncertainty (±%). Higher due to squaring."""
        # Area scales with pixel_size², so uncertainty compounds
        min_area = self.pixel_size_min ** 2
        max_area = self.pixel_size_max ** 2
        central_area = self.pixel_size_um ** 2
        range_half = (max_area - min_area) / 2
        return (range_half / central_area) * 100

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration is from a reliable source."""
        return self.source in (
            CalibrationSource.STAGE_MICROMETER,
            CalibrationSource.KNOWN_STRUCTURE
        )

    @classmethod
    def from_objective(cls, objective: str) -> "MicroscopeCalibration":
        """
        Create calibration from objective preset.

        WARNING: Preset calibrations have significant uncertainty (±50-80%).
        For publication-quality measurements, use from_known_distance() with
        a stage micrometer.

        Args:
            objective: Objective magnification ("4x", "10x", "20x", "40x", "60x", "100x")

        Returns:
            MicroscopeCalibration with preset values and uncertainty tracking
        """
        # Normalize input
        objective_lower = objective.lower().strip()

        # Find matching preset
        for obj_type, preset in CALIBRATION_PRESETS.items():
            if obj_type.value == objective_lower:
                return cls(
                    pixel_size_um=preset.pixel_size_um,
                    objective=preset.objective,
                    description=preset.description,
                    source=CalibrationSource.PRESET,
                    pixel_size_min=preset.pixel_size_min,
                    pixel_size_max=preset.pixel_size_max
                )

        # Try to match without 'x'
        for obj_type, preset in CALIBRATION_PRESETS.items():
            if obj_type.value.replace('x', '') == objective_lower.replace('x', ''):
                return cls(
                    pixel_size_um=preset.pixel_size_um,
                    objective=preset.objective,
                    description=preset.description,
                    source=CalibrationSource.PRESET,
                    pixel_size_min=preset.pixel_size_min,
                    pixel_size_max=preset.pixel_size_max
                )

        raise ValueError(
            f"Unknown objective: {objective}. "
            f"Available: {[o.value for o in ObjectiveType if o != ObjectiveType.CUSTOM]}"
        )

    @classmethod
    def from_known_distance(
        cls,
        pixels: float,
        microns: float,
        objective: str = "custom",
        source: CalibrationSource = CalibrationSource.STAGE_MICROMETER
    ) -> "MicroscopeCalibration":
        """
        Create calibration from known distance measurement.

        Use this with a stage micrometer or known structure for accurate results.

        Args:
            pixels: Distance in pixels
            microns: Same distance in microns
            objective: Objective magnification (for reference)
            source: Calibration source (STAGE_MICROMETER or KNOWN_STRUCTURE)

        Returns:
            Calibrated MicroscopeCalibration with low uncertainty

        Example:
            >>> # If 100 pixels = 50 µm on stage micrometer
            >>> calib = MicroscopeCalibration.from_known_distance(100, 50, "40x")
        """
        if pixels <= 0 or microns <= 0:
            raise ValueError("Both pixels and microns must be positive")

        pixel_size = microns / pixels

        # Estimate measurement uncertainty (~2% for stage micrometer, ~5% for known structure)
        if source == CalibrationSource.STAGE_MICROMETER:
            uncertainty = 2.0  # ±2% - stage micrometers are precise
        elif source == CalibrationSource.KNOWN_STRUCTURE:
            uncertainty = 5.0  # ±5% - biological structures vary
        else:
            uncertainty = 10.0  # ±10% - unknown accuracy

        # Calculate min/max from uncertainty
        factor = uncertainty / 100
        pixel_size_min = pixel_size * (1 - factor)
        pixel_size_max = pixel_size * (1 + factor)

        return cls(
            pixel_size_um=pixel_size,
            objective=objective,
            description=f"Calibrated: {pixels:.1f} px = {microns:.1f} µm",
            source=source,
            uncertainty_percent=uncertainty,
            pixel_size_min=pixel_size_min,
            pixel_size_max=pixel_size_max
        )

    @classmethod
    def from_pixel_size(
        cls,
        pixel_size_um: float,
        objective: str = "custom",
        source: CalibrationSource = CalibrationSource.USER_PROVIDED
    ) -> "MicroscopeCalibration":
        """
        Create calibration from user-provided pixel size.

        Args:
            pixel_size_um: Pixel size in microns
            objective: Objective magnification (for reference)
            source: How the pixel size was determined

        Returns:
            MicroscopeCalibration with specified pixel size
        """
        # Assume ±10% uncertainty for user-provided values
        uncertainty = 10.0
        factor = uncertainty / 100

        return cls(
            pixel_size_um=pixel_size_um,
            objective=objective,
            description="User-provided pixel size",
            source=source,
            uncertainty_percent=uncertainty,
            pixel_size_min=pixel_size_um * (1 - factor),
            pixel_size_max=pixel_size_um * (1 + factor)
        )

    # =========================================================================
    # WARNING GENERATION
    # =========================================================================

    def get_warnings(self) -> list[CalibrationWarning]:
        """
        Generate warnings about calibration accuracy.

        Returns:
            List of CalibrationWarning objects describing potential issues
        """
        warnings = []

        # Check calibration source
        if self.source == CalibrationSource.PRESET:
            warnings.append(CalibrationWarning(
                level="warning",
                code="PRESET_CALIBRATION",
                message=f"Using preset calibration for {self.objective} objective",
                details=(
                    f"Preset pixel size: {self.pixel_size_um:.3f} µm/px "
                    f"(range: {self.pixel_size_min:.3f} - {self.pixel_size_max:.3f} µm/px). "
                    f"Linear uncertainty: ±{self.uncertainty_percent:.0f}%, "
                    f"Area uncertainty: ±{self.area_uncertainty_percent:.0f}%."
                ),
                recommendation=(
                    "For publication-quality measurements, calibrate with a stage "
                    "micrometer using MicroscopeCalibration.from_known_distance()."
                )
            ))

        elif self.source == CalibrationSource.DEFAULT:
            warnings.append(CalibrationWarning(
                level="critical",
                code="NO_CALIBRATION",
                message="No calibration specified - using default values",
                details=(
                    "Measurements are in arbitrary units and cannot be compared "
                    "across images or to literature values."
                ),
                recommendation=(
                    "Specify an objective preset with from_objective() or "
                    "calibrate with a stage micrometer using from_known_distance()."
                )
            ))

        elif self.source == CalibrationSource.USER_PROVIDED:
            warnings.append(CalibrationWarning(
                level="info",
                code="USER_CALIBRATION",
                message="Using user-provided pixel size",
                details=(
                    f"Pixel size: {self.pixel_size_um:.3f} µm/px. "
                    f"Estimated uncertainty: ±{self.uncertainty_percent:.0f}%."
                ),
                recommendation=(
                    "Verify this value was obtained from a reliable source "
                    "(stage micrometer, camera specifications, etc.)."
                )
            ))

        # Check for high uncertainty
        if self.uncertainty_percent > 50:
            warnings.append(CalibrationWarning(
                level="warning",
                code="HIGH_UNCERTAINTY",
                message=f"High calibration uncertainty (±{self.uncertainty_percent:.0f}%)",
                details=(
                    f"Area measurements may vary by ±{self.area_uncertainty_percent:.0f}%. "
                    f"A 100 µm² cell could actually be "
                    f"{100 * (1 - self.area_uncertainty_percent/100):.0f} - "
                    f"{100 * (1 + self.area_uncertainty_percent/100):.0f} µm²."
                ),
                recommendation="Calibrate with a stage micrometer for accurate measurements."
            ))

        # Info about calibrated sources
        if self.source == CalibrationSource.STAGE_MICROMETER:
            warnings.append(CalibrationWarning(
                level="info",
                code="CALIBRATED",
                message="Calibration from stage micrometer",
                details=(
                    f"Pixel size: {self.pixel_size_um:.4f} µm/px. "
                    f"Estimated uncertainty: ±{self.uncertainty_percent:.0f}%."
                ),
                recommendation="Calibration is suitable for quantitative analysis."
            ))

        return warnings

    def get_warning_summary(self) -> dict:
        """
        Get a summary of calibration warnings for API responses.

        Returns:
            Dict with warning information suitable for JSON serialization
        """
        warnings = self.get_warnings()

        # Determine overall status
        levels = [w.level for w in warnings]
        if "critical" in levels:
            status = "critical"
        elif "warning" in levels:
            status = "warning"
        else:
            status = "ok"

        return {
            "status": status,
            "is_calibrated": self.is_calibrated,
            "source": self.source.value,
            "pixel_size_um": self.pixel_size_um,
            "uncertainty_linear_percent": round(self.uncertainty_percent, 1),
            "uncertainty_area_percent": round(self.area_uncertainty_percent, 1),
            "warnings": [
                {
                    "level": w.level,
                    "code": w.code,
                    "message": w.message,
                    "details": w.details,
                    "recommendation": w.recommendation
                }
                for w in warnings
            ]
        }

    # =========================================================================
    # UNIT CONVERSIONS
    # =========================================================================

    def pixels_to_um(self, pixels: float) -> float:
        """Convert distance from pixels to microns."""
        return pixels * self.pixel_size_um

    def um_to_pixels(self, microns: float) -> float:
        """Convert distance from microns to pixels."""
        return microns / self.pixel_size_um

    def pixels_to_um2(self, pixels_squared: float) -> float:
        """Convert area from pixels² to µm²."""
        return pixels_squared * self.pixel_area_um2

    def um2_to_pixels(self, um_squared: float) -> float:
        """Convert area from µm² to pixels²."""
        return um_squared / self.pixel_area_um2

    # =========================================================================
    # MEASUREMENT CONVERSION
    # =========================================================================

    def convert_measurements(self, measurements: dict) -> dict:
        """
        Convert a dictionary of measurements to real units.

        Automatically detects measurement types and applies appropriate conversion:
        - area → µm²
        - perimeter, diameter, major_axis, minor_axis → µm
        - dimensionless (circularity, eccentricity, solidity) → unchanged

        Args:
            measurements: Dict of measurement_name: value_in_pixels

        Returns:
            Dict with converted values and units
        """
        # Measurements that are areas (need µm² conversion)
        area_measurements = {
            'area', 'convex_area', 'filled_area', 'bbox_area',
            'equivalent_diameter_area'
        }

        # Measurements that are lengths (need µm conversion)
        length_measurements = {
            'perimeter', 'major_axis_length', 'minor_axis_length',
            'equivalent_diameter', 'feret_diameter_max', 'diameter',
            'major_axis', 'minor_axis'
        }

        # Dimensionless measurements (no conversion needed)
        dimensionless = {
            'circularity', 'eccentricity', 'solidity', 'extent',
            'orientation', 'aspect_ratio', 'roundness', 'compactness',
            'cell_id', 'label'
        }

        converted = {}

        for key, value in measurements.items():
            key_lower = key.lower()

            if value is None:
                converted[key] = None
                continue

            # Check if it's an area measurement
            if any(area_key in key_lower for area_key in area_measurements):
                converted[key] = self.pixels_to_um2(value)
                converted[f'{key}_unit'] = 'µm²'

            # Check if it's a length measurement
            elif any(len_key in key_lower for len_key in length_measurements):
                converted[key] = self.pixels_to_um(value)
                converted[f'{key}_unit'] = 'µm'

            # Dimensionless - keep as is
            elif any(dim_key in key_lower for dim_key in dimensionless):
                converted[key] = value
                # No unit suffix for dimensionless

            # Unknown - keep as is but warn
            else:
                converted[key] = value

        return converted

    def get_expected_cell_diameter_pixels(self) -> float:
        """
        Get expected cell diameter in pixels for this magnification.

        Useful for setting Cellpose diameter parameter.
        """
        for obj_type, preset in CALIBRATION_PRESETS.items():
            if obj_type.value == self.objective.lower():
                return preset.typical_cell_diameter_um / self.pixel_size_um

        # Default: assume ~15 µm cell
        return 15.0 / self.pixel_size_um

    def get_rbc_diameter_pixels(self) -> float:
        """
        Get expected RBC diameter in pixels.

        RBCs are consistently ~7 µm, useful for filtering.
        """
        return CELL_SIZE_REFERENCE["red_blood_cell"] / self.pixel_size_um

    def get_nucleus_diameter_pixels(self) -> float:
        """
        Get expected nucleus diameter in pixels.

        Typical nuclei are ~6-10 µm in histology.
        """
        return CELL_SIZE_REFERENCE["nucleus_histology"] / self.pixel_size_um

    # =========================================================================
    # INFO METHODS
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"MicroscopeCalibration(objective={self.objective}, "
            f"pixel_size={self.pixel_size_um:.3f} µm/px, "
            f"source={self.source.value})"
        )

    def summary(self) -> str:
        """Get human-readable calibration summary with uncertainty info."""
        # Status indicator
        if self.is_calibrated:
            status = "✓ CALIBRATED"
        elif self.source == CalibrationSource.PRESET:
            status = "⚠ PRESET (verify accuracy)"
        else:
            status = "⚠ UNCALIBRATED"

        lines = [
            f"Microscope Calibration  [{status}]",
            f"=" * 50,
            f"Objective: {self.objective}",
            f"Source: {self.source.value}",
            f"Pixel size: {self.pixel_size_um:.4f} µm/pixel",
            f"Pixel area: {self.pixel_area_um2:.6f} µm²/pixel²",
            f"",
            f"Uncertainty:",
            f"  Pixel size range: {self.pixel_size_min:.4f} - {self.pixel_size_max:.4f} µm/px",
            f"  Linear (±): {self.uncertainty_percent:.1f}%",
            f"  Area (±): {self.area_uncertainty_percent:.1f}%",
            f"",
            f"At this magnification:",
            f"  100 pixels = {self.pixels_to_um(100):.1f} µm (±{self.pixels_to_um(100) * self.uncertainty_percent / 100:.1f} µm)",
            f"  1000 px² = {self.pixels_to_um2(1000):.1f} µm² (±{self.pixels_to_um2(1000) * self.area_uncertainty_percent / 100:.1f} µm²)",
            f"  Expected cell: ~{self.get_expected_cell_diameter_pixels():.0f} px diameter",
            f"  Expected RBC: ~{self.get_rbc_diameter_pixels():.0f} px diameter",
        ]

        if self.description:
            lines.insert(4, f"Description: {self.description}")

        # Add warnings
        warnings = self.get_warnings()
        warning_msgs = [w for w in warnings if w.level == "warning"]
        if warning_msgs:
            lines.append("")
            lines.append("Warnings:")
            for w in warning_msgs:
                lines.append(f"  [{w.code}] {w.message}")
                lines.append(f"    → {w.recommendation}")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_objectives() -> list[dict]:
    """Get list of available objective presets for UI."""
    presets = []
    for obj_type, preset in CALIBRATION_PRESETS.items():
        presets.append({
            "id": obj_type.value,
            "name": f"{preset.objective} Objective",
            "pixel_size_um": preset.pixel_size_um,
            "description": preset.description,
            "typical_cell_um": preset.typical_cell_diameter_um
        })
    return presets


def estimate_objective_from_cell_size(
    cell_diameter_pixels: float,
    expected_cell_um: float = 15.0
) -> str:
    """
    Estimate which objective was used based on cell size.

    Args:
        cell_diameter_pixels: Measured cell diameter in pixels
        expected_cell_um: Expected real cell size in µm (default 15 µm)

    Returns:
        Best matching objective string
    """
    # Calculate implied pixel size
    implied_pixel_size = expected_cell_um / cell_diameter_pixels

    # Find closest preset
    best_match = None
    best_diff = float('inf')

    for obj_type, preset in CALIBRATION_PRESETS.items():
        diff = abs(preset.pixel_size_um - implied_pixel_size)
        if diff < best_diff:
            best_diff = diff
            best_match = obj_type.value

    return best_match
