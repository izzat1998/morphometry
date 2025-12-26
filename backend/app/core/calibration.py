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


@dataclass
class CalibrationPreset:
    """
    Calibration preset for a specific objective.

    Attributes:
        objective: Objective magnification
        pixel_size_um: Size of one pixel in microns
        description: Human-readable description
        typical_cell_diameter_um: Expected cell size at this magnification
    """
    objective: str
    pixel_size_um: float
    description: str
    typical_cell_diameter_um: float  # For reference


# =============================================================================
# CALIBRATION PRESETS
# =============================================================================
# These values are typical for common microscope/camera combinations.
# Users should calibrate with a stage micrometer for precise measurements.

CALIBRATION_PRESETS = {
    # Standard brightfield microscope with typical camera
    # Based on common 1/2" sensor cameras (e.g., 5MP microscope cameras)

    ObjectiveType.OBJ_4X: CalibrationPreset(
        objective="4x",
        pixel_size_um=2.5,
        description="4× objective - low magnification, large field of view",
        typical_cell_diameter_um=50.0  # ~20 pixels
    ),

    ObjectiveType.OBJ_10X: CalibrationPreset(
        objective="10x",
        pixel_size_um=1.0,
        description="10× objective - standard overview magnification",
        typical_cell_diameter_um=30.0  # ~30 pixels
    ),

    ObjectiveType.OBJ_20X: CalibrationPreset(
        objective="20x",
        pixel_size_um=0.5,
        description="20× objective - good for cell morphology",
        typical_cell_diameter_um=20.0  # ~40 pixels
    ),

    ObjectiveType.OBJ_40X: CalibrationPreset(
        objective="40x",
        pixel_size_um=0.25,
        description="40× objective - high power, cellular detail",
        typical_cell_diameter_um=15.0  # ~60 pixels
    ),

    ObjectiveType.OBJ_60X: CalibrationPreset(
        objective="60x",
        pixel_size_um=0.17,
        description="60× objective - high resolution",
        typical_cell_diameter_um=12.0  # ~70 pixels
    ),

    ObjectiveType.OBJ_100X: CalibrationPreset(
        objective="100x",
        pixel_size_um=0.1,
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


class MicroscopeCalibration:
    """
    Microscope calibration for converting pixels to real units.

    This class handles all unit conversions for morphometric measurements.

    Example:
        >>> # Using preset
        >>> calib = MicroscopeCalibration.from_objective("40x")
        >>> print(f"Pixel size: {calib.pixel_size_um} µm")

        >>> # Custom calibration
        >>> calib = MicroscopeCalibration(pixel_size_um=0.32)

        >>> # Convert measurements
        >>> area_um2 = calib.pixels_to_um2(1500)  # 1500 px² → µm²
        >>> perimeter_um = calib.pixels_to_um(120)  # 120 px → µm
    """

    def __init__(
        self,
        pixel_size_um: float = 1.0,
        objective: str = "custom",
        description: str = ""
    ):
        """
        Initialize calibration.

        Args:
            pixel_size_um: Size of one pixel in microns (µm/pixel)
            objective: Objective magnification (for reference)
            description: Optional description
        """
        if pixel_size_um <= 0:
            raise ValueError("Pixel size must be positive")

        self.pixel_size_um = pixel_size_um
        self.objective = objective
        self.description = description

        # Derived values
        self.pixel_area_um2 = pixel_size_um ** 2

    @classmethod
    def from_objective(cls, objective: str) -> "MicroscopeCalibration":
        """
        Create calibration from objective preset.

        Args:
            objective: Objective magnification ("4x", "10x", "20x", "40x", "60x", "100x")

        Returns:
            MicroscopeCalibration with preset values
        """
        # Normalize input
        objective_lower = objective.lower().strip()

        # Find matching preset
        for obj_type, preset in CALIBRATION_PRESETS.items():
            if obj_type.value == objective_lower:
                return cls(
                    pixel_size_um=preset.pixel_size_um,
                    objective=preset.objective,
                    description=preset.description
                )

        # Try to match without 'x'
        for obj_type, preset in CALIBRATION_PRESETS.items():
            if obj_type.value.replace('x', '') == objective_lower.replace('x', ''):
                return cls(
                    pixel_size_um=preset.pixel_size_um,
                    objective=preset.objective,
                    description=preset.description
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
        objective: str = "custom"
    ) -> "MicroscopeCalibration":
        """
        Create calibration from known distance measurement.

        Use this with a stage micrometer or known structure.

        Args:
            pixels: Distance in pixels
            microns: Same distance in microns
            objective: Objective magnification (for reference)

        Returns:
            Calibrated MicroscopeCalibration

        Example:
            >>> # If 100 pixels = 50 µm on stage micrometer
            >>> calib = MicroscopeCalibration.from_known_distance(100, 50, "40x")
        """
        if pixels <= 0 or microns <= 0:
            raise ValueError("Both pixels and microns must be positive")

        pixel_size = microns / pixels

        return cls(
            pixel_size_um=pixel_size,
            objective=objective,
            description=f"Calibrated: {pixels} px = {microns} µm"
        )

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
            f"pixel_size={self.pixel_size_um:.3f} µm/px)"
        )

    def summary(self) -> str:
        """Get human-readable calibration summary."""
        lines = [
            f"Microscope Calibration",
            f"=" * 40,
            f"Objective: {self.objective}",
            f"Pixel size: {self.pixel_size_um:.4f} µm/pixel",
            f"Pixel area: {self.pixel_area_um2:.6f} µm²/pixel²",
            f"",
            f"At this magnification:",
            f"  100 pixels = {self.pixels_to_um(100):.1f} µm",
            f"  1000 px² = {self.pixels_to_um2(1000):.1f} µm²",
            f"  Expected cell: ~{self.get_expected_cell_diameter_pixels():.0f} px diameter",
            f"  Expected RBC: ~{self.get_rbc_diameter_pixels():.0f} px diameter",
        ]
        if self.description:
            lines.insert(3, f"Description: {self.description}")

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
