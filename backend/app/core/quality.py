"""
Image Quality Assessment
========================

Automatically assess microscopy image quality before analysis.

Detects common problems that affect morphometry accuracy:
- Out of focus / blur
- Low tissue content
- Artifacts (folds, bubbles, debris)
- Uneven illumination
- Oversaturation / undersaturation

Usage:
    >>> qc = ImageQualityChecker()
    >>> report = qc.assess(image)
    >>> if report.overall_quality < 0.5:
    ...     print(f"Warning: {report.issues}")
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from scipy import ndimage
from skimage import filters, exposure, morphology


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"   # >0.8
    GOOD = "good"             # 0.6-0.8
    ACCEPTABLE = "acceptable" # 0.4-0.6
    POOR = "poor"             # 0.2-0.4
    UNUSABLE = "unusable"     # <0.2


@dataclass
class QualityReport:
    """
    Complete quality assessment report for an image.

    Attributes:
        overall_quality: Combined quality score (0-1)
        quality_level: Human-readable quality level
        focus_score: Focus/sharpness quality (0-1)
        tissue_percentage: Percentage of image containing tissue
        illumination_uniformity: How even is the lighting (0-1)
        stain_quality: H&E stain separation quality (0-1, for histology)
        saturation_score: Check for over/under exposure (0-1)
        artifact_score: Absence of artifacts (0-1, higher = fewer artifacts)
        issues: List of detected issues
        recommendations: Suggested actions
        is_usable: Whether the image is suitable for analysis
    """
    overall_quality: float
    quality_level: QualityLevel
    focus_score: float
    tissue_percentage: float
    illumination_uniformity: float
    stain_quality: float
    saturation_score: float
    artifact_score: float
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    is_usable: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_quality": round(self.overall_quality, 3),
            "quality_level": self.quality_level.value,
            "focus_score": round(self.focus_score, 3),
            "tissue_percentage": round(self.tissue_percentage, 1),
            "illumination_uniformity": round(self.illumination_uniformity, 3),
            "stain_quality": round(self.stain_quality, 3),
            "saturation_score": round(self.saturation_score, 3),
            "artifact_score": round(self.artifact_score, 3),
            "issues": self.issues,
            "recommendations": self.recommendations,
            "is_usable": self.is_usable
        }


class ImageQualityChecker:
    """
    Comprehensive image quality assessment for microscopy images.

    Analyzes multiple quality aspects and provides actionable feedback.
    """

    def __init__(
        self,
        min_tissue_percentage: float = 10.0,
        min_focus_score: float = 0.3,
        is_histology: bool = True
    ):
        """
        Initialize quality checker.

        Args:
            min_tissue_percentage: Minimum tissue content required (%)
            min_focus_score: Minimum acceptable focus score
            is_histology: Whether to check H&E-specific quality metrics
        """
        self.min_tissue_percentage = min_tissue_percentage
        self.min_focus_score = min_focus_score
        self.is_histology = is_histology

    def assess(self, image: np.ndarray) -> QualityReport:
        """
        Perform complete quality assessment.

        Args:
            image: RGB or grayscale image

        Returns:
            QualityReport with all quality metrics
        """
        issues = []
        recommendations = []

        # Convert to grayscale for some analyses
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 1. Focus/Sharpness Assessment
        focus_score = self._assess_focus(gray)
        if focus_score < self.min_focus_score:
            issues.append(f"Image is out of focus (score: {focus_score:.2f})")
            recommendations.append("Refocus the microscope or recapture the image")

        # 2. Tissue Content
        tissue_percentage = self._assess_tissue_content(image, gray)
        if tissue_percentage < self.min_tissue_percentage:
            issues.append(f"Low tissue content ({tissue_percentage:.1f}%)")
            recommendations.append("Select a region with more tissue")

        # 3. Illumination Uniformity
        illumination_uniformity = self._assess_illumination(gray)
        if illumination_uniformity < 0.5:
            issues.append("Uneven illumination detected")
            recommendations.append("Adjust microscope illumination (Köhler alignment)")

        # 4. Saturation Check (over/under exposure)
        saturation_score = self._assess_saturation(gray)
        if saturation_score < 0.5:
            issues.append("Image has exposure problems")
            recommendations.append("Adjust exposure time or light intensity")

        # 5. Stain Quality (for histology)
        if self.is_histology and len(image.shape) == 3:
            stain_quality = self._assess_stain_quality(image)
            if stain_quality < 0.4:
                issues.append("Poor stain quality or separation")
                recommendations.append("Check staining protocol or use stain normalization")
        else:
            stain_quality = 1.0  # Not applicable

        # 6. Artifact Detection
        artifact_score = self._detect_artifacts(image, gray)
        if artifact_score < 0.5:
            issues.append("Artifacts detected (folds, bubbles, or debris)")
            recommendations.append("Select a cleaner region or re-prepare the slide")

        # Calculate overall quality (weighted average)
        weights = {
            'focus': 0.30,
            'tissue': 0.20,
            'illumination': 0.15,
            'saturation': 0.15,
            'stain': 0.10,
            'artifact': 0.10
        }

        overall_quality = (
            focus_score * weights['focus'] +
            min(tissue_percentage / 50.0, 1.0) * weights['tissue'] +  # Normalize tissue %
            illumination_uniformity * weights['illumination'] +
            saturation_score * weights['saturation'] +
            stain_quality * weights['stain'] +
            artifact_score * weights['artifact']
        )

        # Determine quality level
        if overall_quality >= 0.8:
            quality_level = QualityLevel.EXCELLENT
        elif overall_quality >= 0.6:
            quality_level = QualityLevel.GOOD
        elif overall_quality >= 0.4:
            quality_level = QualityLevel.ACCEPTABLE
        elif overall_quality >= 0.2:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.UNUSABLE

        # Determine if usable
        is_usable = (
            focus_score >= 0.2 and
            tissue_percentage >= 5.0 and
            overall_quality >= 0.3
        )

        if not is_usable:
            issues.insert(0, "⚠️ IMAGE MAY NOT BE SUITABLE FOR ANALYSIS")

        return QualityReport(
            overall_quality=overall_quality,
            quality_level=quality_level,
            focus_score=focus_score,
            tissue_percentage=tissue_percentage,
            illumination_uniformity=illumination_uniformity,
            stain_quality=stain_quality,
            saturation_score=saturation_score,
            artifact_score=artifact_score,
            issues=issues,
            recommendations=recommendations,
            is_usable=is_usable
        )

    def _assess_focus(self, gray: np.ndarray) -> float:
        """
        Assess image focus using Laplacian variance.

        Higher variance = sharper image = better focus.
        """
        # Laplacian operator detects edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to 0-1 scale
        # Typical values: <100 = blurry, 100-500 = ok, >500 = sharp
        # Adjusted for microscopy images
        normalized = min(variance / 500.0, 1.0)

        return normalized

    def _assess_tissue_content(
        self,
        image: np.ndarray,
        gray: np.ndarray
    ) -> float:
        """
        Estimate percentage of image containing tissue.

        For H&E: Tissue is darker/colored, background is white.
        """
        if len(image.shape) == 3:
            # For color images, use saturation and brightness
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]

            # Tissue has some saturation and is not pure white
            tissue_mask = (saturation > 20) | (value < 220)
        else:
            # For grayscale, use Otsu thresholding
            threshold = filters.threshold_otsu(gray)
            tissue_mask = gray < threshold

        # Clean up with morphology (remove small noise)
        tissue_mask = morphology.area_opening(tissue_mask, area_threshold=100)

        tissue_percentage = (np.sum(tissue_mask) / tissue_mask.size) * 100

        return tissue_percentage

    def _assess_illumination(self, gray: np.ndarray) -> float:
        """
        Assess illumination uniformity.

        Divides image into regions and checks if brightness is consistent.
        """
        h, w = gray.shape

        # Divide into 4x4 grid
        grid_h, grid_w = h // 4, w // 4
        region_means = []

        for i in range(4):
            for j in range(4):
                region = gray[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                region_means.append(np.mean(region))

        # Calculate coefficient of variation
        mean_brightness = np.mean(region_means)
        std_brightness = np.std(region_means)

        if mean_brightness > 0:
            cv = std_brightness / mean_brightness
            # Lower CV = more uniform. CV of 0.1 is good, 0.3 is bad
            uniformity = max(0, 1 - cv * 3)
        else:
            uniformity = 0

        return uniformity

    def _assess_saturation(self, gray: np.ndarray) -> float:
        """
        Check for over/under saturation (exposure problems).

        Good images use the full dynamic range without clipping.
        """
        # Check for overexposed pixels (white clipping)
        overexposed = np.sum(gray >= 250) / gray.size
        # Check for underexposed pixels (black clipping)
        underexposed = np.sum(gray <= 5) / gray.size

        # Calculate score (penalize clipping)
        score = 1.0

        if overexposed > 0.1:  # More than 10% overexposed
            score -= min(overexposed * 2, 0.5)

        if underexposed > 0.1:  # More than 10% underexposed
            score -= min(underexposed * 2, 0.5)

        # Also check dynamic range usage
        p5, p95 = np.percentile(gray, [5, 95])
        dynamic_range = (p95 - p5) / 255.0

        if dynamic_range < 0.3:  # Low contrast
            score -= 0.2

        return max(0, score)

    def _assess_stain_quality(self, image: np.ndarray) -> float:
        """
        Assess H&E stain quality and separation.

        Good H&E has distinct purple (H) and pink (E) regions.
        """
        # Convert to LAB for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1].astype(np.float32)  # Green-Red axis
        b_channel = lab[:, :, 2].astype(np.float32)  # Blue-Yellow axis

        # Get tissue mask (non-white regions)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < 230

        if np.sum(tissue_mask) < 100:
            return 0.5  # Not enough tissue to assess

        # In good H&E:
        # - Hematoxylin (purple/blue) has low A (greenish), low B (bluish)
        # - Eosin (pink) has high A (reddish), variable B

        a_tissue = a_channel[tissue_mask]
        b_tissue = b_channel[tissue_mask]

        # Check for color variation (good staining has range)
        a_range = np.percentile(a_tissue, 95) - np.percentile(a_tissue, 5)
        b_range = np.percentile(b_tissue, 95) - np.percentile(b_tissue, 5)

        # Normalize: expect A range of 20-60, B range of 20-60
        a_score = min(a_range / 40, 1.0)
        b_score = min(b_range / 40, 1.0)

        # Combined score
        stain_quality = (a_score + b_score) / 2

        return stain_quality

    def _detect_artifacts(
        self,
        image: np.ndarray,
        gray: np.ndarray
    ) -> float:
        """
        Detect common artifacts: folds, bubbles, debris.

        Returns score where 1.0 = no artifacts, 0.0 = many artifacts.
        """
        artifact_pixels = 0
        total_pixels = gray.size

        # 1. Detect folds (dark linear structures)
        # Folds create very dark lines
        very_dark = gray < 30
        fold_pixels = np.sum(very_dark)

        # 2. Detect bubbles (circular bright regions)
        very_bright = gray > 250
        # Bubbles are large connected bright regions
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            low_saturation = hsv[:, :, 1] < 10
            bubble_candidates = very_bright & low_saturation
        else:
            bubble_candidates = very_bright

        # 3. Detect debris (small dark spots not part of tissue)
        # This is harder - for now, count isolated dark spots
        dark_spots = gray < 50
        dark_spots = morphology.area_opening(dark_spots, area_threshold=5)
        small_dark = dark_spots & ~morphology.area_opening(dark_spots, area_threshold=100)

        artifact_pixels = fold_pixels + np.sum(bubble_candidates) + np.sum(small_dark)
        artifact_ratio = artifact_pixels / total_pixels

        # Score: 1.0 if <1% artifacts, 0.0 if >10% artifacts
        artifact_score = max(0, 1 - artifact_ratio * 10)

        return artifact_score


def quick_quality_check(image: np.ndarray) -> dict:
    """
    Quick quality assessment for API responses.

    Returns dict with key metrics for JSON response.
    """
    checker = ImageQualityChecker()
    report = checker.assess(image)
    return report.to_dict()
