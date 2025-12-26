"""
Image Preprocessing Pipeline
============================
Comprehensive preprocessing filters for microscopy images.

This module provides filters optimized for different microscopy modalities:
- Brightfield: Contrast enhancement, background subtraction
- Fluorescence: Denoising, intensity normalization
- Phase Contrast: Halo removal, edge enhancement
- H&E Stained: Color deconvolution, stain normalization

Key Insight:
    Preprocessing quality directly impacts segmentation accuracy.
    The order of operations matters: denoise -> enhance -> normalize.
"""

import numpy as np
import cv2
from skimage import exposure, filters, restoration, morphology, color
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ImageModality(str, Enum):
    """Supported microscopy image types."""
    BRIGHTFIELD = "brightfield"
    FLUORESCENCE = "fluorescence"
    PHASE_CONTRAST = "phase_contrast"
    HE_STAINED = "he_stained"
    AUTO = "auto"  # Auto-detect modality


def detect_image_modality(image: np.ndarray) -> tuple[ImageModality, float, dict]:
    """
    Automatically detect the microscopy image modality.

    Analyzes color distribution, background intensity, and other features
    to determine if the image is H&E stained, fluorescence, brightfield, etc.

    Args:
        image: Input image (RGB or grayscale)

    Returns:
        Tuple of (detected_modality, confidence_score, feature_dict)

    Detection Logic:
        - H&E: Pink/purple colors, light background, specific color ratios
        - Fluorescence: Dark background, bright spots, high contrast
        - Phase Contrast: Gray tones, halo patterns, medium background
        - Brightfield: Light background, low saturation, gray/brown tones
    """
    features = {}

    # Handle grayscale images
    if len(image.shape) == 2:
        # Grayscale - likely fluorescence or phase contrast
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        features['is_grayscale'] = True
        features['mean_intensity'] = mean_intensity
        features['std_intensity'] = std_intensity

        if mean_intensity < 50:  # Dark background
            return ImageModality.FLUORESCENCE, 0.7, features
        else:
            return ImageModality.PHASE_CONTRAST, 0.6, features

    features['is_grayscale'] = False

    # === COLOR ANALYSIS ===
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # RGB channel statistics
    r_mean, g_mean, b_mean = image[:,:,0].mean(), image[:,:,1].mean(), image[:,:,2].mean()
    features['r_mean'] = r_mean
    features['g_mean'] = g_mean
    features['b_mean'] = b_mean

    # Saturation analysis
    saturation = hsv[:,:,1]
    sat_mean = saturation.mean()
    features['saturation_mean'] = sat_mean

    # Value (brightness) analysis
    value = hsv[:,:,2]
    val_mean = value.mean()
    features['brightness_mean'] = val_mean

    # LAB analysis (A channel: green-red, B channel: blue-yellow)
    l_channel = lab[:,:,0]
    a_channel = lab[:,:,1]  # Green(-) to Red(+)
    b_channel = lab[:,:,2]  # Blue(-) to Yellow(+)
    features['lab_l_mean'] = l_channel.mean()
    features['lab_a_mean'] = a_channel.mean()
    features['lab_b_mean'] = b_channel.mean()

    # === BACKGROUND DETECTION ===
    # Find the brightest 20% of pixels (likely background)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright_threshold = np.percentile(gray, 80)
    bright_mask = gray >= bright_threshold
    bg_pixels = np.sum(bright_mask)
    total_pixels = image.shape[0] * image.shape[1]
    features['bright_background_ratio'] = bg_pixels / total_pixels

    # Background color (mean of bright pixels)
    if bg_pixels > 100:
        bg_r = image[:,:,0][bright_mask].mean()
        bg_g = image[:,:,1][bright_mask].mean()
        bg_b = image[:,:,2][bright_mask].mean()
        features['bg_r'] = bg_r
        features['bg_g'] = bg_g
        features['bg_b'] = bg_b
    else:
        bg_r, bg_g, bg_b = r_mean, g_mean, b_mean

    # === H&E DETECTION ===
    # H&E characteristics:
    # - Light/white background
    # - Pink (eosin) and purple/blue (hematoxylin) colors
    # - A channel shifted toward red (positive), B channel variable

    he_score = 0.0

    # Check for light background
    if val_mean > 120:
        he_score += 0.2

    # Check for pink/purple colors (H&E typical)
    # In LAB: A > 128 indicates reddish, background should be neutral
    if a_channel.mean() > 125:  # Slight red shift (eosin)
        he_score += 0.2

    # Check for purple (hematoxylin) - blue with some red
    # Hue around 270-300 degrees (purple range) in some pixels
    hue = hsv[:,:,0]
    purple_mask = ((hue > 120) & (hue < 160)) & (saturation > 30)  # Purple in OpenCV hue
    pink_mask = ((hue > 160) | (hue < 10)) & (saturation > 30)  # Pink/red range
    purple_ratio = np.sum(purple_mask) / total_pixels
    pink_ratio = np.sum(pink_mask) / total_pixels
    features['purple_ratio'] = purple_ratio
    features['pink_ratio'] = pink_ratio

    if purple_ratio > 0.05:  # At least 5% purple pixels
        he_score += 0.3
    if pink_ratio > 0.05:  # At least 5% pink pixels
        he_score += 0.2

    # Check saturation - H&E has moderate saturation
    if 20 < sat_mean < 100:
        he_score += 0.1

    features['he_score'] = he_score

    # === FLUORESCENCE DETECTION ===
    # Fluorescence characteristics:
    # - Dark/black background
    # - High contrast bright spots
    # - Often dominated by single color channel

    fluor_score = 0.0

    # Dark background
    dark_threshold = np.percentile(gray, 20)
    dark_mask = gray <= dark_threshold
    dark_ratio = np.sum(dark_mask) / total_pixels
    features['dark_background_ratio'] = dark_ratio

    if val_mean < 80:  # Dark overall
        fluor_score += 0.3
    if dark_ratio > 0.3:  # Lots of dark pixels
        fluor_score += 0.2

    # High contrast (bright spots on dark background)
    contrast = gray.std()
    features['contrast'] = contrast
    if contrast > 50 and val_mean < 100:
        fluor_score += 0.2

    # Single channel dominance (e.g., green for GFP, blue for DAPI)
    channel_max = max(r_mean, g_mean, b_mean)
    channel_min = min(r_mean, g_mean, b_mean)
    channel_ratio = channel_max / (channel_min + 1)
    features['channel_dominance_ratio'] = channel_ratio
    if channel_ratio > 1.5:  # One channel is dominant
        fluor_score += 0.3

    features['fluorescence_score'] = fluor_score

    # === PHASE CONTRAST DETECTION ===
    # Phase contrast characteristics:
    # - Gray tones (low saturation)
    # - Medium brightness
    # - Characteristic halos (bright-dark edges)

    phase_score = 0.0

    # Low saturation (grayscale-ish)
    if sat_mean < 30:
        phase_score += 0.3

    # Medium brightness
    if 80 < val_mean < 180:
        phase_score += 0.2

    # Check for halo pattern (edges with adjacent bright/dark)
    # Using Laplacian to detect edge patterns
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_std = np.std(laplacian)
    features['laplacian_std'] = laplacian_std
    if laplacian_std > 20:  # Strong edges
        phase_score += 0.2

    # Low color variation
    if channel_ratio < 1.2:
        phase_score += 0.3

    features['phase_contrast_score'] = phase_score

    # === BRIGHTFIELD DETECTION ===
    # Brightfield characteristics:
    # - Light background
    # - Low to medium saturation
    # - Brown/gray cell tones

    brightfield_score = 0.0

    # Light background
    if val_mean > 150:
        brightfield_score += 0.2

    # Low saturation
    if sat_mean < 50:
        brightfield_score += 0.2

    # Check for brownish tones (common in brightfield)
    brown_mask = (hue > 10) & (hue < 30) & (saturation > 20)
    brown_ratio = np.sum(brown_mask) / total_pixels
    features['brown_ratio'] = brown_ratio
    if brown_ratio > 0.1:
        brightfield_score += 0.3

    # Not H&E (less pink/purple)
    if purple_ratio < 0.03 and pink_ratio < 0.03:
        brightfield_score += 0.3

    features['brightfield_score'] = brightfield_score

    # === FINAL DECISION ===
    scores = {
        ImageModality.HE_STAINED: he_score,
        ImageModality.FLUORESCENCE: fluor_score,
        ImageModality.PHASE_CONTRAST: phase_score,
        ImageModality.BRIGHTFIELD: brightfield_score,
    }

    # Find the highest scoring modality
    best_modality = max(scores, key=scores.get)
    best_score = scores[best_modality]

    # Confidence based on score difference
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        confidence = min(best_score, (sorted_scores[0] - sorted_scores[1]) + 0.5)
    else:
        confidence = best_score

    confidence = min(confidence, 1.0)  # Cap at 1.0

    features['all_scores'] = {k.value: v for k, v in scores.items()}
    features['detected_modality'] = best_modality.value
    features['confidence'] = confidence

    return best_modality, confidence, features


# =============================================================================
# MACENKO STAIN NORMALIZATION
# =============================================================================

class MacenkoNormalizer:
    """
    Macenko stain normalization for H&E histology images.

    This method normalizes staining appearance across different slides by:
    1. Converting RGB to Optical Density (OD) space
    2. Finding stain vectors using SVD (Singular Value Decomposition)
    3. Separating stain concentrations
    4. Normalizing to a reference standard
    5. Reconstructing the normalized image

    Reference:
        Macenko et al. "A method for normalizing histology slides for
        quantitative analysis" (ISBI 2009)

    Usage:
        >>> normalizer = MacenkoNormalizer()
        >>> normalized = normalizer.normalize(image)
    """

    # Reference stain vectors (from well-stained H&E slide)
    # These are the "ideal" H&E stain directions in OD space
    REFERENCE_STAIN_MATRIX = np.array([
        [0.5626, 0.2159],  # Hematoxylin (R, G, B) -> OD
        [0.7201, 0.8012],
        [0.4062, 0.5581]
    ])

    # Reference maximum stain concentrations
    REFERENCE_MAX_CONC = np.array([1.9705, 1.0308])

    def __init__(
        self,
        luminosity_threshold: float = 0.8,
        angular_percentile: float = 99
    ):
        """
        Initialize the Macenko normalizer.

        Args:
            luminosity_threshold: Pixels with OD below this are considered background
            angular_percentile: Percentile for robust stain vector estimation
        """
        self.luminosity_threshold = luminosity_threshold
        self.angular_percentile = angular_percentile

    # -------------------------------------------------------------------------
    # STEP 1: RGB to Optical Density Conversion
    # -------------------------------------------------------------------------

    def rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to Optical Density (OD) space.

        Beer-Lambert Law: OD = -log10(I / I0)
        Where I0 = 255 (white/no stain), I = pixel intensity

        In OD space:
        - Background (white) → low OD values (~0)
        - Stained tissue → high OD values
        - Stain contributions are ADDITIVE (easier to separate)

        Args:
            image: RGB image (0-255)

        Returns:
            OD image (same shape as input)
        """
        # Avoid log(0) by adding small epsilon
        # Normalize to [0, 1] range first
        image_normalized = image.astype(np.float64) / 255.0

        # Clip to avoid log(0) - minimum value ~0.004 (1/255)
        image_normalized = np.clip(image_normalized, 1/255, 1)

        # Convert to OD: OD = -log10(I/I0) = -log10(I) since I0=1
        od = -np.log10(image_normalized)

        return od

    def od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """
        Convert Optical Density back to RGB.

        Inverse of Beer-Lambert: I = I0 * 10^(-OD)

        Args:
            od: Optical Density image

        Returns:
            RGB image (0-255, uint8)
        """
        # I = 10^(-OD)
        rgb = np.power(10, -od)

        # Scale to 0-255 and clip
        rgb = rgb * 255
        rgb = np.clip(rgb, 0, 255)

        return rgb.astype(np.uint8)

    # -------------------------------------------------------------------------
    # STEP 2: Extract Stain Vectors using SVD
    # -------------------------------------------------------------------------

    def get_tissue_mask(self, od: np.ndarray) -> np.ndarray:
        """
        Create mask for tissue pixels (exclude background).

        Background pixels have low optical density (close to white).
        We only want to analyze stained tissue for vector extraction.

        Args:
            od: Optical Density image (H, W, 3)

        Returns:
            Boolean mask where True = tissue pixel
        """
        # Calculate total OD per pixel (sum across RGB channels)
        od_total = np.sum(od, axis=2)

        # Tissue has higher OD than background
        # Threshold based on luminosity_threshold
        threshold = -np.log10(self.luminosity_threshold)
        tissue_mask = od_total > threshold

        return tissue_mask

    def extract_stain_vectors(self, image: np.ndarray) -> np.ndarray:
        """
        Extract stain vectors from the image using SVD.

        The key insight: In OD space, pixels lie on a 2D plane spanned by
        the two stain vectors (Hematoxylin and Eosin). SVD finds this plane.

        Algorithm:
        1. Convert to OD space
        2. Keep only tissue pixels (remove background)
        3. Use SVD to find the plane containing most variance
        4. Project to this plane and find extreme angles (pure H and E)

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Stain matrix (3, 2) where columns are H and E vectors
        """
        # Convert to OD
        od = self.rgb_to_od(image)

        # Get tissue mask
        tissue_mask = self.get_tissue_mask(od)

        # Flatten and keep only tissue pixels
        # Shape: (H*W, 3) -> filter -> (N_tissue, 3)
        od_flat = od.reshape(-1, 3)
        tissue_od = od_flat[tissue_mask.flatten()]

        if len(tissue_od) < 100:
            # Not enough tissue pixels, return reference matrix
            return self.REFERENCE_STAIN_MATRIX.copy()

        # Center the data (subtract mean)
        od_mean = np.mean(tissue_od, axis=0)
        od_centered = tissue_od - od_mean

        # SVD: Find principal components
        # U, S, Vt = svd(data)
        # Vt rows are principal directions
        try:
            _, _, Vt = np.linalg.svd(od_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return self.REFERENCE_STAIN_MATRIX.copy()

        # Project onto plane spanned by first 2 principal components
        # These capture most of the variance (H and E stain directions)
        plane_vectors = Vt[:2, :].T  # Shape: (3, 2)

        # Project tissue pixels onto this 2D plane
        projected = np.dot(od_centered, plane_vectors)  # Shape: (N_tissue, 2)

        # Convert to polar coordinates to find extreme angles
        # Pure H and pure E pixels will be at extreme angles
        angles = np.arctan2(projected[:, 1], projected[:, 0])

        # Find robust min/max angles using percentiles
        min_angle = np.percentile(angles, 100 - self.angular_percentile)
        max_angle = np.percentile(angles, self.angular_percentile)

        # Convert angles back to vectors
        vec1 = np.array([np.cos(min_angle), np.sin(min_angle)])
        vec2 = np.array([np.cos(max_angle), np.sin(max_angle)])

        # Transform back to 3D OD space
        stain1 = np.dot(plane_vectors, vec1)
        stain2 = np.dot(plane_vectors, vec2)

        # Normalize vectors (unit length)
        stain1 = stain1 / np.linalg.norm(stain1)
        stain2 = stain2 / np.linalg.norm(stain2)

        # Ensure Hematoxylin is first (it has more blue component)
        # H stain: higher OD in blue channel
        # E stain: higher OD in red channel
        if stain1[0] > stain2[0]:  # Compare red channel OD
            # stain1 is Eosin, stain2 is Hematoxylin -> swap
            stain1, stain2 = stain2, stain1

        # Build stain matrix (3 rows for RGB, 2 cols for H and E)
        stain_matrix = np.column_stack([stain1, stain2])

        return stain_matrix

    # -------------------------------------------------------------------------
    # STEP 3: Calculate Stain Concentrations
    # -------------------------------------------------------------------------

    def get_concentrations(
        self,
        image: np.ndarray,
        stain_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate stain concentrations for each pixel.

        Given stain vectors, we can decompose each pixel's OD into
        contributions from each stain: OD = C_h * V_h + C_e * V_e

        This is solved as a least-squares problem using pseudo-inverse.

        Args:
            image: RGB image (H, W, 3)
            stain_matrix: Stain vectors (3, 2)

        Returns:
            Tuple of (concentrations, max_concentrations)
            - concentrations: (H, W, 2) array of H and E concentrations per pixel
            - max_concentrations: (2,) array of max concentration for each stain
        """
        # Convert to OD
        od = self.rgb_to_od(image)
        h, w, _ = od.shape

        # Reshape to (N_pixels, 3)
        od_flat = od.reshape(-1, 3)

        # Solve: OD = Stain_Matrix @ Concentrations
        # Concentrations = (Stain_Matrix^T @ Stain_Matrix)^-1 @ Stain_Matrix^T @ OD
        # This is the pseudo-inverse solution

        # Compute pseudo-inverse of stain matrix
        # stain_matrix: (3, 2), pinv: (2, 3)
        stain_pinv = np.linalg.pinv(stain_matrix)

        # Calculate concentrations for all pixels at once
        # (2, 3) @ (3, N) -> (2, N)
        concentrations_flat = np.dot(stain_pinv, od_flat.T).T  # (N, 2)

        # Clip negative concentrations (physically impossible)
        concentrations_flat = np.clip(concentrations_flat, 0, None)

        # Reshape back to image dimensions
        concentrations = concentrations_flat.reshape(h, w, 2)

        # Calculate maximum concentrations (99th percentile for robustness)
        max_c_h = np.percentile(concentrations[:, :, 0], 99)
        max_c_e = np.percentile(concentrations[:, :, 1], 99)
        max_concentrations = np.array([max_c_h, max_c_e])

        return concentrations, max_concentrations

    # -------------------------------------------------------------------------
    # STEP 4: Normalize and Reconstruct
    # -------------------------------------------------------------------------

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Perform full Macenko stain normalization.

        This is the main method that combines all steps:
        1. Extract stain vectors from the source image
        2. Calculate stain concentrations
        3. Normalize concentrations to reference standard
        4. Reconstruct image using reference stain vectors

        Args:
            image: RGB image (H, W, 3), uint8

        Returns:
            Normalized RGB image (H, W, 3), uint8
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input must be RGB image (H, W, 3)")

        h, w, _ = image.shape

        # Step 1: Extract stain vectors from this image
        source_stain_matrix = self.extract_stain_vectors(image)

        # Step 2: Get stain concentrations
        concentrations, source_max_conc = self.get_concentrations(
            image, source_stain_matrix
        )

        # Step 3: Normalize concentrations to reference standard
        # Scale so that max concentration matches reference
        normalized_conc = concentrations.copy()

        # Avoid division by zero
        for i in range(2):
            if source_max_conc[i] > 0.01:
                normalized_conc[:, :, i] = (
                    concentrations[:, :, i] *
                    (self.REFERENCE_MAX_CONC[i] / source_max_conc[i])
                )

        # Step 4: Reconstruct image using REFERENCE stain vectors
        # OD_normalized = Reference_Stain_Matrix @ Normalized_Concentrations
        conc_flat = normalized_conc.reshape(-1, 2)  # (N, 2)

        # (3, 2) @ (2, N) -> (3, N)
        od_normalized_flat = np.dot(self.REFERENCE_STAIN_MATRIX, conc_flat.T).T

        # Reshape to image
        od_normalized = od_normalized_flat.reshape(h, w, 3)

        # Convert back to RGB
        normalized_rgb = self.od_to_rgb(od_normalized)

        return normalized_rgb

    def get_stain_channels(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate image into individual stain channels (H and E).

        Useful for analyzing each stain independently.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Tuple of (hematoxylin_image, eosin_image) as RGB images
        """
        h, w, _ = image.shape

        # Extract stain vectors
        stain_matrix = self.extract_stain_vectors(image)

        # Get concentrations
        concentrations, _ = self.get_concentrations(image, stain_matrix)

        # Reconstruct each stain separately
        # Hematoxylin only
        h_conc = np.zeros((h, w, 2))
        h_conc[:, :, 0] = concentrations[:, :, 0]
        h_od = np.dot(stain_matrix, h_conc.reshape(-1, 2).T).T.reshape(h, w, 3)
        h_rgb = self.od_to_rgb(h_od)

        # Eosin only
        e_conc = np.zeros((h, w, 2))
        e_conc[:, :, 1] = concentrations[:, :, 1]
        e_od = np.dot(stain_matrix, e_conc.reshape(-1, 2).T).T.reshape(h, w, 3)
        e_rgb = self.od_to_rgb(e_od)

        return h_rgb, e_rgb


@dataclass
class PreprocessingResult:
    """Container for preprocessing output with metadata."""
    image: np.ndarray
    original_shape: tuple[int, ...]
    applied_filters: list[str]
    modality: ImageModality
    histogram_before: np.ndarray
    histogram_after: np.ndarray
    # Auto-detection results (if modality was auto-detected)
    detected_modality: ImageModality | None = None
    detection_confidence: float | None = None
    detection_features: dict | None = None


class ImagePreprocessor:
    """
    Preprocessing pipeline for microscopy images.

    Applies a sequence of filters tailored to the image modality
    to prepare images for optimal cell segmentation.

    Example:
        >>> preprocessor = ImagePreprocessor()
        >>> result = preprocessor.process(image, modality="brightfield")
        >>> processed_image = result.image
    """

    def __init__(self):
        self.applied_filters: list[str] = []

    # =========================================================================
    # WHITE BALANCE CORRECTION
    # =========================================================================

    def white_balance_gray_world(self, image: np.ndarray) -> np.ndarray:
        """
        Gray World white balance correction.

        Assumes the average color in the image should be neutral gray.
        Works well for images with diverse colors.

        Best for: General microscopy images with color cast.
        """
        if len(image.shape) != 3:
            return image  # Grayscale, no correction needed

        self.applied_filters.append("white_balance_gray_world")

        result = image.astype(np.float32)

        # Calculate mean of each channel
        avg_r = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])

        # Target: average gray value
        avg_gray = (avg_r + avg_g + avg_b) / 3.0

        # Scale each channel to match target
        if avg_r > 0:
            result[:, :, 0] *= avg_gray / avg_r
        if avg_g > 0:
            result[:, :, 1] *= avg_gray / avg_g
        if avg_b > 0:
            result[:, :, 2] *= avg_gray / avg_b

        return np.clip(result, 0, 255).astype(np.uint8)

    def white_balance_white_patch(self, image: np.ndarray, percentile: float = 99) -> np.ndarray:
        """
        White Patch (Max RGB) white balance correction.

        Assumes the brightest pixels in the image should be white.
        Uses percentile to be robust against outliers.

        Best for: H&E images where background should be white.
        """
        if len(image.shape) != 3:
            return image

        self.applied_filters.append(f"white_balance_white_patch(p={percentile})")

        result = image.astype(np.float32)

        # Find the bright reference point (percentile to avoid outliers)
        max_r = np.percentile(result[:, :, 0], percentile)
        max_g = np.percentile(result[:, :, 1], percentile)
        max_b = np.percentile(result[:, :, 2], percentile)

        # Scale to make the brightest point white (255)
        if max_r > 0:
            result[:, :, 0] *= 255.0 / max_r
        if max_g > 0:
            result[:, :, 1] *= 255.0 / max_g
        if max_b > 0:
            result[:, :, 2] *= 255.0 / max_b

        return np.clip(result, 0, 255).astype(np.uint8)

    def white_balance_adaptive(
        self,
        image: np.ndarray,
        method: Literal["gray_world", "white_patch", "combined"] = "combined"
    ) -> np.ndarray:
        """
        Adaptive white balance with automatic method selection.

        For H&E images, detects background regions and uses them as white reference.
        This is the recommended method for histology images.

        Args:
            image: RGB input image
            method: "gray_world", "white_patch", or "combined" (default)
        """
        if len(image.shape) != 3:
            return image

        self.applied_filters.append(f"white_balance_adaptive({method})")

        if method == "gray_world":
            return self.white_balance_gray_world(image)
        elif method == "white_patch":
            return self.white_balance_white_patch(image)

        # Combined method: detect background and use as white reference
        result = image.astype(np.float32)

        # Convert to grayscale to find bright regions (background in H&E)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find background pixels (bright regions, typically > 200 in H&E)
        # Use Otsu to find threshold adaptively
        threshold = filters.threshold_otsu(gray)
        # Background is brighter than tissue
        bright_threshold = max(threshold, 180)
        background_mask = gray > bright_threshold

        # If we found enough background pixels, use them as reference
        if np.sum(background_mask) > (image.shape[0] * image.shape[1] * 0.05):  # >5% background
            # Get mean color of background
            bg_r = np.mean(result[:, :, 0][background_mask])
            bg_g = np.mean(result[:, :, 1][background_mask])
            bg_b = np.mean(result[:, :, 2][background_mask])

            # Scale to make background white (target: 240 to avoid clipping)
            target = 240.0
            if bg_r > 0:
                result[:, :, 0] *= target / bg_r
            if bg_g > 0:
                result[:, :, 1] *= target / bg_g
            if bg_b > 0:
                result[:, :, 2] *= target / bg_b
        else:
            # Fall back to gray world if not enough background
            return self.white_balance_gray_world(image)

        return np.clip(result, 0, 255).astype(np.uint8)

    def white_balance_histology(self, image: np.ndarray) -> np.ndarray:
        """
        Specialized white balance for histology images with strong color cast.

        This method is designed for H&E images captured with improper white balance
        (common with phone cameras or miscalibrated microscopes).

        Uses LAB color space for more perceptually accurate correction.
        """
        if len(image.shape) != 3:
            return image

        self.applied_filters.append("white_balance_histology")

        # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Find the brightest 10% of pixels (likely background)
        l_channel = lab[:, :, 0]
        bright_threshold = np.percentile(l_channel, 90)
        bright_mask = l_channel >= bright_threshold

        if np.sum(bright_mask) > 100:  # Need at least 100 pixels
            # Get the A and B values of bright pixels
            a_bg = np.mean(lab[:, :, 1][bright_mask])
            b_bg = np.mean(lab[:, :, 2][bright_mask])

            # In LAB, neutral gray is A=128, B=128
            # Shift all pixels to neutralize the background color cast
            lab[:, :, 1] = lab[:, :, 1] - (a_bg - 128)
            lab[:, :, 2] = lab[:, :, 2] - (b_bg - 128)

        # Clip to valid range
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)

        # Convert back to RGB
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return result

    # =========================================================================
    # DENOISING FILTERS
    # =========================================================================

    def denoise_gaussian(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction.

        Best for: General noise reduction, smooth noise patterns.
        Trade-off: May blur cell edges at high sigma values.

        Args:
            image: Input image (2D or 3D array)
            sigma: Standard deviation of Gaussian kernel (1-3 recommended)
        """
        self.applied_filters.append(f"gaussian_blur(sigma={sigma})")
        return gaussian(image, sigma=sigma, preserve_range=True)

    def denoise_median(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Apply median filter to remove salt-and-pepper noise.

        Best for: Salt-and-pepper noise, preserves edges better than Gaussian.
        Recommended: size=3 for minor noise, size=5 for heavy noise.
        """
        self.applied_filters.append(f"median_filter(size={size})")
        if image.ndim == 3:
            # Apply to each channel separately
            return np.stack([
                ndimage.median_filter(image[:, :, c], size=size)
                for c in range(image.shape[2])
            ], axis=2)
        return ndimage.median_filter(image, size=size)

    def denoise_bilateral(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """
        Apply bilateral filter - edge-preserving denoising.

        Best for: Preserving sharp cell boundaries while removing noise.
        This is often the best choice for cell segmentation.

        Args:
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
        """
        self.applied_filters.append(f"bilateral(d={d})")

        # Convert to uint8 if needed (bilateral requires it)
        if image.dtype != np.uint8:
            img_uint8 = self._normalize_to_uint8(image)
        else:
            img_uint8 = image

        result = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)

        # Convert back to original dtype
        if image.dtype != np.uint8:
            return result.astype(image.dtype) * (image.max() / 255.0)
        return result

    def denoise_nlm(
        self,
        image: np.ndarray,
        h: float = 10,
        template_window: int = 7,
        search_window: int = 21
    ) -> np.ndarray:
        """
        Non-Local Means denoising - high quality but slower.

        Best for: High-quality results when processing time is not critical.
        Most effective for repetitive textures in cells.

        Args:
            h: Filter strength (10-20 recommended)
            template_window: Size of template patch
            search_window: Size of area where search is performed
        """
        self.applied_filters.append(f"nlm_denoise(h={h})")

        img_uint8 = self._normalize_to_uint8(image)

        if len(img_uint8.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(
                img_uint8, None, h, h, template_window, search_window
            )
        else:
            result = cv2.fastNlMeansDenoising(
                img_uint8, None, h, template_window, search_window
            )

        if image.dtype != np.uint8:
            return result.astype(image.dtype) * (image.max() / 255.0)
        return result

    # =========================================================================
    # CONTRAST ENHANCEMENT
    # =========================================================================

    def enhance_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Best for: Brightfield and H&E images with uneven illumination.
        Enhances local contrast without amplifying noise excessively.

        Args:
            clip_limit: Threshold for contrast limiting (1-4 recommended)
            tile_grid_size: Size of grid for histogram equalization
        """
        self.applied_filters.append(f"clahe(clip={clip_limit})")

        img_uint8 = self._normalize_to_uint8(image)

        if len(img_uint8.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            result = clahe.apply(img_uint8)

        if image.dtype != np.uint8:
            return result.astype(np.float32) / 255.0
        return result

    def enhance_adaptive_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        Automatic gamma correction based on image statistics.

        Adjusts brightness adaptively - brightens dark images,
        tones down overexposed images.
        """
        self.applied_filters.append("adaptive_gamma")

        # Calculate optimal gamma based on mean intensity
        mean_intensity = np.mean(image)

        if image.dtype == np.uint8:
            normalized_mean = mean_intensity / 255.0
        else:
            normalized_mean = mean_intensity / image.max() if image.max() > 0 else 0.5

        # Gamma < 1 brightens, > 1 darkens
        gamma = np.log(0.5) / np.log(normalized_mean + 0.001)
        gamma = np.clip(gamma, 0.5, 2.5)  # Limit correction range

        return exposure.adjust_gamma(image, gamma)

    def enhance_contrast_stretching(
        self,
        image: np.ndarray,
        low_percentile: float = 2,
        high_percentile: float = 98
    ) -> np.ndarray:
        """
        Stretch contrast using percentile-based intensity rescaling.

        Robust to outliers - ignores extreme pixel values.
        """
        self.applied_filters.append(f"contrast_stretch({low_percentile}-{high_percentile})")

        p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
        return exposure.rescale_intensity(image, in_range=(p_low, p_high))

    # =========================================================================
    # MODALITY-SPECIFIC PREPROCESSING
    # =========================================================================

    def preprocess_brightfield(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized pipeline for brightfield microscopy.

        Steps:
        1. Background subtraction (rolling ball)
        2. CLAHE for local contrast
        3. Light denoising
        """
        self.applied_filters.append("brightfield_pipeline")

        # Rolling ball background subtraction
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Estimate background with large morphological opening
        background = morphology.opening(gray, morphology.disk(50))
        subtracted = gray.astype(np.float32) - background.astype(np.float32)
        subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)

        # Enhance contrast
        enhanced = self.enhance_clahe(subtracted, clip_limit=2.0)

        # Light denoising
        return self.denoise_bilateral(enhanced, d=5)

    def preprocess_fluorescence(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized pipeline for fluorescence microscopy.

        Steps:
        1. Background subtraction
        2. Strong denoising (fluorescence typically has more noise)
        3. Intensity normalization
        """
        self.applied_filters.append("fluorescence_pipeline")

        # Fluorescence often needs stronger denoising
        denoised = self.denoise_nlm(image, h=15)

        # Subtract background (rolling ball with smaller radius)
        if len(denoised.shape) == 3:
            gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        else:
            gray = denoised

        background = morphology.opening(gray, morphology.disk(30))
        subtracted = gray.astype(np.float32) - background.astype(np.float32)
        subtracted = np.clip(subtracted, 0, subtracted.max())

        # Normalize intensity
        return self.enhance_contrast_stretching(subtracted)

    def preprocess_phase_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized pipeline for phase contrast microscopy.

        Phase contrast images have characteristic halos around cells.
        This pipeline reduces halos while preserving cell boundaries.
        """
        self.applied_filters.append("phase_contrast_pipeline")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Median filter to reduce halo artifacts
        denoised = self.denoise_median(gray, size=5)

        # Unsharp masking to enhance edges
        blurred = gaussian(denoised, sigma=3)
        sharpened = denoised + 0.5 * (denoised - blurred)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # CLAHE for local contrast
        return self.enhance_clahe(sharpened, clip_limit=3.0)

    def preprocess_he_stained(
        self,
        image: np.ndarray,
        use_stain_normalization: bool = True
    ) -> np.ndarray:
        """
        Optimized pipeline for H&E stained histology images.

        Pipeline:
        1. White balance correction (fixes color cast from camera/microscope)
        2. Macenko stain normalization (standardizes H&E colors across slides)
        3. Color deconvolution to separate H&E channels
        4. Extract hematoxylin channel (highlights nuclei)
        5. Contrast enhancement (CLAHE)

        Args:
            image: RGB input image
            use_stain_normalization: Whether to apply Macenko normalization

        Returns:
            Grayscale hematoxylin channel optimized for nuclei segmentation
        """
        self.applied_filters.append("he_stained_pipeline")

        if len(image.shape) != 3:
            raise ValueError("H&E preprocessing requires RGB image")

        # Step 1: WHITE BALANCE CORRECTION
        # Fix color cast from camera/microscope (green tint, etc.)
        corrected = self.white_balance_histology(image)

        # Step 2: MACENKO STAIN NORMALIZATION (new!)
        # Standardize H&E colors to reference standard
        # This ensures consistent segmentation across different slides
        if use_stain_normalization:
            try:
                normalizer = MacenkoNormalizer()
                normalized = normalizer.normalize(corrected)
                self.applied_filters.append("macenko_normalization")
            except Exception as e:
                # If normalization fails, continue with just white-balanced image
                normalized = corrected
                self.applied_filters.append(f"macenko_skipped:{str(e)[:30]}")
        else:
            normalized = corrected

        # Step 3: Color deconvolution for H&E
        # Standard H&E stain vectors (Ruifrok and Johnston)
        stain_matrix = np.array([
            [0.65, 0.70, 0.29],   # Hematoxylin (blue-purple)
            [0.07, 0.99, 0.11],   # Eosin (pink)
            [0.27, 0.57, 0.78]    # Residual/DAB
        ])

        # Convert to optical density (OD = -log10(I/I0))
        image_od = -np.log10((normalized.astype(np.float32) + 1) / 256)

        # Deconvolve using pseudo-inverse
        stain_matrix_inv = np.linalg.pinv(stain_matrix)
        deconvolved = np.dot(image_od.reshape(-1, 3), stain_matrix_inv.T)
        deconvolved = deconvolved.reshape(normalized.shape)

        # Step 4: Extract hematoxylin channel (nuclei have high OD values)
        hematoxylin = deconvolved[:, :, 0]

        # Clip negative values
        hematoxylin = np.clip(hematoxylin, 0, None)

        # Rescale to use full dynamic range (0-255)
        # High OD = high stain = nuclei = bright in output
        hematoxylin = exposure.rescale_intensity(hematoxylin, out_range=(0, 255))

        # Step 5: Enhance contrast with CLAHE
        enhanced = self.enhance_clahe(hematoxylin.astype(np.uint8), clip_limit=3.0)

        return enhanced

    # =========================================================================
    # MAIN PROCESSING INTERFACE
    # =========================================================================

    def process(
        self,
        image: np.ndarray,
        modality: ImageModality | str = ImageModality.BRIGHTFIELD,
        custom_pipeline: list[str] | None = None
    ) -> PreprocessingResult:
        """
        Process an image with the appropriate preprocessing pipeline.

        Args:
            image: Input image (HxW or HxWxC)
            modality: Type of microscopy image, or "auto" for auto-detection
            custom_pipeline: Optional list of filter names to apply

        Returns:
            PreprocessingResult with processed image and metadata
        """
        self.applied_filters = []

        # Store original histogram
        if len(image.shape) == 3:
            hist_before = np.histogram(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), bins=256)[0]
        else:
            hist_before = np.histogram(image, bins=256)[0]

        # Ensure modality is enum
        if isinstance(modality, str):
            modality = ImageModality(modality.lower())

        # Auto-detect modality if requested
        detected_modality = None
        detection_confidence = None
        detection_features = None

        if modality == ImageModality.AUTO:
            detected_modality, detection_confidence, detection_features = detect_image_modality(image)
            modality = detected_modality
            self.applied_filters.append(f"auto_detected:{modality.value}(conf={detection_confidence:.2f})")

        # Apply modality-specific pipeline
        if custom_pipeline:
            processed = self._apply_custom_pipeline(image, custom_pipeline)
        else:
            if modality == ImageModality.BRIGHTFIELD:
                processed = self.preprocess_brightfield(image)
            elif modality == ImageModality.FLUORESCENCE:
                processed = self.preprocess_fluorescence(image)
            elif modality == ImageModality.PHASE_CONTRAST:
                processed = self.preprocess_phase_contrast(image)
            elif modality == ImageModality.HE_STAINED:
                processed = self.preprocess_he_stained(image)
            else:
                # Default: basic denoising and enhancement
                processed = self.denoise_bilateral(image)
                processed = self.enhance_clahe(processed)

        # Store processed histogram
        if len(processed.shape) == 3:
            hist_after = np.histogram(cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY), bins=256)[0]
        else:
            hist_after = np.histogram(processed, bins=256)[0]

        return PreprocessingResult(
            image=processed,
            original_shape=image.shape,
            applied_filters=self.applied_filters,
            modality=modality,
            histogram_before=hist_before,
            histogram_after=hist_after,
            detected_modality=detected_modality,
            detection_confidence=detection_confidence,
            detection_features=detection_features
        )

    def _apply_custom_pipeline(
        self,
        image: np.ndarray,
        pipeline: list[str]
    ) -> np.ndarray:
        """Apply a custom sequence of filters."""
        result = image.copy()

        filter_map = {
            # White balance
            "white_balance": lambda img: self.white_balance_adaptive(img),
            "white_balance_gray_world": lambda img: self.white_balance_gray_world(img),
            "white_balance_white_patch": lambda img: self.white_balance_white_patch(img),
            "white_balance_histology": lambda img: self.white_balance_histology(img),
            # Denoising
            "gaussian": lambda img: self.denoise_gaussian(img),
            "median": lambda img: self.denoise_median(img),
            "bilateral": lambda img: self.denoise_bilateral(img),
            "nlm": lambda img: self.denoise_nlm(img),
            # Enhancement
            "clahe": lambda img: self.enhance_clahe(img),
            "gamma": lambda img: self.enhance_adaptive_gamma(img),
            "contrast_stretch": lambda img: self.enhance_contrast_stretching(img),
        }

        for filter_name in pipeline:
            if filter_name in filter_map:
                result = filter_map[filter_name](result)
            else:
                raise ValueError(f"Unknown filter: {filter_name}")

        return result

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """Convert any image to uint8 format."""
        if image.dtype == np.uint8:
            return image

        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)

        return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)


# Convenience function for quick processing
def preprocess_image(
    image: np.ndarray,
    modality: str = "brightfield"
) -> np.ndarray:
    """
    Quick preprocessing function.

    Args:
        image: Input microscopy image
        modality: One of "brightfield", "fluorescence", "phase_contrast", "he_stained"

    Returns:
        Preprocessed image ready for segmentation
    """
    preprocessor = ImagePreprocessor()
    result = preprocessor.process(image, modality=modality)
    return result.image
