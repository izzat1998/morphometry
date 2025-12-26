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


@dataclass
class PreprocessingResult:
    """Container for preprocessing output with metadata."""
    image: np.ndarray
    original_shape: tuple[int, ...]
    applied_filters: list[str]
    modality: ImageModality
    histogram_before: np.ndarray
    histogram_after: np.ndarray


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

    def preprocess_he_stained(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized pipeline for H&E stained histology images.

        Performs stain normalization and separates hematoxylin/eosin channels.
        Returns the hematoxylin channel which highlights nuclei.
        """
        self.applied_filters.append("he_stained_pipeline")

        if len(image.shape) != 3:
            raise ValueError("H&E preprocessing requires RGB image")

        # Color deconvolution for H&E
        # Standard H&E stain vectors (Ruifrok and Johnston)
        stain_matrix = np.array([
            [0.65, 0.70, 0.29],   # Hematoxylin
            [0.07, 0.99, 0.11],   # Eosin
            [0.27, 0.57, 0.78]    # DAB (background)
        ])

        # Convert to optical density
        image_od = -np.log10((image.astype(np.float32) + 1) / 256)

        # Deconvolve
        stain_matrix_inv = np.linalg.pinv(stain_matrix)
        deconvolved = np.dot(image_od.reshape(-1, 3), stain_matrix_inv.T)
        deconvolved = deconvolved.reshape(image.shape)

        # Extract hematoxylin channel (nuclei)
        hematoxylin = deconvolved[:, :, 0]
        hematoxylin = exposure.rescale_intensity(hematoxylin, out_range=(0, 255))

        # Enhance contrast
        return self.enhance_clahe(hematoxylin.astype(np.uint8), clip_limit=2.5)

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
            modality: Type of microscopy image
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
            histogram_after=hist_after
        )

    def _apply_custom_pipeline(
        self,
        image: np.ndarray,
        pipeline: list[str]
    ) -> np.ndarray:
        """Apply a custom sequence of filters."""
        result = image.copy()

        filter_map = {
            "gaussian": lambda img: self.denoise_gaussian(img),
            "median": lambda img: self.denoise_median(img),
            "bilateral": lambda img: self.denoise_bilateral(img),
            "nlm": lambda img: self.denoise_nlm(img),
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
