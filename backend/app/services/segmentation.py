"""
Cell Segmentation Service
=========================
Multi-model cell segmentation using state-of-the-art deep learning.

Supported Models:
- Cellpose (v4+): General-purpose cell and nuclei segmentation
- Cellpose-SAM: Zero-shot segmentation with superhuman generalization
- Classical methods: Watershed, Otsu thresholding (as fallbacks)

Key Insight:
    Cellpose 4.x includes Cellpose-SAM which combines the original Cellpose
    with Meta's Segment Anything Model. This provides excellent results
    across diverse cell types without retraining.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import logging

# Import cellpose (handles GPU detection automatically)
try:
    from cellpose import models, core
    CELLPOSE_AVAILABLE = True
    GPU_AVAILABLE = core.use_gpu()
except ImportError:
    CELLPOSE_AVAILABLE = False
    GPU_AVAILABLE = False

# Classical segmentation fallbacks
from skimage import segmentation, filters, morphology, measure
from scipy import ndimage

logger = logging.getLogger(__name__)


class SegmentationModel(str, Enum):
    """Available segmentation models."""
    CELLPOSE_CYTO3 = "cyto3"          # Best for cytoplasm (recommended)
    CELLPOSE_NUCLEI = "nuclei"         # Best for nuclei only
    CELLPOSE_CYTO2 = "cyto2"           # Legacy cytoplasm model
    CELLPOSE_SAM = "cpsam"             # Cellpose-SAM (superhuman generalization)
    WATERSHED = "watershed"            # Classical watershed
    OTSU = "otsu"                      # Simple thresholding


# Recommended diameters by image modality
# These are tuned for typical microscopy images
DIAMETER_BY_MODALITY = {
    'brightfield': 30.0,      # Standard cell culture
    'fluorescence': 30.0,     # Fluorescent markers
    'phase_contrast': 30.0,   # Phase contrast microscopy
    'he_stain': 10.0,         # H&E histology - small nuclei!
    'he_stained': 10.0,       # Alias
    'histology': 10.0,        # Alias
    'default': 30.0,          # Fallback
}

# Recommended diameters by model type
DIAMETER_BY_MODEL = {
    'nuclei': 17.0,           # Nuclei are smaller
    'cyto3': 30.0,            # Cytoplasm
    'cyto2': 30.0,            # Cytoplasm
    'cpsam': 30.0,            # Cellpose-SAM
}


@dataclass
class SegmentationResult:
    """Container for segmentation output."""
    masks: np.ndarray                  # Label image (0=background, 1..N=cells)
    outlines: np.ndarray               # Binary outline image
    cell_count: int                    # Number of detected cells
    model_used: str                    # Model name
    diameter_used: float               # Cell diameter used
    confidence_scores: list[float] = field(default_factory=list)  # Per-cell confidence
    flows: np.ndarray | None = None    # Optional flow fields (for visualization)


class CellSegmenter:
    """
    Unified interface for cell segmentation.

    Automatically selects the best available model and handles
    GPU/CPU fallback gracefully.

    Example:
        >>> segmenter = CellSegmenter(model="cyto3")
        >>> result = segmenter.segment(preprocessed_image)
        >>> print(f"Found {result.cell_count} cells")
    """

    def __init__(
        self,
        model: SegmentationModel | str = SegmentationModel.CELLPOSE_CYTO3,
        use_gpu: bool = True,
        diameter: float | None = None
    ):
        """
        Initialize the segmenter.

        Args:
            model: Segmentation model to use
            use_gpu: Whether to use GPU acceleration
            diameter: Expected cell diameter in pixels (None = auto-detect)
        """
        if isinstance(model, str):
            model = SegmentationModel(model)

        self.model_type = model
        self.diameter = diameter
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self._cellpose_model = None

        if not CELLPOSE_AVAILABLE and model.value.startswith("c"):
            logger.warning(
                "Cellpose not available. Install with: pip install cellpose"
            )

    def _load_cellpose_model(self) -> None:
        """Lazy-load Cellpose model (saves memory until needed)."""
        if self._cellpose_model is not None:
            return

        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is required for this model. Install with: pip install cellpose")

        model_name = self.model_type.value

        logger.info(f"Loading Cellpose model '{model_name}' (GPU: {self.use_gpu})")

        # Cellpose 4.x uses pretrained_model parameter (not model_type!)
        # model_type is deprecated and ignored in v4.0.1+
        self._cellpose_model = models.CellposeModel(
            pretrained_model=model_name,
            gpu=self.use_gpu
        )

        # Default diameter is set later based on context (see set_diameter_for_modality)

    def segment(
        self,
        image: np.ndarray,
        channels: list[int] | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15
    ) -> SegmentationResult:
        """
        Segment cells in an image.

        Args:
            image: Input image (HxW or HxWxC)
            channels: Channel configuration [cytoplasm, nucleus]
                     - [0, 0]: grayscale
                     - [1, 0]: red channel cytoplasm
                     - [2, 3]: green cyto, blue nuclei
            flow_threshold: Flow error threshold (lower = stricter)
            cellprob_threshold: Cell probability threshold
            min_size: Minimum cell size in pixels

        Returns:
            SegmentationResult with masks and metadata
        """
        # Use classical methods if not using Cellpose
        if self.model_type == SegmentationModel.WATERSHED:
            return self._segment_watershed(image, min_size)
        elif self.model_type == SegmentationModel.OTSU:
            return self._segment_otsu(image, min_size)

        # Cellpose segmentation
        self._load_cellpose_model()

        # Auto-detect channels if not specified
        if channels is None:
            channels = self._detect_channels(image)

        # Run segmentation
        logger.info(f"Running segmentation with diameter={self.diameter}")

        # Call the Cellpose model's inference method
        # Note: .eval() is Cellpose's standard API method name, not Python's eval()
        inference_result = self._cellpose_model.eval(
            image,
            diameter=self.diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size
        )

        # Unpack results (format varies by model type)
        if len(inference_result) == 4:
            masks, flows, styles, diams = inference_result
        else:
            masks, flows, styles = inference_result
            diams = self.diameter or 30.0

        # Generate outlines
        outlines = self._masks_to_outlines(masks)

        # Count cells
        cell_count = masks.max()

        # Calculate confidence scores from flows if available
        confidence_scores = self._calculate_confidence(masks, flows)

        return SegmentationResult(
            masks=masks,
            outlines=outlines,
            cell_count=cell_count,
            model_used=self.model_type.value,
            diameter_used=float(diams) if isinstance(diams, (int, float)) else float(diams[0]) if len(diams) > 0 else 30.0,
            confidence_scores=confidence_scores,
            flows=flows[0] if flows else None
        )

    def _detect_channels(self, image: np.ndarray) -> list[int]:
        """Auto-detect appropriate channel configuration."""
        if image.ndim == 2:
            return [0, 0]  # Grayscale

        if image.shape[2] == 1:
            return [0, 0]  # Single channel

        if image.shape[2] >= 3:
            # Check if it's a true color image or pseudo-color
            # For RGB images, use grayscale mode
            return [0, 0]

        return [0, 0]

    def _masks_to_outlines(self, masks: np.ndarray) -> np.ndarray:
        """Convert label masks to binary outlines."""
        outlines = np.zeros(masks.shape, dtype=bool)

        for label in range(1, int(masks.max()) + 1):
            mask = (masks == label).astype(np.uint8)
            # Find edges using morphological gradient
            dilated = morphology.dilation(mask)
            eroded = morphology.erosion(mask)
            outline = (dilated > 0) & (eroded == 0)
            outlines |= outline

        return outlines.astype(np.uint8) * 255

    def _calculate_confidence(
        self,
        masks: np.ndarray,
        flows: list | None
    ) -> list[float]:
        """
        Calculate per-cell confidence scores using multiple factors.

        Confidence is computed from:
        1. Cell probability (cellprob) - Cellpose's confidence that pixels are cells
        2. Flow consistency - How well flow vectors converge to cell center
        3. Morphology - Shape regularity (circularity, solidity)

        Each factor is weighted to produce a final 0-1 confidence score.

        Args:
            masks: Label image where each cell has a unique integer ID
            flows: Cellpose output [flow_field, cellprob, style] or None

        Returns:
            List of confidence scores, one per cell (empty if flows unavailable)
        """
        if masks.max() == 0:
            return []

        # Ensure masks are integer type
        masks_int = masks.astype(np.int32)

        # Get region properties for morphology calculations
        props_list = measure.regionprops(masks_int)

        # Extract flow components if available
        cellprob = None
        flow_field = None

        if flows is not None and len(flows) >= 2:
            # flows[0] = flow field (2, H, W) - Y and X gradients
            # flows[1] = cell probability map (H, W)
            # flows[2] = style vectors (optional)
            flow_field = flows[0] if len(flows[0].shape) == 3 else None
            cellprob = flows[1] if len(flows) > 1 else None

        confidence_scores = []

        for props in props_list:
            label = props.label
            cell_mask = (masks_int == label)

            # === Factor 1: Cell Probability (weight: 0.4) ===
            # Mean cellprob within the cell region
            cellprob_score = 0.5  # Default if not available
            if cellprob is not None:
                try:
                    cell_probs = cellprob[cell_mask]
                    if len(cell_probs) > 0:
                        # Cellprob is typically in range [-6, 6], sigmoid to [0, 1]
                        mean_prob = np.mean(cell_probs)
                        # Convert to 0-1 range using sigmoid
                        cellprob_score = 1 / (1 + np.exp(-mean_prob))
                except (IndexError, ValueError):
                    pass

            # === Factor 2: Flow Consistency (weight: 0.3) ===
            # Measures how well flow vectors point toward cell center
            flow_score = 0.5  # Default if not available
            if flow_field is not None and flow_field.shape[0] >= 2:
                try:
                    flow_score = self._compute_flow_consistency(
                        flow_field, cell_mask, props.centroid
                    )
                except (IndexError, ValueError):
                    pass

            # === Factor 3: Morphology Score (weight: 0.3) ===
            # Combines circularity and solidity
            morph_score = self._compute_morphology_score(props)

            # === Weighted combination ===
            # Weights emphasize cellprob (most reliable from Cellpose)
            confidence = (
                0.4 * cellprob_score +
                0.3 * flow_score +
                0.3 * morph_score
            )

            # Clamp to [0, 1]
            confidence = float(np.clip(confidence, 0.0, 1.0))
            confidence_scores.append(confidence)

        return confidence_scores

    def _compute_flow_consistency(
        self,
        flow_field: np.ndarray,
        cell_mask: np.ndarray,
        centroid: tuple[float, float]
    ) -> float:
        """
        Compute flow consistency score for a cell.

        Measures how well the flow vectors within a cell point toward
        the cell's centroid. High consistency = high confidence.

        Args:
            flow_field: (2, H, W) array of Y and X flow components
            cell_mask: Boolean mask for this cell
            centroid: (row, col) centroid of the cell

        Returns:
            Flow consistency score in [0, 1]
        """
        # Get coordinates of cell pixels
        rows, cols = np.where(cell_mask)
        if len(rows) < 4:  # Too few pixels to compute meaningful flow
            return 0.5

        # Get flow vectors at cell pixels
        flow_y = flow_field[0][cell_mask]
        flow_x = flow_field[1][cell_mask]

        # Compute expected flow direction (toward centroid)
        cy, cx = centroid
        expected_y = cy - rows
        expected_x = cx - cols

        # Normalize expected vectors
        expected_mag = np.sqrt(expected_y**2 + expected_x**2) + 1e-8
        expected_y = expected_y / expected_mag
        expected_x = expected_x / expected_mag

        # Normalize actual flow vectors
        flow_mag = np.sqrt(flow_y**2 + flow_x**2) + 1e-8
        flow_y_norm = flow_y / flow_mag
        flow_x_norm = flow_x / flow_mag

        # Compute cosine similarity (dot product of unit vectors)
        # Values range from -1 (opposite) to 1 (same direction)
        cos_sim = flow_y_norm * expected_y + flow_x_norm * expected_x

        # Average cosine similarity, shifted to [0, 1]
        mean_cos_sim = np.mean(cos_sim)
        flow_score = (mean_cos_sim + 1) / 2  # Map [-1, 1] to [0, 1]

        return float(flow_score)

    def _compute_morphology_score(self, props) -> float:
        """
        Compute morphology-based confidence score.

        Combines circularity and solidity to assess shape regularity.
        Real cells tend to have smooth, convex boundaries.

        Args:
            props: regionprops object for the cell

        Returns:
            Morphology score in [0, 1]
        """
        area = props.area
        perimeter = props.perimeter

        # Circularity: 1.0 for perfect circle, lower for irregular shapes
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularity = min(circularity, 1.0)  # Cap at 1.0
        else:
            circularity = 0.5

        # Solidity: ratio of area to convex hull area
        # High solidity = fewer concavities = more cell-like
        try:
            solidity = props.solidity
        except (AttributeError, ZeroDivisionError):
            solidity = 0.5

        # Combine: weight solidity slightly higher (cells should be convex)
        morph_score = 0.4 * circularity + 0.6 * solidity

        return float(morph_score)

    # =========================================================================
    # CLASSICAL SEGMENTATION METHODS (Fallbacks)
    # =========================================================================

    def _segment_watershed(
        self,
        image: np.ndarray,
        min_size: int = 15
    ) -> SegmentationResult:
        """
        Watershed segmentation - classical method.

        Works well for well-separated cells but struggles with touching cells.
        Use as fallback when deep learning is unavailable.
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        # Threshold using Otsu
        threshold = filters.threshold_otsu(gray)
        binary = gray > threshold

        # Distance transform
        distance = ndimage.distance_transform_edt(binary)

        # Find local maxima as markers
        from skimage.feature import peak_local_max
        coords = peak_local_max(
            distance,
            min_distance=20,
            threshold_abs=5,
            labels=binary
        )

        # Create markers
        markers = np.zeros(binary.shape, dtype=int)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

        # Watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)

        # Remove small objects
        labels = morphology.remove_small_objects(labels, min_size=min_size)

        # Relabel
        labels = measure.label(labels > 0)

        outlines = self._masks_to_outlines(labels)

        return SegmentationResult(
            masks=labels,
            outlines=outlines,
            cell_count=labels.max(),
            model_used="watershed",
            diameter_used=30.0,
            confidence_scores=[]
        )

    def _segment_otsu(
        self,
        image: np.ndarray,
        min_size: int = 15
    ) -> SegmentationResult:
        """
        Simple Otsu thresholding segmentation.

        Best for high-contrast images with clear foreground/background.
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        # Otsu threshold
        threshold = filters.threshold_otsu(gray)
        binary = gray > threshold

        # Clean up
        binary = morphology.remove_small_holes(binary, area_threshold=100)
        binary = morphology.remove_small_objects(binary, min_size=min_size)

        # Label connected components
        labels = measure.label(binary)

        outlines = self._masks_to_outlines(labels)

        return SegmentationResult(
            masks=labels,
            outlines=outlines,
            cell_count=labels.max(),
            model_used="otsu",
            diameter_used=30.0,
            confidence_scores=[]
        )


# =========================================================================
# BATCH PROCESSING
# =========================================================================

def segment_batch(
    images: list[np.ndarray],
    model: str = "cyto3",
    diameter: float | None = None,
    use_gpu: bool = True
) -> list[SegmentationResult]:
    """
    Segment multiple images efficiently.

    Loads the model once and processes all images.

    Args:
        images: List of input images
        model: Segmentation model name
        diameter: Expected cell diameter
        use_gpu: Whether to use GPU

    Returns:
        List of SegmentationResult for each image
    """
    segmenter = CellSegmenter(model=model, diameter=diameter, use_gpu=use_gpu)

    results = []
    for i, image in enumerate(images):
        logger.info(f"Processing image {i+1}/{len(images)}")
        result = segmenter.segment(image)
        results.append(result)

    return results


# Convenience function
def segment_cells(
    image: np.ndarray,
    model: str = "cyto3",
    diameter: float | None = None
) -> tuple[np.ndarray, int]:
    """
    Quick cell segmentation function.

    Args:
        image: Input image
        model: Model name ("cyto3", "nuclei", "cpsam", etc.)
        diameter: Cell diameter (None = auto)

    Returns:
        Tuple of (masks, cell_count)
    """
    segmenter = CellSegmenter(model=model, diameter=diameter)
    result = segmenter.segment(image)
    return result.masks, result.cell_count
