"""
Visualization Section
=====================
Segmentatsiya vizualizatsiyasi bo'limi.
"""

from typing import List, Any, Optional
import numpy as np

from .base import BaseSection


class VisualizationSection(BaseSection):
    """
    Vizualizatsiya bo'limi.

    Asl rasm va segmentatsiya overlay.
    """

    def build(
        self,
        original_image: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Any]:
        """
        Vizualizatsiya bo'limini yaratish.

        Args:
            original_image: Asl rasm
            masks: Segmentatsiya maskalari

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        if original_image is None or masks is None:
            return elements

        # Section header
        header = self.get_label('segmentation_results', "Segmentatsiya Natijalari")
        elements.append(self.text_builder.section_header(header))

        # Chart labels
        chart_labels = self.labels.get('charts', {})

        # Build overlay image
        overlay = self.chart_builder.build_segmentation_overlay(
            original_image,
            masks,
            chart_labels
        )

        if overlay:
            elements.append(overlay)
            elements.append(self.text_builder.spacer(15))

        return elements
