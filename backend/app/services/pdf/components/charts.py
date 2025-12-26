"""
Chart Builder Component
=======================
Grafik va diagrammalar yaratish uchun komponent.
"""

import io
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from reportlab.platypus import Image as RLImage
from reportlab.lib.colors import Color

from ..themes.base import BaseTheme


class ChartBuilder:
    """
    Grafik yaratuvchi.

    Gistogrammalar, overlay rasmlar va boshqa vizualizatsiyalar.
    """

    def __init__(self, theme: BaseTheme):
        self.theme = theme
        # Set matplotlib defaults
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def _color_to_hex(self, color: Color) -> str:
        """
        ReportLab Color obyektini hex stringga aylantirish.

        Args:
            color: ReportLab Color obyekti

        Returns:
            Hex rang kodi (#RRGGBB)
        """
        # hexval() returns '0xRRGGBB', matplotlib needs '#RRGGBB'
        hex_val = color.hexval()
        return '#' + hex_val[2:] if hex_val.startswith('0x') else hex_val

    def build_histogram_grid(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        labels: Dict[str, str],
        stat_labels: Dict[str, str],
        width: float = 450,
        height: float = 360
    ) -> Optional[RLImage]:
        """
        Gistogramma to'plamini yaratish.

        Args:
            df: Ma'lumotlar DataFrame
            metrics: Ko'rsatgichlar ro'yxati
            labels: Metrik nomlari lug'ati
            stat_labels: Statistik teglar lug'ati
            width: Rasm kengligi
            height: Rasm balandligi

        Returns:
            RLImage yoki None
        """
        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        # Color from theme - using clean method
        bar_color = self._color_to_hex(self.theme.colors.primary)

        for idx, metric in enumerate(available_metrics[:4]):
            values = df[metric].dropna()
            axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7, color=bar_color)

            metric_label = labels.get(metric, metric)
            axes[idx].set_xlabel(metric_label, fontsize=9)
            axes[idx].set_ylabel(stat_labels.get('count', 'Soni'), fontsize=9)
            axes[idx].set_title(f"{metric_label} taqsimoti", fontsize=10)

            # Statistics annotation
            mean_label = stat_labels.get('mean', "O'rtacha")
            std_label = stat_labels.get('std', "St. og'ish")
            stats_text = f"n={len(values)}\n{mean_label}={values.mean():.2f}\n{std_label}={values.std():.2f}"

            axes[idx].text(
                0.95, 0.95, stats_text,
                transform=axes[idx].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        # Hide unused subplots
        for idx in range(len(available_metrics), 4):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return RLImage(buf, width=width, height=height)

    def build_segmentation_overlay(
        self,
        original_image: np.ndarray,
        masks: np.ndarray,
        labels: Dict[str, str],
        width: float = 450,
        height: float = 225
    ) -> Optional[RLImage]:
        """
        Segmentatsiya overlay rasmini yaratish.

        Args:
            original_image: Asl rasm
            masks: Segmentatsiya maskalari
            labels: Teglar lug'ati
            width: Rasm kengligi
            height: Rasm balandligi

        Returns:
            RLImage yoki None
        """
        if original_image is None or masks is None:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        if original_image.ndim == 3:
            axes[0].imshow(original_image)
        else:
            axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(labels.get('original_image', 'Asl rasm'), fontsize=10)
        axes[0].axis('off')

        # Segmentation overlay
        if original_image.ndim == 3:
            axes[1].imshow(original_image)
        else:
            axes[1].imshow(original_image, cmap='gray')

        # Overlay colored masks
        cmap = plt.cm.Set3
        masked = np.ma.masked_where(masks == 0, masks)
        axes[1].imshow(masked, cmap=cmap, alpha=0.5)

        cells_label = labels.get('cells_detected', 'aniqlangan hujayralar')
        seg_label = labels.get('segmentation', 'Segmentatsiya')
        axes[1].set_title(f"{seg_label} ({masks.max()} {cells_label})", fontsize=10)
        axes[1].axis('off')

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return RLImage(buf, width=width, height=height)

    def build_single_histogram(
        self,
        values: np.ndarray,
        label: str,
        stat_labels: Dict[str, str],
        width: float = 300,
        height: float = 200
    ) -> Optional[RLImage]:
        """
        Bitta gistogramma yaratish.

        Args:
            values: Qiymatlar
            label: Ko'rsatkich nomi
            stat_labels: Statistik teglar
            width: Rasm kengligi
            height: Rasm balandligi

        Returns:
            RLImage yoki None
        """
        if len(values) == 0:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(stat_labels.get('count', 'Soni'), fontsize=9)
        ax.set_title(f"{label} taqsimoti", fontsize=10)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return RLImage(buf, width=width, height=height)
