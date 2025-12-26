"""
Distribution Section
====================
Taqsimot tahlili bo'limi.
"""

from typing import List, Any, Dict
import pandas as pd

from .base import BaseSection


class DistributionSection(BaseSection):
    """
    Taqsimot tahlili bo'limi.

    Ko'rsatkichlar gistogrammalari.
    """

    # Default metrics for distribution analysis
    DEFAULT_METRICS = ['area', 'circularity', 'eccentricity', 'solidity']

    def build(
        self,
        df: pd.DataFrame,
        metrics: List[str] = None,
        metric_labels: Dict[str, str] = None,
        **kwargs
    ) -> List[Any]:
        """
        Taqsimot bo'limini yaratish.

        Args:
            df: Ma'lumotlar DataFrame
            metrics: Ko'rsatadigan ko'rsatkichlar
            metric_labels: Ko'rsatkich nomlari lug'ati

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        if len(df) == 0:
            return elements

        # Section header
        header = self.get_label('distribution_analysis')
        elements.append(self.text_builder.section_header(header))

        # Default metrics
        if metrics is None:
            metrics = self.DEFAULT_METRICS

        # Build labels dict from labels if not provided
        if metric_labels is None:
            metric_labels = {}
            for metric in metrics:
                metric_labels[metric] = self.get_metric_label(metric)

        # Stat labels for histogram
        stat_labels = {
            'count': self.get_stat_label('count'),
            'mean': self.get_stat_label('mean'),
            'std': self.get_stat_label('std'),
        }

        # Build histogram grid
        histogram = self.chart_builder.build_histogram_grid(
            df,
            metrics,
            metric_labels,
            stat_labels
        )

        if histogram:
            elements.append(histogram)

        elements.append(self.text_builder.spacer(15))

        return elements
