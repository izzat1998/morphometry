"""
Summary Section
===============
Umumiy statistika bo'limi.
"""

from typing import List, Any, Dict
import pandas as pd

from .base import BaseSection


class SummarySection(BaseSection):
    """
    Umumiy statistika bo'limi.

    Asosiy ko'rsatkichlarning statistik xulosasi.
    """

    # Default metrics to show in summary
    DEFAULT_METRICS = ['area', 'perimeter', 'circularity', 'eccentricity', 'solidity', 'aspect_ratio']

    def build(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, str] = None,
        **kwargs
    ) -> List[Any]:
        """
        Statistika bo'limini yaratish.

        Args:
            df: Ma'lumotlar DataFrame
            metrics: Ko'rsatkichlar lug'ati {internal_name: uzbek_label}

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        # Section header
        header = self.get_label('summary_statistics')
        elements.append(self.text_builder.section_header(header))

        if len(df) == 0:
            no_cells = self.get_message('no_cells_detected', "Hujayralar aniqlanmadi.")
            elements.append(self.text_builder.body(no_cells))
            return elements

        # Use provided metrics or build from labels
        if metrics is None:
            metrics = {}
            for metric in self.DEFAULT_METRICS:
                metrics[metric] = self.get_metric_label(metric)

        # Build headers using helper method
        headers = [
            self.get_stat_label('metric'),
            self.get_stat_label('mean'),
            self.get_stat_label('std'),
            self.get_stat_label('min'),
            self.get_stat_label('max'),
            self.get_stat_label('median')
        ]

        # Build data rows
        data = []
        for col, label in metrics.items():
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    data.append([
                        label,
                        f"{values.mean():.2f}",
                        f"{values.std():.2f}",
                        f"{values.min():.2f}",
                        f"{values.max():.2f}",
                        f"{values.median():.2f}"
                    ])

        if data:
            table = self.table_builder.build_statistics_table(headers, data)
            elements.append(table)

        elements.append(self.text_builder.spacer(15))

        return elements
