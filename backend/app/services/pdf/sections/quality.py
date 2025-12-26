"""
Quality Section
===============
Sifat baholash bo'limi.
"""

from typing import List, Any
import pandas as pd

from .base import BaseSection


class QualitySection(BaseSection):
    """
    Sifat baholash bo'limi.

    Hujayra soni, segmentatsiya sifati va boshqalar.
    """

    def build(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> List[Any]:
        """
        Sifat baholash bo'limini yaratish.

        Args:
            df: Ma'lumotlar DataFrame

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        # Section header
        header = self.get_label('quality_assessment')
        elements.append(self.text_builder.section_header(header))

        if len(df) == 0:
            no_cells = self.get_message('no_cells_detected', "Hujayralar aniqlanmadi.")
            elements.append(self.text_builder.body(no_cells))
            return elements

        # Quality ratings - use helper methods
        excellent = self.get_quality_label('excellent')
        good = self.get_quality_label('good')
        acceptable = self.get_quality_label('acceptable')
        poor = self.get_quality_label('poor')

        quality_items = []

        # Cell count assessment
        cell_count = len(df)
        cell_count_label = self.get_quality_label('cell_count')

        if cell_count >= 100:
            count_quality = excellent
        elif cell_count >= 30:
            count_quality = good
        elif cell_count >= 10:
            count_quality = acceptable
        else:
            count_quality = poor

        quality_items.append([cell_count_label, str(cell_count), count_quality])

        # Circularity assessment
        if 'circularity' in df.columns:
            circ_label = self.get_metric_label('circularity')
            mean_circ = df['circularity'].mean()

            if mean_circ >= 0.8:
                circ_quality = excellent
            elif mean_circ >= 0.6:
                circ_quality = good
            else:
                circ_quality = acceptable

            quality_items.append([circ_label, f"{mean_circ:.2f}", circ_quality])

        # Segmentation quality (based on solidity)
        if 'solidity' in df.columns:
            seg_label = self.get_quality_label('segmentation_quality')
            mean_solid = df['solidity'].mean()

            if mean_solid >= 0.95:
                seg_quality = excellent
            elif mean_solid >= 0.85:
                seg_quality = good
            else:
                seg_quality = acceptable

            quality_items.append([seg_label, f"{mean_solid:.2f}", seg_quality])

        # Build table - use labels from UZBEK_LABELS
        headers = [
            self.get_stat_label('metric'),
            self.get_label('quality.value', "Qiymat"),
            self.get_label('quality.rating', "Baho")
        ]

        table = self.table_builder.build_quality_table(headers, quality_items)
        elements.append(table)

        elements.append(self.text_builder.spacer(15))

        return elements
