"""
Conclusions Section
===================
Xulosa va tavsiyalar bo'limi.
"""

from typing import List, Any, Dict
import pandas as pd

from .base import BaseSection


class ConclusionsSection(BaseSection):
    """
    Xulosa va tavsiyalar bo'limi.

    Avtomatik xulosa va malumot jadvali.
    """

    def build(
        self,
        df: pd.DataFrame,
        reference_ranges: Dict = None,
        **kwargs
    ) -> List[Any]:
        """
        Xulosa bo'limini yaratish.

        Args:
            df: Ma'lumotlar DataFrame
            reference_ranges: Malumot oralig'i lug'ati

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        # Section header
        header = self.get_label('conclusions_header')
        elements.append(self.text_builder.section_header(header))

        # Decorative line
        elements.append(self.text_builder.accent_line())

        if len(df) == 0:
            no_cells = self.get_message('no_cells_detected', "Hujayralar aniqlanmadi.")
            elements.append(self.text_builder.body(no_cells))
            return elements

        # Generate conclusions based on data
        conclusions = self._generate_conclusions(df)

        # Add numbered conclusions
        for i, conclusion in enumerate(conclusions, 1):
            elements.append(self.text_builder.conclusion(f"{i}. {conclusion}"))
            elements.append(self.text_builder.spacer(8))

        # Add interpretation guide
        elements.append(self.text_builder.spacer(15))
        guide_header = self.get_label('interpretation_guide')
        elements.append(self.text_builder.subsection_header(guide_header))

        # Reference ranges table
        elements.extend(self._build_reference_table(reference_ranges))

        return elements

    def _get_conclusion_template(self, key: str, default: str) -> str:
        """Get conclusion template text."""
        return self.get_label(f'conclusions.{key}', default)

    def _generate_conclusions(self, df: pd.DataFrame) -> List[str]:
        """Xulosa matnlarini yaratish."""
        conclusions = []

        # Cell count check
        cell_count = len(df)
        if cell_count < 10:
            conclusions.append(
                self._get_conclusion_template('low_cell_count',
                    "Kam hujayra soni - natijalar statistik jihatdan cheklangan bo'lishi mumkin.")
            )

        # Shape analysis
        if 'circularity' in df.columns:
            mean_circ = df['circularity'].mean()
            std_circ = df['circularity'].std()

            if mean_circ >= 0.7 and std_circ < 0.15:
                conclusions.append(
                    self._get_conclusion_template('normal',
                        "Tahlil qilingan hujayralar normal morfologik xususiyatlarni ko'rsatmoqda.")
                )
            elif mean_circ < 0.6:
                conclusions.append(
                    self._get_conclusion_template('abnormal_shape',
                        "Ba'zi hujayralarda shakl anomaliyalari kuzatildi. Qo'shimcha tekshiruv tavsiya etiladi.")
                )

            # High variability check
            if std_circ > 0.2:
                conclusions.append(
                    self._get_conclusion_template('high_variability',
                        "Hujayra o'lchamlarida yuqori o'zgaruvchanlik mavjud.")
                )

        # If no issues, add normal conclusion
        if not conclusions:
            conclusions.append(
                self._get_conclusion_template('normal',
                    "Tahlil qilingan hujayralar normal morfologik xususiyatlarni ko'rsatmoqda.")
            )

        return conclusions

    def _build_reference_table(self, reference_ranges: Dict = None) -> List[Any]:
        """Malumot jadvalini yaratish."""
        elements = []

        if reference_ranges is None:
            reference_ranges = self.labels.get('reference_ranges', {})

        if not reference_ranges:
            return elements

        # Headers using helper methods
        headers = [
            self.get_stat_label('metric'),
            self.get_label('reference_ranges_header.normal_range', "Normal oraliq"),
            self.get_label('reference_ranges_header.description', "Izoh")
        ]

        # Data rows
        data = []
        for metric, info in reference_ranges.items():
            if isinstance(info, dict) and 'normal' in info:
                metric_name = self.get_metric_label(metric)
                range_str = f"{info['normal'][0]:.1f} - {info['normal'][1]:.1f}"
                description = info.get('description', '')
                data.append([metric_name, range_str, description])

        if data:
            table = self.table_builder.build_reference_table(headers, data)
            elements.append(table)

        return elements
