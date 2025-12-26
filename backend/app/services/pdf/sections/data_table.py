"""
Data Table Section
==================
Alohida hujayra o'lchovlari bo'limi.
"""

from typing import List, Any, Dict
import pandas as pd

from .base import BaseSection


class DataTableSection(BaseSection):
    """
    Ma'lumotlar jadvali bo'limi.

    Alohida hujayralar o'lchovlari.
    """

    # Default columns to show in table
    DEFAULT_COLUMNS = ['cell_id', 'area', 'perimeter', 'circularity', 'eccentricity', 'solidity']

    def build(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        column_labels: Dict[str, str] = None,
        max_rows: int = 50,
        **kwargs
    ) -> List[Any]:
        """
        Ma'lumotlar jadvali bo'limini yaratish.

        Args:
            df: Ma'lumotlar DataFrame
            columns: Ko'rsatiladigan ustunlar
            column_labels: Ustun nomlari lug'ati
            max_rows: Maksimal qatorlar soni

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        if len(df) == 0:
            return elements

        # Section header
        header = self.get_label('individual_measurements')
        elements.append(self.text_builder.section_header(header))

        # Default columns
        if columns is None:
            columns = self.DEFAULT_COLUMNS

        # Filter to available columns
        available_cols = [c for c in columns if c in df.columns]

        if not available_cols:
            return elements

        # Build labels from metrics_short if not provided
        if column_labels is None:
            column_labels = {}
            metrics_short = self.labels.get('metrics_short', {})
            for col in available_cols:
                column_labels[col] = metrics_short.get(col, col)

        # Create headers
        headers = [column_labels.get(col, col) for col in available_cols]

        # Build table
        table, remaining = self.table_builder.dataframe_to_table(
            df,
            columns=available_cols,
            headers=headers,
            max_rows=max_rows
        )

        elements.append(table)

        # Add note about remaining rows
        if remaining > 0:
            more_template = self.get_message('more_rows', "... (yana {n} qator)")
            more_text = more_template.format(n=remaining)
            elements.append(self.text_builder.spacer(8))
            elements.append(self.text_builder.body(more_text))

        return elements
