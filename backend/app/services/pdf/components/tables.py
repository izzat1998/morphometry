"""
Table Builder Component
=======================
Jadval yaratish uchun komponent.
"""

from typing import List, Any, Optional
import pandas as pd
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

from ..themes.base import BaseTheme


class TableBuilder:
    """
    Jadval yaratuvchi.

    Turli xil jadval turlarini yaratish uchun ishlatiladi.
    """

    def __init__(self, theme: BaseTheme):
        self.theme = theme

    def build_data_table(
        self,
        data: List[List[Any]],
        col_widths: Optional[List[float]] = None,
        header_row: bool = True
    ) -> Table:
        """
        Ma'lumotlar jadvalini yaratish.

        Args:
            data: Jadval ma'lumotlari (ro'yxatlar ro'yxati)
            col_widths: Ustun kengliklari
            header_row: Birinchi qator sarlavha ekanligini ko'rsatadi

        Returns:
            Table: ReportLab jadval obyekti
        """
        table = Table(data, colWidths=col_widths)

        if header_row:
            style = self.theme.get_table_style('header')
        else:
            style = self.theme.get_table_style('simple')

        table.setStyle(style)
        return table

    def build_key_value_table(
        self,
        data: List[tuple],
        key_width: float = 120,
        value_width: float = 300
    ) -> Table:
        """
        Kalit-qiymat jadvalini yaratish.

        Args:
            data: (kalit, qiymat) juftliklari ro'yxati
            key_width: Kalit ustuni kengligi
            value_width: Qiymat ustuni kengligi

        Returns:
            Table: ReportLab jadval obyekti
        """
        table_data = [[f"{k}:", v] for k, v in data]
        table = Table(table_data, colWidths=[key_width, value_width])

        style = TableStyle([
            ('FONTNAME', (0, 0), (0, -1), self.theme.fonts.bold),
            ('FONTNAME', (1, 0), (1, -1), self.theme.fonts.regular),
            ('FONTSIZE', (0, 0), (-1, -1), self.theme.fonts.body_size),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.theme.colors.text),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ])

        table.setStyle(style)
        return table

    def build_statistics_table(
        self,
        headers: List[str],
        data: List[List[Any]],
        col_widths: Optional[List[float]] = None
    ) -> Table:
        """
        Statistika jadvalini yaratish.

        Args:
            headers: Sarlavha qatori
            data: Ma'lumotlar qatorlari
            col_widths: Ustun kengliklari

        Returns:
            Table: ReportLab jadval obyekti
        """
        table_data = [headers] + data

        if col_widths is None:
            col_widths = [120] + [60] * (len(headers) - 1)

        table = Table(table_data, colWidths=col_widths)

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.theme.colors.primary),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), self.theme.fonts.bold),
            ('FONTNAME', (0, 1), (-1, -1), self.theme.fonts.regular),
            ('FONTSIZE', (0, 0), (-1, 0), self.theme.fonts.table_header_size),
            ('FONTSIZE', (0, 1), (-1, -1), self.theme.fonts.table_body_size),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, self.theme.colors.border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.theme.colors.background_alt])
        ])

        table.setStyle(style)
        return table

    def build_quality_table(
        self,
        headers: List[str],
        data: List[List[Any]],
        col_widths: Optional[List[float]] = None
    ) -> Table:
        """
        Sifat baholash jadvalini yaratish.

        Args:
            headers: Sarlavha qatori
            data: Ma'lumotlar qatorlari
            col_widths: Ustun kengliklari

        Returns:
            Table: ReportLab jadval obyekti
        """
        table_data = [headers] + data

        if col_widths is None:
            col_widths = [150, 80, 80]

        table = Table(table_data, colWidths=col_widths)

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.theme.colors.primary),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), self.theme.fonts.bold),
            ('FONTNAME', (0, 1), (-1, -1), self.theme.fonts.regular),
            ('FONTSIZE', (0, 0), (-1, -1), self.theme.fonts.body_size),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, self.theme.colors.border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.theme.colors.background_alt])
        ])

        table.setStyle(style)
        return table

    def build_reference_table(
        self,
        headers: List[str],
        data: List[List[Any]],
        col_widths: Optional[List[float]] = None
    ) -> Table:
        """
        Malumot (reference) jadvalini yaratish.

        Args:
            headers: Sarlavha qatori
            data: Ma'lumotlar qatorlari
            col_widths: Ustun kengliklari

        Returns:
            Table: ReportLab jadval obyekti
        """
        table_data = [headers] + data

        if col_widths is None:
            col_widths = [100, 80, 250]

        table = Table(table_data, colWidths=col_widths)

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.theme.colors.secondary),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), self.theme.fonts.bold),
            ('FONTNAME', (0, 1), (-1, -1), self.theme.fonts.regular),
            ('FONTSIZE', (0, 0), (-1, -1), self.theme.fonts.small_size),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, self.theme.colors.border),
        ])

        table.setStyle(style)
        return table

    def dataframe_to_table(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        headers: Optional[List[str]] = None,
        max_rows: int = 50,
        col_width: float = 65
    ) -> tuple:
        """
        DataFrame'ni jadvalga aylantirish.

        Args:
            df: Pandas DataFrame
            columns: Foydalaniladigan ustunlar
            headers: O'zbekcha sarlavhalar
            max_rows: Maksimal qatorlar soni
            col_width: Ustun kengligi

        Returns:
            tuple: (Table, qolgan_qatorlar_soni)
        """
        if columns is None:
            columns = list(df.columns)[:6]

        if headers is None:
            headers = columns

        display_df = df[columns].head(max_rows)

        table_data = [headers]
        for _, row in display_df.iterrows():
            row_data = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    row_data.append(f"{val:.3f}")
                else:
                    row_data.append(str(val))
            table_data.append(row_data)

        remaining = len(df) - max_rows if len(df) > max_rows else 0

        col_widths = [col_width] * len(columns)
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(self.theme.get_table_style('header'))

        return table, remaining
