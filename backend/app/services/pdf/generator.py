"""
PDF Generator
=============
Asosiy PDF generatori - fasad sinfi.

Bu modul barcha PDF yaratish funksiyalarini birlashtiradi
va tashqi foydalanuvchilar uchun oddiy API taqdim etadi.
"""

from pathlib import Path
from typing import Optional, Literal
import pandas as pd
import numpy as np

from .templates.morphometry_report import (
    MorphometryReportGenerator,
    ReportMetadata
)

import logging

logger = logging.getLogger(__name__)


class PDFGenerator:
    """
    PDF yaratish uchun asosiy klass.

    Bu klass turli xil hisobot shablonlarini boshqaradi
    va ularni yaratish uchun yagona nuqta vazifasini bajaradi.

    Foydalanish:
        generator = PDFGenerator()
        generator.generate_morphometry_report(df, output_path, metadata)
    """

    def __init__(self):
        """Generatorni yaratish."""
        self._morphometry = None

    @property
    def morphometry(self) -> MorphometryReportGenerator:
        """Morfometriya hisoboti generatori (lazy initialization)."""
        if self._morphometry is None:
            self._morphometry = MorphometryReportGenerator()
        return self._morphometry

    def generate_morphometry_report(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        metadata: Optional[ReportMetadata] = None,
        original_image: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        include_histograms: bool = True,
        include_cell_table: bool = True,
        max_cells_in_table: int = 50
    ) -> Path:
        """
        Morfometriya hisobotini yaratish.

        Args:
            df: Ma'lumotlar DataFrame
            output_path: Chiqish fayli yo'li
            metadata: Hisobot ma'lumotlari
            original_image: Asl rasm
            masks: Segmentatsiya maskalari
            include_histograms: Gistogrammalarni qo'shish
            include_cell_table: Hujayra jadvalini qo'shish
            max_cells_in_table: Jadvaldagi maksimal hujayralar

        Returns:
            Yaratilgan fayl yo'li
        """
        return self.morphometry.generate(
            df=df,
            output_path=output_path,
            metadata=metadata,
            original_image=original_image,
            masks=masks,
            include_histograms=include_histograms,
            include_cell_table=include_cell_table,
            max_cells_in_table=max_cells_in_table
        )


# Qulaylik funksiyasi
def generate_pdf(
    df: pd.DataFrame,
    output_path: str | Path,
    template: Literal["morphometry"] = "morphometry",
    **kwargs
) -> Path:
    """
    PDF hisobotini yaratish.

    Args:
        df: Ma'lumotlar DataFrame
        output_path: Chiqish fayli yo'li
        template: Shablon nomi
        **kwargs: Qo'shimcha parametrlar

    Returns:
        Yaratilgan fayl yo'li
    """
    generator = PDFGenerator()

    if template == "morphometry":
        return generator.generate_morphometry_report(df, output_path, **kwargs)
    else:
        raise ValueError(f"Noma'lum shablon: {template}")
