"""
Morphometry Report Template
===========================
Morfometriya tahlili hisoboti shabloni.

Barcha matnlar O'zbek tilida.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, PageBreak

from ..themes.medical import MedicalTheme
from ..components.header_footer import HeaderFooterBuilder
from ..sections import (
    TitleSection,
    MetadataSection,
    SummarySection,
    QualitySection,
    VisualizationSection,
    DistributionSection,
    ConclusionsSection,
    DataTableSection
)

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# O'ZBEK TILIDAGI TEGLAR
# =============================================================================

UZBEK_LABELS = {
    # Sarlavhalar
    'report_title': "Morfometriya Tahlili Hisoboti",
    'analysis_info': "Tahlil Ma'lumotlari",
    'summary_statistics': "Umumiy Statistika",
    'segmentation_results': "Segmentatsiya Natijalari",
    'distribution_analysis': "Taqsimot Tahlili",
    'individual_measurements': "Alohida Hujayra O'lchovlari",
    'conclusions_header': "Xulosa va Tavsiyalar",
    'quality_assessment': "Sifat Baholash",
    'interpretation_guide': "Natijalarni Talqin Qilish",

    # Ma'lumotlar teglari
    'sample_id': "Namuna raqami",
    'image_file': "Rasm fayli",
    'analysis_date': "Tahlil sanasi",
    'total_cells': "Jami hujayralar",
    'pixel_size': "Piksel o'lchami",
    'segmentation_model': "Segmentatsiya modeli",
    'preprocessing': "Oldindan ishlov berish",
    'not_available': "Mavjud emas",

    # Ko'rsatkichlar (metrics)
    'metrics': {
        'area': "Maydon (px²)",
        'perimeter': "Perimetr (px)",
        'circularity': "Aylanasimonlik",
        'eccentricity': "Ekssentriklik",
        'solidity': "Zichlik",
        'aspect_ratio': "Tomonlar nisbati",
        'intensity_mean': "O'rtacha intensivlik",
    },

    # Qisqa ko'rsatkich nomlari (jadval uchun)
    'metrics_short': {
        'cell_id': "Hujayra raqami",
        'area': "Maydon",
        'perimeter': "Perimetr",
        'circularity': "Aylanasimonlik",
        'eccentricity': "Ekssentriklik",
        'solidity': "Zichlik",
        'aspect_ratio': "Nisbat",
    },

    # Statistik teglar
    'stats': {
        'metric': "Ko'rsatkich",
        'mean': "O'rtacha",
        'std': "St. og'ish",
        'min': "Min",
        'max': "Maks",
        'median': "Mediana",
        'count': "Soni",
    },

    # Grafik teglari
    'charts': {
        'original_image': "Asl rasm",
        'segmentation': "Segmentatsiya",
        'cells_detected': "aniqlangan hujayralar",
        'distribution': "taqsimoti",
        'count': "Soni",
    },

    # Sifat baholash
    'quality': {
        'excellent': "A'lo",
        'good': "Yaxshi",
        'acceptable': "Qoniqarli",
        'poor': "Yomon",
        'cell_count': "Hujayra soni",
        'segmentation_quality': "Segmentatsiya sifati",
        'image_quality': "Rasm sifati",
        'value': "Qiymat",
        'rating': "Baho",
    },

    # Xulosa shablonlari
    'conclusions': {
        'normal': "Tahlil qilingan hujayralar normal morfologik xususiyatlarni ko'rsatmoqda.",
        'abnormal_shape': "Ba'zi hujayralarda shakl anomaliyalari kuzatildi. Qo'shimcha tekshiruv tavsiya etiladi.",
        'high_variability': "Hujayra o'lchamlarida yuqori o'zgaruvchanlik mavjud.",
        'low_cell_count': "Kam hujayra soni - natijalar statistik jihatdan cheklangan bo'lishi mumkin.",
    },

    # Malumot oralig'i
    'reference_ranges': {
        'circularity': {
            'normal': (0.7, 1.0),
            'description': "Normal hujayralar uchun aylanasimonlik 0.7-1.0 oralig'ida bo'ladi",
        },
        'solidity': {
            'normal': (0.9, 1.0),
            'description': "Sog'lom hujayralar yuqori zichlikka ega (>0.9)",
        },
        'aspect_ratio': {
            'normal': (1.0, 2.0),
            'description': "Normal hujayralar nisbatan simmetrik (1.0-2.0)",
        },
    },

    # Malumot jadvali sarlavhalari
    'reference_ranges_header': {
        'normal_range': "Normal oraliq",
        'description': "Izoh",
    },

    # PDF elementlari
    'pdf': {
        'page': "Sahifa",
        'report_id': "Hisobot raqami",
        'generated_by': "Tizim tomonidan yaratildi",
        'confidential': "MAXFIY - Faqat tibbiy xodimlar uchun",
    },

    # Xabarlar
    'messages': {
        'no_cells_detected': "Hujayralar aniqlanmadi.",
        'more_rows': "... (yana {n} qator)",
    },

    # Oldindan ishlov berish bosqichlari
    'preprocessing_steps': {
        'denoise': "Shovqinni yo'qotish",
        'clahe': "Kontrastni yaxshilash",
        'normalize': "Normalizatsiya",
        'background_subtraction': "Fon olib tashlash",
        'gamma_correction': "Gamma tuzatish",
        'sharpen': "Keskinlashtirish",
        'smooth': "Silliqlashtirish",
    },
}


@dataclass
class ReportMetadata:
    """Hisobot ma'lumotlari."""
    title: str = UZBEK_LABELS['report_title']
    sample_id: str = ""
    image_filename: str = ""
    analysis_date: str = ""
    preprocessing_steps: List[str] = field(default_factory=list)
    segmentation_model: str = ""
    pixel_size_um: float = 1.0
    notes: str = ""

    def __post_init__(self):
        if not self.analysis_date:
            self.analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MorphometryReportGenerator:
    """
    Morfometriya hisoboti generatori.

    Professional tibbiy hisobot yaratadi.
    Barcha matnlar O'zbek tilida.

    Foydalanish:
        generator = MorphometryReportGenerator()
        generator.generate(df, "hisobot.pdf", metadata)
    """

    def __init__(self, page_size: tuple = A4):
        """
        Generatorni yaratish.

        Args:
            page_size: Sahifa o'lchami
        """
        self.page_size = page_size
        self.theme = MedicalTheme()
        self.labels = UZBEK_LABELS

        # Initialize sections
        self._init_sections()

    def _init_sections(self):
        """Bo'limlarni yaratish."""
        self.title_section = TitleSection(self.theme, self.labels)
        self.metadata_section = MetadataSection(self.theme, self.labels)
        self.summary_section = SummarySection(self.theme, self.labels)
        self.quality_section = QualitySection(self.theme, self.labels)
        self.visualization_section = VisualizationSection(self.theme, self.labels)
        self.distribution_section = DistributionSection(self.theme, self.labels)
        self.conclusions_section = ConclusionsSection(self.theme, self.labels)
        self.data_table_section = DataTableSection(self.theme, self.labels)

    def generate(
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
        Hisobotni yaratish.

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
        output_path = Path(output_path)
        metadata = metadata or ReportMetadata()

        # Translate preprocessing steps
        translated_steps = []
        for step in metadata.preprocessing_steps:
            translated = self.labels['preprocessing_steps'].get(step, step)
            translated_steps.append(translated)

        # Create header/footer builder
        header_footer = HeaderFooterBuilder(
            theme=self.theme,
            page_size=self.page_size,
            report_id=metadata.sample_id or output_path.stem[:8].upper(),
            analysis_date=metadata.analysis_date,
            labels=self.labels['pdf']
        )

        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=25*mm,
            bottomMargin=22*mm
        )

        elements = []

        # =================================================================
        # SAHIFA 1: Sarlavha, Ma'lumotlar, Statistika, Sifat
        # =================================================================

        # Title section
        subtitle = f"{metadata.image_filename or 'Mikroskopiya tahlili'} | {metadata.analysis_date}"
        elements.extend(self.title_section.build(
            title=metadata.title,
            subtitle=subtitle
        ))

        # Metadata section
        elements.extend(self.metadata_section.build(
            sample_id=metadata.sample_id,
            image_filename=metadata.image_filename,
            analysis_date=metadata.analysis_date,
            total_cells=len(df),
            pixel_size=metadata.pixel_size_um,
            segmentation_model=metadata.segmentation_model,
            preprocessing_steps=translated_steps
        ))

        # Summary section
        elements.extend(self.summary_section.build(
            df=df,
            metrics=self.labels['metrics']
        ))

        # Quality section
        elements.extend(self.quality_section.build(df=df))

        # Visualization section
        elements.extend(self.visualization_section.build(
            original_image=original_image,
            masks=masks
        ))

        # =================================================================
        # SAHIFA 2: Taqsimot tahlili
        # =================================================================
        if include_histograms and len(df) > 0:
            elements.append(PageBreak())
            elements.extend(self.distribution_section.build(
                df=df,
                metric_labels=self.labels['metrics']
            ))

        # =================================================================
        # SAHIFA 3: Xulosa va tavsiyalar
        # =================================================================
        elements.append(PageBreak())
        elements.extend(self.conclusions_section.build(
            df=df,
            reference_ranges=self.labels['reference_ranges']
        ))

        # =================================================================
        # SAHIFA 4+: Alohida hujayra ma'lumotlari
        # =================================================================
        if include_cell_table and len(df) > 0:
            elements.append(PageBreak())
            elements.extend(self.data_table_section.build(
                df=df,
                column_labels=self.labels['metrics_short'],
                max_rows=max_cells_in_table
            ))

        # Build PDF with header/footer
        doc.build(
            elements,
            onFirstPage=header_footer.draw,
            onLaterPages=header_footer.draw
        )

        logger.info(f"Hisobot yaratildi: {output_path}")

        return output_path
