"""
Report Generation Service
=========================
Generate publication-quality reports from morphometry analysis.
O'zbekcha hisobotlar yaratish xizmati.

Supports multiple output formats:
- PDF: Complete report with images, tables, and statistics
- Excel: Detailed data tables for further analysis
- JSON: Machine-readable format for integration

Key Insight:
    Scientific reports should include:
    1. Methods (preprocessing, segmentation model used)
    2. Summary statistics (mean, std, median)
    3. Individual cell data
    4. Visualizations (histograms, overlays)
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, List

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4

# Import new modular PDF system
from app.services.pdf import MorphometryReportGenerator
from app.services.pdf.templates.morphometry_report import (
    ReportMetadata as PDFReportMetadata
)

# Import Uzbek translations for Excel/JSON
from app.core.translations import (
    REPORT_TITLE, AUTHOR,
    SAMPLE_ID, IMAGE_FILE, ANALYSIS_DATE, TOTAL_CELLS, PIXEL_SIZE,
    SEGMENTATION_MODEL, MESSAGES, EXCEL_SHEETS
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """Metadata for the analysis report. / Tahlil hisoboti uchun ma'lumotlar."""
    title: str = REPORT_TITLE  # "Morfometriya Tahlili Hisoboti"
    author: str = AUTHOR  # "Morfometriya Tahlil Tizimi"
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

    def to_pdf_metadata(self) -> PDFReportMetadata:
        """Convert to PDF module metadata format."""
        return PDFReportMetadata(
            title=self.title,
            sample_id=self.sample_id,
            image_filename=self.image_filename,
            analysis_date=self.analysis_date,
            preprocessing_steps=self.preprocessing_steps,
            segmentation_model=self.segmentation_model,
            pixel_size_um=self.pixel_size_um,
            notes=self.notes
        )


class ReportGenerator:
    """
    Generate comprehensive analysis reports.
    O'zbekcha hisobotlarni yaratish.

    Supports PDF, Excel, and JSON output formats with
    publication-quality visualizations.

    Example:
        >>> generator = ReportGenerator()
        >>> generator.generate_pdf(
        ...     df=morphometry_df,
        ...     output_path="report.pdf",
        ...     metadata=ReportMetadata(title="My Analysis")
        ... )
    """

    def __init__(self, page_size: tuple = A4):
        """
        Initialize report generator.

        Args:
            page_size: Page size for PDF reports (A4 or letter)
        """
        self.page_size = page_size
        self._pdf_generator = None

    @property
    def pdf_generator(self) -> MorphometryReportGenerator:
        """Lazy-initialized PDF generator."""
        if self._pdf_generator is None:
            self._pdf_generator = MorphometryReportGenerator(page_size=self.page_size)
        return self._pdf_generator

    # =========================================================================
    # PDF GENERATION (delegates to modular PDF system)
    # =========================================================================

    def generate_pdf(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        metadata: ReportMetadata | None = None,
        original_image: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        include_histograms: bool = True,
        include_cell_table: bool = True,
        max_cells_in_table: int = 50
    ) -> Path:
        """
        Generate a professional PDF report.

        Args:
            df: DataFrame with morphometry measurements
            output_path: Output file path
            metadata: Report metadata
            original_image: Original microscopy image
            masks: Segmentation masks
            include_histograms: Include distribution histograms
            include_cell_table: Include individual cell data table
            max_cells_in_table: Maximum cells to show in table

        Returns:
            Path to generated PDF
        """
        metadata = metadata or ReportMetadata()

        # Convert to PDF module metadata format
        pdf_metadata = metadata.to_pdf_metadata()

        # Delegate to modular PDF generator
        return self.pdf_generator.generate(
            df=df,
            output_path=output_path,
            metadata=pdf_metadata,
            original_image=original_image,
            masks=masks,
            include_histograms=include_histograms,
            include_cell_table=include_cell_table,
            max_cells_in_table=max_cells_in_table
        )

    # =========================================================================
    # EXCEL GENERATION
    # =========================================================================

    def generate_excel(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        metadata: ReportMetadata | None = None,
        summary_stats: dict | None = None
    ) -> Path:
        """
        Generate an Excel report with multiple sheets.

        Args:
            df: DataFrame with morphometry measurements
            output_path: Output file path
            metadata: Report metadata
            summary_stats: Pre-calculated summary statistics

        Returns:
            Path to generated Excel file
        """
        output_path = Path(output_path)
        metadata = metadata or ReportMetadata()

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: All measurements - Hujayra o'lchovlari
            df.to_excel(writer, sheet_name=EXCEL_SHEETS['measurements'], index=False)

            # Sheet 2: Summary statistics - Umumiy statistika
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats).T
                summary_df.to_excel(writer, sheet_name=EXCEL_SHEETS['summary'])
            else:
                # Calculate basic summary
                summary_df = df.describe()
                summary_df.to_excel(writer, sheet_name=EXCEL_SHEETS['summary'])

            # Sheet 3: Metadata - Ma'lumotlar
            meta_df = pd.DataFrame([{
                REPORT_TITLE.split()[0]: metadata.title,  # Sarlavha
                'Muallif': metadata.author,
                SAMPLE_ID: metadata.sample_id,
                IMAGE_FILE: metadata.image_filename,
                ANALYSIS_DATE: metadata.analysis_date,
                SEGMENTATION_MODEL: metadata.segmentation_model,
                f'{PIXEL_SIZE} (µm)': metadata.pixel_size_um,
                TOTAL_CELLS: len(df),
                'Izohlar': metadata.notes
            }]).T
            meta_df.columns = ['Qiymat']  # "Value" in Uzbek
            meta_df.to_excel(writer, sheet_name=EXCEL_SHEETS['metadata'])

        logger.info(f"Excel report generated: {output_path}")
        return output_path

    # =========================================================================
    # JSON GENERATION
    # =========================================================================

    def generate_json(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        metadata: ReportMetadata | None = None,
        summary_stats: dict | None = None
    ) -> Path:
        """
        Generate a JSON report for programmatic access.

        Args:
            df: DataFrame with morphometry measurements
            output_path: Output file path
            metadata: Report metadata
            summary_stats: Pre-calculated summary statistics

        Returns:
            Path to generated JSON file
        """
        output_path = Path(output_path)
        metadata = metadata or ReportMetadata()

        report_data = {
            'metadata': {
                'title': metadata.title,
                'author': metadata.author,
                'sample_id': metadata.sample_id,
                'image_filename': metadata.image_filename,
                'analysis_date': metadata.analysis_date,
                'segmentation_model': metadata.segmentation_model,
                'pixel_size_um': metadata.pixel_size_um,
                'preprocessing_steps': metadata.preprocessing_steps,
                'notes': metadata.notes
            },
            'summary': {
                'total_cells': len(df),
                'statistics': summary_stats or df.describe().to_dict()
            },
            'measurements': df.to_dict(orient='records')
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"JSON report generated: {output_path}")
        return output_path


# Convenience function
def generate_report(
    df: pd.DataFrame,
    output_path: str | Path,
    format: Literal["pdf", "excel", "json"] = "pdf",
    **kwargs
) -> Path:
    """
    Quick report generation function.

    Args:
        df: Morphometry measurements DataFrame
        output_path: Output file path
        format: Output format ("pdf", "excel", "json")
        **kwargs: Additional arguments for the specific generator

    Returns:
        Path to generated report
    """
    generator = ReportGenerator()

    if format == "pdf":
        return generator.generate_pdf(df, output_path, **kwargs)
    elif format == "excel":
        return generator.generate_excel(df, output_path, **kwargs)
    elif format == "json":
        return generator.generate_json(df, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")
