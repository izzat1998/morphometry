"""
Report Generation Service
=========================
Generate publication-quality reports from morphometry analysis.

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

import io
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import logging

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """Metadata for the analysis report."""
    title: str = "Morphometry Analysis Report"
    author: str = "Morphometry Analysis System"
    sample_id: str = ""
    image_filename: str = ""
    analysis_date: str = ""
    preprocessing_steps: list[str] = None
    segmentation_model: str = ""
    pixel_size_um: float = 1.0
    notes: str = ""

    def __post_init__(self):
        if not self.analysis_date:
            self.analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.preprocessing_steps is None:
            self.preprocessing_steps = []


class ReportGenerator:
    """
    Generate comprehensive analysis reports.

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
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Configure custom text styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10
        ))
        # BodyText already exists in default stylesheet, customize it instead
        self.styles['BodyText'].fontSize = 10
        self.styles['BodyText'].spaceAfter = 6

    # =========================================================================
    # PDF GENERATION
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
        Generate a PDF report.

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
        output_path = Path(output_path)
        metadata = metadata or ReportMetadata()

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )

        elements = []

        # Title
        elements.append(Paragraph(metadata.title, self.styles['ReportTitle']))
        elements.append(Spacer(1, 12))

        # Metadata section
        elements.extend(self._build_metadata_section(metadata, len(df)))
        elements.append(Spacer(1, 20))

        # Summary statistics
        elements.extend(self._build_summary_section(df))
        elements.append(Spacer(1, 20))

        # Visualization section
        if original_image is not None and masks is not None:
            elements.extend(self._build_image_section(original_image, masks))
            elements.append(Spacer(1, 20))

        # Histograms
        if include_histograms and len(df) > 0:
            elements.append(PageBreak())
            elements.extend(self._build_histogram_section(df))

        # Individual cell data table
        if include_cell_table and len(df) > 0:
            elements.append(PageBreak())
            elements.extend(self._build_cell_table_section(df, max_cells_in_table))

        # Build PDF
        doc.build(elements)
        logger.info(f"PDF report generated: {output_path}")

        return output_path

    def _build_metadata_section(
        self,
        metadata: ReportMetadata,
        cell_count: int
    ) -> list:
        """Build metadata information section."""
        elements = []
        elements.append(Paragraph("Analysis Information", self.styles['SectionHeader']))

        info_data = [
            ["Sample ID:", metadata.sample_id or "N/A"],
            ["Image File:", metadata.image_filename or "N/A"],
            ["Analysis Date:", metadata.analysis_date],
            ["Total Cells:", str(cell_count)],
            ["Pixel Size:", f"{metadata.pixel_size_um} µm"],
            ["Segmentation Model:", metadata.segmentation_model or "N/A"],
        ]

        if metadata.preprocessing_steps:
            info_data.append([
                "Preprocessing:",
                ", ".join(metadata.preprocessing_steps)
            ])

        table = Table(info_data, colWidths=[100, 300])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))

        elements.append(table)
        return elements

    def _build_summary_section(self, df: pd.DataFrame) -> list:
        """Build summary statistics section."""
        elements = []
        elements.append(Paragraph("Summary Statistics", self.styles['SectionHeader']))

        if len(df) == 0:
            elements.append(Paragraph("No cells detected.", self.styles['BodyText']))
            return elements

        # Key metrics to summarize
        metrics = {
            'area': 'Area (px²)',
            'perimeter': 'Perimeter (px)',
            'circularity': 'Circularity',
            'eccentricity': 'Eccentricity',
            'solidity': 'Solidity',
            'aspect_ratio': 'Aspect Ratio'
        }

        # Build summary table
        summary_data = [['Metric', 'Mean', 'Std', 'Min', 'Max', 'Median']]

        for col, label in metrics.items():
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    summary_data.append([
                        label,
                        f"{values.mean():.2f}",
                        f"{values.std():.2f}",
                        f"{values.min():.2f}",
                        f"{values.max():.2f}",
                        f"{values.median():.2f}"
                    ])

        table = Table(summary_data, colWidths=[120, 60, 60, 60, 60, 60])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)])
        ]))

        elements.append(table)
        return elements

    def _build_image_section(
        self,
        original_image: np.ndarray,
        masks: np.ndarray
    ) -> list:
        """Build image visualization section."""
        elements = []
        elements.append(Paragraph("Segmentation Results", self.styles['SectionHeader']))

        # Create overlay visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        if original_image.ndim == 3:
            axes[0].imshow(original_image)
        else:
            axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Segmentation overlay
        if original_image.ndim == 3:
            axes[1].imshow(original_image)
        else:
            axes[1].imshow(original_image, cmap='gray')

        # Overlay colored masks
        from matplotlib.colors import ListedColormap
        cmap = plt.cm.Set3
        masked = np.ma.masked_where(masks == 0, masks)
        axes[1].imshow(masked, cmap=cmap, alpha=0.5)
        axes[1].set_title(f'Segmentation ({masks.max()} cells)')
        axes[1].axis('off')

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Add to PDF
        img = RLImage(buf, width=450, height=225)
        elements.append(img)

        return elements

    def _build_histogram_section(self, df: pd.DataFrame) -> list:
        """Build histogram visualizations."""
        elements = []
        elements.append(Paragraph("Distribution Analysis", self.styles['SectionHeader']))

        # Metrics to plot
        metrics = ['area', 'circularity', 'eccentricity', 'solidity']
        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            return elements

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics[:4]):
            values = df[metric].dropna()
            axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(metric.replace('_', ' ').title())
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Distribution')

            # Add statistics annotation
            stats_text = f'n={len(values)}\nµ={values.mean():.2f}\nσ={values.std():.2f}'
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

        img = RLImage(buf, width=450, height=360)
        elements.append(img)

        return elements

    def _build_cell_table_section(
        self,
        df: pd.DataFrame,
        max_rows: int = 50
    ) -> list:
        """Build individual cell data table."""
        elements = []
        elements.append(Paragraph("Individual Cell Measurements", self.styles['SectionHeader']))

        # Select columns for table
        columns = ['cell_id', 'area', 'perimeter', 'circularity', 'eccentricity', 'solidity']
        available_cols = [c for c in columns if c in df.columns]

        if not available_cols:
            return elements

        # Limit rows
        display_df = df[available_cols].head(max_rows)

        # Build table data
        table_data = [available_cols]  # Header
        for _, row in display_df.iterrows():
            table_data.append([
                f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col])
                for col in available_cols
            ])

        if len(df) > max_rows:
            table_data.append([f"... ({len(df) - max_rows} more rows)"] + [''] * (len(available_cols) - 1))

        col_width = 400 / len(available_cols)
        table = Table(table_data, colWidths=[col_width] * len(available_cols))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        elements.append(table)
        return elements

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
            # Sheet 1: All measurements
            df.to_excel(writer, sheet_name='Cell Measurements', index=False)

            # Sheet 2: Summary statistics
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats).T
                summary_df.to_excel(writer, sheet_name='Summary Statistics')
            else:
                # Calculate basic summary
                summary_df = df.describe()
                summary_df.to_excel(writer, sheet_name='Summary Statistics')

            # Sheet 3: Metadata
            meta_df = pd.DataFrame([{
                'Title': metadata.title,
                'Author': metadata.author,
                'Sample ID': metadata.sample_id,
                'Image File': metadata.image_filename,
                'Analysis Date': metadata.analysis_date,
                'Segmentation Model': metadata.segmentation_model,
                'Pixel Size (µm)': metadata.pixel_size_um,
                'Total Cells': len(df),
                'Notes': metadata.notes
            }]).T
            meta_df.columns = ['Value']
            meta_df.to_excel(writer, sheet_name='Metadata')

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
