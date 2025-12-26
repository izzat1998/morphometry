"""
PDF Generation Module
=====================
Professional PDF report generation with modular architecture.

Tashkiliy tuzilma:
- themes/     : Ranglar, shriftlar, uslublar
- components/ : Qayta ishlatiladigan komponentlar (jadvallar, grafiklar)
- sections/   : Hisobot bo'limlari
- templates/  : Tayyor hisobot shablonlari

Usage:
    from app.services.pdf import MorphometryReportGenerator

    generator = MorphometryReportGenerator()
    generator.generate(df, output_path, metadata)
"""

from .generator import PDFGenerator
from .templates.morphometry_report import MorphometryReportGenerator

__all__ = ['PDFGenerator', 'MorphometryReportGenerator']
