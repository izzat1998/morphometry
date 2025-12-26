"""
Base Theme
==========
Asosiy mavzu klassi - barcha mavzular uchun asos.
"""

from dataclasses import dataclass, field
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColorPalette:
    """Rang palitrasi."""
    primary: Color = HexColor('#1a5f7a')
    secondary: Color = HexColor('#159895')
    accent: Color = HexColor('#57c5b6')
    success: Color = HexColor('#2d6a4f')
    warning: Color = HexColor('#e76f51')
    error: Color = HexColor('#dc3545')
    text: Color = HexColor('#212529')
    text_light: Color = HexColor('#6c757d')
    background: Color = HexColor('#ffffff')
    background_alt: Color = HexColor('#f8f9fa')
    border: Color = HexColor('#dee2e6')


@dataclass
class FontConfig:
    """Shrift sozlamalari."""
    regular: str = 'DejaVuSans'
    bold: str = 'DejaVuSans-Bold'
    regular_path: str = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold_path: str = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

    # Font sizes
    title_size: int = 22
    subtitle_size: int = 10
    section_header_size: int = 13
    subsection_header_size: int = 11
    body_size: int = 9
    small_size: int = 8
    table_header_size: int = 9
    table_body_size: int = 8
    footer_size: int = 8


class BaseTheme:
    """
    Asosiy mavzu klassi.

    Barcha mavzular uchun umumiy funksiyalar va uslublar.
    """

    def __init__(self):
        self.colors = ColorPalette()
        self.fonts = FontConfig()
        self._register_fonts()
        self.styles = self._create_styles()

    def _register_fonts(self):
        """Shriftlarni ro'yxatdan o'tkazish."""
        try:
            pdfmetrics.registerFont(TTFont(self.fonts.regular, self.fonts.regular_path))
            pdfmetrics.registerFont(TTFont(self.fonts.bold, self.fonts.bold_path))
            logger.info(f"Shriftlar ro'yxatdan o'tkazildi: {self.fonts.regular}")
        except Exception as e:
            logger.warning(f"Shriftlarni ro'yxatdan o'tkazib bo'lmadi: {e}")
            self.fonts.regular = 'Helvetica'
            self.fonts.bold = 'Helvetica-Bold'

    def _create_styles(self) -> dict:
        """Uslublarni yaratish."""
        base_styles = getSampleStyleSheet()

        # Update base styles with our font
        for style_name in ['Normal', 'BodyText', 'Heading1', 'Heading2', 'Heading3']:
            if style_name in base_styles:
                base_styles[style_name].fontName = self.fonts.regular

        custom_styles = {}

        # Title style
        custom_styles['Title'] = ParagraphStyle(
            name='Title',
            parent=base_styles['Heading1'],
            fontName=self.fonts.bold,
            fontSize=self.fonts.title_size,
            spaceAfter=8,
            alignment=TA_CENTER,
            textColor=self.colors.primary
        )

        # Subtitle style
        custom_styles['Subtitle'] = ParagraphStyle(
            name='Subtitle',
            parent=base_styles['Normal'],
            fontName=self.fonts.regular,
            fontSize=self.fonts.subtitle_size,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=self.colors.text_light
        )

        # Section header style
        custom_styles['SectionHeader'] = ParagraphStyle(
            name='SectionHeader',
            parent=base_styles['Heading2'],
            fontName=self.fonts.bold,
            fontSize=self.fonts.section_header_size,
            spaceBefore=16,
            spaceAfter=8,
            textColor=self.colors.primary
        )

        # Subsection header style
        custom_styles['SubsectionHeader'] = ParagraphStyle(
            name='SubsectionHeader',
            parent=base_styles['Heading3'],
            fontName=self.fonts.bold,
            fontSize=self.fonts.subsection_header_size,
            spaceBefore=12,
            spaceAfter=6,
            textColor=self.colors.secondary
        )

        # Body text style
        custom_styles['Body'] = ParagraphStyle(
            name='Body',
            parent=base_styles['BodyText'],
            fontName=self.fonts.regular,
            fontSize=self.fonts.body_size,
            spaceAfter=6,
            textColor=self.colors.text
        )

        # Footer style
        custom_styles['Footer'] = ParagraphStyle(
            name='Footer',
            parent=base_styles['Normal'],
            fontName=self.fonts.regular,
            fontSize=self.fonts.footer_size,
            textColor=self.colors.text_light
        )

        # Conclusion box style
        custom_styles['Conclusion'] = ParagraphStyle(
            name='Conclusion',
            parent=base_styles['BodyText'],
            fontName=self.fonts.regular,
            fontSize=self.fonts.body_size + 1,
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=10,
            borderColor=self.colors.accent,
            borderWidth=2,
            borderPadding=8,
            backColor=self.colors.background_alt
        )

        # Quality good style
        custom_styles['QualityGood'] = ParagraphStyle(
            name='QualityGood',
            parent=base_styles['Normal'],
            fontName=self.fonts.bold,
            fontSize=self.fonts.body_size,
            textColor=self.colors.success
        )

        # Quality warning style
        custom_styles['QualityWarning'] = ParagraphStyle(
            name='QualityWarning',
            parent=base_styles['Normal'],
            fontName=self.fonts.bold,
            fontSize=self.fonts.body_size,
            textColor=self.colors.warning
        )

        return custom_styles

    def get_style(self, name: str) -> ParagraphStyle:
        """Uslubni olish."""
        return self.styles.get(name, self.styles['Body'])

    def get_table_style(self, style_type: str = 'default') -> list:
        """Jadval uslubini olish."""
        from reportlab.platypus import TableStyle

        if style_type == 'header':
            return TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.primary),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), self.fonts.bold),
                ('FONTNAME', (0, 1), (-1, -1), self.fonts.regular),
                ('FONTSIZE', (0, 0), (-1, 0), self.fonts.table_header_size),
                ('FONTSIZE', (0, 1), (-1, -1), self.fonts.table_body_size),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, self.colors.border),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors.background_alt])
            ])
        elif style_type == 'simple':
            return TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), self.fonts.regular),
                ('FONTSIZE', (0, 0), (-1, -1), self.fonts.body_size),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ])
        else:
            return TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), self.fonts.regular),
                ('FONTSIZE', (0, 0), (-1, -1), self.fonts.body_size),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ])
