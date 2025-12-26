"""
Header/Footer Builder Component
===============================
Sahifa sarlavhasi va pastki qismi yaratish uchun komponent.
"""

from reportlab.lib import colors
from reportlab.lib.units import mm

from ..themes.base import BaseTheme


class HeaderFooterBuilder:
    """
    Sahifa sarlavhasi va pastki qismi yaratuvchi.

    Har bir sahifada ko'rinadigan elementlar.
    """

    def __init__(
        self,
        theme: BaseTheme,
        page_size: tuple,
        report_id: str = "",
        analysis_date: str = "",
        labels: dict = None
    ):
        self.theme = theme
        self.page_size = page_size
        self.report_id = report_id
        self.analysis_date = analysis_date
        self.labels = labels or {}

    def draw(self, canvas, doc):
        """
        Sarlavha va pastki qismni chizish.

        Args:
            canvas: ReportLab canvas
            doc: ReportLab document
        """
        canvas.saveState()

        self._draw_header(canvas, doc)
        self._draw_footer(canvas, doc)

        canvas.restoreState()

    def _draw_header(self, canvas, doc):
        """Sarlavhani chizish."""
        page_width = self.page_size[0]
        page_height = self.page_size[1]

        # Header line
        canvas.setStrokeColor(self.theme.colors.primary)
        canvas.setLineWidth(2)
        canvas.line(20*mm, page_height - 15*mm, page_width - 20*mm, page_height - 15*mm)

        # Report ID (left)
        canvas.setFont(self.theme.fonts.regular, 8)
        canvas.setFillColor(colors.grey)
        report_label = self.labels.get('report_id', 'Hisobot raqami')
        canvas.drawString(20*mm, page_height - 12*mm, f"{report_label}: {self.report_id}")

        # Date (right)
        canvas.drawRightString(page_width - 20*mm, page_height - 12*mm, self.analysis_date)

    def _draw_footer(self, canvas, doc):
        """Pastki qismni chizish."""
        page_width = self.page_size[0]

        # Footer line
        canvas.setStrokeColor(self.theme.colors.border)
        canvas.setLineWidth(0.5)
        canvas.line(20*mm, 15*mm, page_width - 20*mm, 15*mm)

        # Page number (center)
        page_label = self.labels.get('page', 'Sahifa')
        page_text = f"{page_label} {doc.page}"
        canvas.setFont(self.theme.fonts.regular, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(page_width / 2, 10*mm, page_text)

        # System name (left)
        generated_by = self.labels.get('generated_by', 'Tizim tomonidan yaratildi')
        canvas.drawString(20*mm, 10*mm, generated_by)

        # Confidential notice (right)
        confidential = self.labels.get('confidential', 'MAXFIY')
        canvas.setFont(self.theme.fonts.regular, 7)
        canvas.setFillColor(self.theme.colors.warning)
        canvas.drawRightString(page_width - 20*mm, 10*mm, confidential)

    def update_info(self, report_id: str = None, analysis_date: str = None):
        """Ma'lumotlarni yangilash."""
        if report_id is not None:
            self.report_id = report_id
        if analysis_date is not None:
            self.analysis_date = analysis_date
