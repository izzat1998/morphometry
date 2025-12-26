"""
Medical Theme
=============
Tibbiy hisobotlar uchun maxsus mavzu.

Professional teal color scheme with medical styling.
"""

from reportlab.lib.colors import HexColor
from .base import BaseTheme, ColorPalette, FontConfig


class MedicalColorPalette(ColorPalette):
    """Tibbiy hisobotlar uchun rang palitrasi."""

    def __init__(self):
        super().__init__()
        # Professional teal/medical colors
        self.primary = HexColor('#1a5f7a')       # Asosiy - to'q ko'k-yashil
        self.secondary = HexColor('#159895')     # Ikkinchi darajali - ko'k-yashil
        self.accent = HexColor('#57c5b6')        # Urg'u - och ko'k-yashil
        self.success = HexColor('#2d6a4f')       # Muvaffaqiyat - yashil
        self.warning = HexColor('#e76f51')       # Ogohlantirish - qizg'ish
        self.error = HexColor('#dc3545')         # Xato - qizil
        self.text = HexColor('#212529')          # Matn - to'q kulrang
        self.text_light = HexColor('#6c757d')    # Och matn
        self.background = HexColor('#ffffff')    # Fon - oq
        self.background_alt = HexColor('#f8f9fa')  # Altern. fon
        self.border = HexColor('#dee2e6')        # Chegara


class MedicalTheme(BaseTheme):
    """
    Tibbiy hisobotlar uchun mavzu.

    Xususiyatlar:
    - Professional teal rang sxemasi
    - Tibbiy hujjatlar uchun mos uslublar
    - DejaVu Sans shrifti (O'zbek tili uchun)
    """

    def __init__(self):
        # Override colors before calling parent init
        self.colors = MedicalColorPalette()
        self.fonts = FontConfig()
        self._register_fonts()
        self.styles = self._create_styles()

    def get_header_color(self):
        """Sarlavha rangini olish."""
        return self.colors.primary

    def get_accent_color(self):
        """Urg'u rangini olish."""
        return self.colors.accent

    def get_confidential_color(self):
        """Maxfiylik belgisi rangini olish."""
        return self.colors.warning
