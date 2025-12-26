"""
Text Builder Component
======================
Matn elementlarini yaratish uchun komponent.
"""

from typing import List, Optional
from reportlab.platypus import Paragraph, Spacer, HRFlowable

from ..themes.base import BaseTheme


class TextBuilder:
    """
    Matn elementlari yaratuvchi.

    Sarlavhalar, paragraflar, bo'linish chiziqlari va boshqalar.
    """

    def __init__(self, theme: BaseTheme):
        self.theme = theme

    def title(self, text: str) -> Paragraph:
        """Sarlavha yaratish."""
        return Paragraph(text, self.theme.get_style('Title'))

    def subtitle(self, text: str) -> Paragraph:
        """Kichik sarlavha yaratish."""
        return Paragraph(text, self.theme.get_style('Subtitle'))

    def section_header(self, text: str) -> Paragraph:
        """Bo'lim sarlavhasi yaratish."""
        return Paragraph(text, self.theme.get_style('SectionHeader'))

    def subsection_header(self, text: str) -> Paragraph:
        """Kichik bo'lim sarlavhasi yaratish."""
        return Paragraph(text, self.theme.get_style('SubsectionHeader'))

    def body(self, text: str) -> Paragraph:
        """Asosiy matn yaratish."""
        return Paragraph(text, self.theme.get_style('Body'))

    def conclusion(self, text: str) -> Paragraph:
        """Xulosa qutisi yaratish."""
        return Paragraph(text, self.theme.get_style('Conclusion'))

    def spacer(self, height: float = 12) -> Spacer:
        """Bo'sh joy yaratish."""
        return Spacer(1, height)

    def horizontal_line(
        self,
        color: Optional[any] = None,
        thickness: float = 2,
        space_before: float = 5,
        space_after: float = 15
    ) -> HRFlowable:
        """
        Gorizontal chiziq yaratish.

        Args:
            color: Chiziq rangi (None bo'lsa, primary rang)
            thickness: Chiziq qalinligi
            space_before: Oldidan bo'sh joy
            space_after: Keyinidan bo'sh joy

        Returns:
            HRFlowable
        """
        if color is None:
            color = self.theme.colors.primary

        return HRFlowable(
            width="100%",
            thickness=thickness,
            color=color,
            spaceBefore=space_before,
            spaceAfter=space_after
        )

    def accent_line(self) -> HRFlowable:
        """Urg'u chizig'i yaratish."""
        return HRFlowable(
            width="100%",
            thickness=1,
            color=self.theme.colors.accent,
            spaceBefore=5,
            spaceAfter=10
        )

    def numbered_list(self, items: List[str], style_name: str = 'Conclusion') -> List:
        """
        Raqamlangan ro'yxat yaratish.

        Args:
            items: Elementlar ro'yxati
            style_name: Uslub nomi

        Returns:
            Elementlar ro'yxati
        """
        elements = []
        style = self.theme.get_style(style_name)

        for i, item in enumerate(items, 1):
            elements.append(Paragraph(f"{i}. {item}", style))
            elements.append(self.spacer(8))

        return elements

    def bullet_list(self, items: List[str], style_name: str = 'Body') -> List:
        """
        Nuqtali ro'yxat yaratish.

        Args:
            items: Elementlar ro'yxati
            style_name: Uslub nomi

        Returns:
            Elementlar ro'yxati
        """
        elements = []
        style = self.theme.get_style(style_name)

        for item in items:
            elements.append(Paragraph(f"• {item}", style))
            elements.append(self.spacer(4))

        return elements
