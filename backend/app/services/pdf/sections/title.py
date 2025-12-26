"""
Title Section
=============
Hisobot sarlavhasi bo'limi.
"""

from typing import List, Any

from .base import BaseSection


class TitleSection(BaseSection):
    """
    Sarlavha bo'limi.

    Hisobot sarlavhasi, kichik sarlavha va ajratuvchi chiziq.
    """

    def build(
        self,
        title: str,
        subtitle: str = "",
        **kwargs
    ) -> List[Any]:
        """
        Sarlavha bo'limini yaratish.

        Args:
            title: Asosiy sarlavha
            subtitle: Kichik sarlavha (sana, fayl nomi va h.k.)

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        # Main title
        elements.append(self.text_builder.title(title))

        # Subtitle
        if subtitle:
            elements.append(self.text_builder.subtitle(subtitle))

        # Decorative line
        elements.append(self.text_builder.horizontal_line())

        return elements
