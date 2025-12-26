"""
Base Section
============
Barcha bo'limlar uchun asosiy klass.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Union

from ..themes.base import BaseTheme
from ..components.tables import TableBuilder
from ..components.charts import ChartBuilder
from ..components.text import TextBuilder


class BaseSection(ABC):
    """
    Asosiy bo'lim klassi.

    Barcha bo'limlar uchun umumiy interfeys.
    """

    def __init__(self, theme: BaseTheme, labels: dict):
        """
        Bo'limni yaratish.

        Args:
            theme: Mavzu obyekti
            labels: O'zbekcha teglar lug'ati
        """
        self.theme = theme
        self.labels = labels
        self.table_builder = TableBuilder(theme)
        self.chart_builder = ChartBuilder(theme)
        self.text_builder = TextBuilder(theme)

    @abstractmethod
    def build(self, **kwargs) -> List[Any]:
        """
        Bo'lim elementlarini yaratish.

        Returns:
            ReportLab elementlari ro'yxati
        """
        pass

    def get_label(self, key: str, default: str = "") -> Union[str, dict]:
        """
        Tegni olish - ichki kalitlarni ham qo'llab-quvvatlaydi.

        Args:
            key: Kalit nomi ('messages.no_cells_detected' kabi nuqta bilan ajratilgan)
            default: Standart qiymat

        Returns:
            Teg qiymati yoki default

        Misollar:
            get_label('report_title')  # Oddiy kalit
            get_label('messages.no_cells_detected')  # Ichki kalit
            get_label('stats.mean')  # stats dict ichidan
        """
        keys = key.split('.')
        value = self.labels

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def get_metric_label(self, metric: str) -> str:
        """Ko'rsatkich nomini olish."""
        metrics = self.labels.get('metrics', {})
        return metrics.get(metric, metric)

    def get_stat_label(self, stat: str) -> str:
        """Statistik teg olish."""
        stats = self.labels.get('stats', {})
        return stats.get(stat, stat)

    def get_quality_label(self, quality: str) -> str:
        """Sifat baholash tegini olish."""
        quality_labels = self.labels.get('quality', {})
        return quality_labels.get(quality, quality)

    def get_message(self, message_key: str, default: str = "") -> str:
        """Xabar matnini olish."""
        messages = self.labels.get('messages', {})
        return messages.get(message_key, default)
