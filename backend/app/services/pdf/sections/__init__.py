"""
PDF Sections Module
===================
Hisobot bo'limlari.
"""

from .base import BaseSection
from .title import TitleSection
from .metadata import MetadataSection
from .summary import SummarySection
from .quality import QualitySection
from .visualizations import VisualizationSection
from .distributions import DistributionSection
from .conclusions import ConclusionsSection
from .data_table import DataTableSection

__all__ = [
    'BaseSection',
    'TitleSection',
    'MetadataSection',
    'SummarySection',
    'QualitySection',
    'VisualizationSection',
    'DistributionSection',
    'ConclusionsSection',
    'DataTableSection'
]
