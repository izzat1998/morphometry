"""
Metadata Section
================
Tahlil ma'lumotlari bo'limi.
"""

from typing import List, Any, Optional

from .base import BaseSection


class MetadataSection(BaseSection):
    """
    Ma'lumotlar bo'limi.

    Namuna, fayl, sana va boshqa ma'lumotlar.
    """

    def build(
        self,
        sample_id: str = "",
        image_filename: str = "",
        analysis_date: str = "",
        total_cells: int = 0,
        pixel_size: float = 1.0,
        segmentation_model: str = "",
        preprocessing_steps: Optional[List[str]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Ma'lumotlar bo'limini yaratish.

        Args:
            sample_id: Namuna raqami
            image_filename: Rasm fayli nomi
            analysis_date: Tahlil sanasi
            total_cells: Jami hujayralar soni
            pixel_size: Piksel o'lchami (µm)
            segmentation_model: Segmentatsiya modeli
            preprocessing_steps: Oldindan ishlov berish bosqichlari

        Returns:
            Elementlar ro'yxati
        """
        elements = []

        # Section header
        header = self.get_label('analysis_info', "Tahlil Ma'lumotlari")
        elements.append(self.text_builder.section_header(header))

        # Build key-value pairs
        na = self.get_label('not_available', "Mavjud emas")

        data = [
            (self.get_label('sample_id', "Namuna raqami"), sample_id or na),
            (self.get_label('image_file', "Rasm fayli"), image_filename or na),
            (self.get_label('analysis_date', "Tahlil sanasi"), analysis_date),
            (self.get_label('total_cells', "Jami hujayralar"), str(total_cells)),
            (self.get_label('pixel_size', "Piksel o'lchami"), f"{pixel_size} µm"),
            (self.get_label('segmentation_model', "Segmentatsiya modeli"), segmentation_model or na),
        ]

        # Add preprocessing steps if available
        if preprocessing_steps:
            prep_label = self.get_label('preprocessing', "Oldindan ishlov berish")
            prep_value = ", ".join(preprocessing_steps)
            data.append((prep_label, prep_value))

        # Create table
        table = self.table_builder.build_key_value_table(data)
        elements.append(table)

        elements.append(self.text_builder.spacer(15))

        return elements
