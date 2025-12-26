"""
Uzbek Translations for Morphometry Reports
==========================================
Medical and scientific terminology in Uzbek language.

O'zbekcha tibbiy va ilmiy terminologiya.
"""

# Report titles and headers
REPORT_TITLE = "Morfometriya Tahlili Hisoboti"
ANALYSIS_INFO = "Tahlil Ma'lumotlari"
SUMMARY_STATISTICS = "Umumiy Statistika"
SEGMENTATION_RESULTS = "Segmentatsiya Natijalari"
DISTRIBUTION_ANALYSIS = "Taqsimot Tahlili"
INDIVIDUAL_CELL_MEASUREMENTS = "Alohida Hujayra O'lchovlari"

# Metadata labels
SAMPLE_ID = "Namuna raqami"
IMAGE_FILE = "Rasm fayli"
ANALYSIS_DATE = "Tahlil sanasi"
TOTAL_CELLS = "Jami hujayralar"
PIXEL_SIZE = "Piksel o'lchami"
SEGMENTATION_MODEL = "Segmentatsiya modeli"
PREPROCESSING = "Oldindan ishlov berish"
NOT_AVAILABLE = "Mavjud emas"
AUTHOR = "Morfometriya Tahlil Tizimi"

# Medical/Morphometric terms
METRICS = {
    'area': "Maydon (px²)",
    'area_um': "Maydon (µm²)",
    'perimeter': "Perimetr (px)",
    'perimeter_um': "Perimetr (µm)",
    'circularity': "Aylanasimonlik",
    'eccentricity': "Ekssentriklik",
    'solidity': "Zichlik",
    'aspect_ratio': "Tomonlar nisbati",
    'intensity_mean': "O'rtacha intensivlik",
    'intensity_std': "Intensivlik standart og'ishi",
    'intensity_min': "Minimal intensivlik",
    'intensity_max': "Maksimal intensivlik",
    'major_axis_length': "Katta o'q uzunligi",
    'minor_axis_length': "Kichik o'q uzunligi",
    'orientation': "Orientatsiya",
    'convex_area': "Qavariq maydon",
    'extent': "Kenglik",
    'equivalent_diameter': "Ekvivalent diametr",
    'feret_diameter_max': "Maksimal Feret diametri",
}

# Short metric names for table headers
METRIC_SHORT = {
    'cell_id': "Hujayra ID",
    'area': "Maydon",
    'perimeter': "Perimetr",
    'circularity': "Aylanasimonlik",
    'eccentricity': "Ekssentriklik",
    'solidity': "Zichlik",
    'aspect_ratio': "Nisbat",
    'intensity_mean': "Int. o'rtacha",
    'centroid_x': "Markaz X",
    'centroid_y': "Markaz Y",
}

# Statistical terms
STAT_LABELS = {
    'metric': "Ko'rsatkich",
    'mean': "O'rtacha",
    'std': "St. og'ish",
    'min': "Min",
    'max': "Maks",
    'median': "Mediana",
    'count': "Soni",
}

# Chart/Graph labels
CHART_LABELS = {
    'original_image': "Asl rasm",
    'segmentation': "Segmentatsiya",
    'cells_detected': "aniqlangan hujayralar",
    'distribution': "Taqsimot",
    'count': "Soni",
    'frequency': "Chastota",
}

# Messages
MESSAGES = {
    'no_cells_detected': "Hujayralar aniqlanmadi.",
    'more_rows': "... (yana {n} qator)",
    'report_generated': "Hisobot yaratildi",
    'analysis_complete': "Tahlil muvaffaqiyatli yakunlandi",
    'processing': "Ishlov berilmoqda...",
}

# Preprocessing step translations
PREPROCESSING_STEPS = {
    'denoise': "Shovqinni yo'qotish",
    'clahe': "Kontrastni yaxshilash (CLAHE)",
    'normalize': "Normalizatsiya",
    'background_subtraction': "Fon olib tashlash",
    'gamma_correction': "Gamma tuzatish",
    'sharpen': "Keskinlashtirish",
    'smooth': "Silliqlashtirish",
}

# Modality translations
MODALITIES = {
    'brightfield': "Yorug' maydon",
    'fluorescence': "Fluoressensiya",
    'phase_contrast': "Fazaviy kontrast",
    'he_stained': "Gematoksilin-eozin bo'yalgan",
}

# Model translations
MODELS = {
    'cyto3': "Cellpose Cyto3",
    'nuclei': "Cellpose Nuclei",
    'cyto2': "Cellpose Cyto2",
    'cpsam': "Cellpose-SAM",
    'watershed': "Watershed",
    'otsu': "Otsu binarizatsiya",
}

# Excel sheet names
EXCEL_SHEETS = {
    'measurements': "Hujayra o'lchovlari",
    'summary': "Umumiy statistika",
    'metadata': "Ma'lumotlar",
}

# PDF professional elements
PDF_ELEMENTS = {
    'page': "Sahifa",
    'of': "dan",
    'report_id': "Hisobot raqami",
    'generated_by': "Tizim tomonidan yaratildi",
    'confidential': "MAXFIY - Faqat tibbiy xodimlar uchun",
    'conclusions': "Xulosa va Tavsiyalar",
    'quality_assessment': "Sifat Baholash",
    'interpretation_guide': "Natijalarni Talqin Qilish",
}

# Quality assessment labels
QUALITY_LABELS = {
    'excellent': "A'lo",
    'good': "Yaxshi",
    'acceptable': "Qoniqarli",
    'poor': "Yomon",
    'cell_count': "Hujayra soni",
    'segmentation_quality': "Segmentatsiya sifati",
    'image_quality': "Rasm sifati",
}

# Reference ranges for interpretation
REFERENCE_RANGES = {
    'circularity': {
        'normal': (0.7, 1.0),
        'description': "Normal hujayralar uchun aylanasimonlik 0.7-1.0 oralig'ida bo'ladi",
    },
    'solidity': {
        'normal': (0.9, 1.0),
        'description': "Sog'lom hujayralar yuqori zichlikka ega (>0.9)",
    },
    'aspect_ratio': {
        'normal': (1.0, 2.0),
        'description': "Normal hujayralar nisbatan simmetrik (1.0-2.0)",
    },
}

# Conclusion templates
CONCLUSION_TEMPLATES = {
    'normal': "Tahlil qilingan hujayralar normal morfologik xususiyatlarni ko'rsatmoqda.",
    'abnormal_shape': "Ba'zi hujayralarda shakl anomaliyalari kuzatildi. Qo'shimcha tekshiruv tavsiya etiladi.",
    'high_variability': "Hujayra o'lchamlarida yuqori o'zgaruvchanlik mavjud.",
    'low_cell_count': "Kam hujayra soni - natijalar statistik jihatdan cheklangan bo'lishi mumkin.",
}

# JSON field translations (for reference)
JSON_FIELDS = {
    'metadata': "ma'lumotlar",
    'summary': "xulosa",
    'measurements': "o'lchovlar",
    'total_cells': "jami_hujayralar",
    'statistics': "statistika",
}


def get_metric_label(metric_name: str, with_unit: bool = True) -> str:
    """
    Get Uzbek label for a metric.

    Args:
        metric_name: Internal metric name (e.g., 'area', 'circularity')
        with_unit: Include unit in label

    Returns:
        Uzbek label for the metric
    """
    if with_unit and metric_name in METRICS:
        return METRICS[metric_name]
    elif metric_name in METRIC_SHORT:
        return METRIC_SHORT[metric_name]
    else:
        # Fallback: capitalize and replace underscores
        return metric_name.replace('_', ' ').title()


def get_preprocessing_label(step: str) -> str:
    """Get Uzbek label for preprocessing step."""
    return PREPROCESSING_STEPS.get(step, step)


def get_modality_label(modality: str) -> str:
    """Get Uzbek label for imaging modality."""
    return MODALITIES.get(modality, modality)


def get_model_label(model: str) -> str:
    """Get Uzbek label for segmentation model."""
    return MODELS.get(model, model)
