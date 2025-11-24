"""
Модуль для анализа патологических данных из Whole Slide Images.

Поддерживает:
- Агрегацию предсказаний патологий
- PCA анализ и создание шкалы оценки
- Спектральный анализ с выявлением мод
- Веб-дашборд для визуализации
"""

from . import aggregate
from . import pca_scoring
from . import spectral_analysis
from . import preprocessing
from . import eda
from . import scale_comparison

# dashboard не импортируется здесь, чтобы избежать циклических импортов
# Используйте: from scale import dashboard напрямую

__all__ = [
    "aggregate",
    "pca_scoring",
    "spectral_analysis",
    "preprocessing",
    "eda",
    "scale_comparison",
]

