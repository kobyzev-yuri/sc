"""
Модуль для анализа патологических данных из Whole Slide Images.

Поддерживает:
- Агрегацию предсказаний патологий
- PCA анализ и создание шкалы оценки
- Спектральный анализ с выявлением мод
- Кластеризацию данных
- Веб-дашборд для визуализации
"""

from . import aggregate
from . import pca_scoring
from . import spectral_analysis
from . import clustering
from . import cluster_scoring
from . import cluster_comparison
from . import method_comparison
from . import preprocessing
from . import eda
from . import scale_comparison

# dashboard не импортируется здесь, чтобы избежать циклических импортов
# Используйте: from scale import dashboard напрямую

__all__ = [
    "aggregate",
    "pca_scoring",
    "spectral_analysis",
    "clustering",
    "cluster_scoring",
    "cluster_comparison",
    "method_comparison",
    "preprocessing",
    "eda",
    "scale_comparison",
]

