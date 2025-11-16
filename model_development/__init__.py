"""
Модуль для развития методов построения моделей для шкалы патологии.

Содержит:
- Автоматизированный подбор признаков
- Экспорт результатов в формат experiments
- Версионирование конфигураций
- Система отслеживания экспериментов
"""

from .feature_selection_automated import (
    FeatureSelector,
    evaluate_feature_set,
    identify_sample_type,
    run_feature_selection_analysis,
)

from .feature_selection_export import (
    export_to_dashboard_config,
    export_to_experiment_format,
)

from .experiment_tracker import (
    ExperimentTracker,
    register_experiment_from_directory,
)

__all__ = [
    'FeatureSelector',
    'evaluate_feature_set',
    'identify_sample_type',
    'run_feature_selection_analysis',
    'export_to_dashboard_config',
    'export_to_experiment_format',
    'ExperimentTracker',
    'register_experiment_from_directory',
]

