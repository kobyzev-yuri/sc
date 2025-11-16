# Развитие методов построения моделей для шкалы патологии

Директория содержит код для автоматизированного подбора признаков и развития методов построения моделей для медицинской шкалы патологии.

## Структура

- `feature_selection_automated.py` - Автоматизированный подбор признаков с различными методами
- `feature_selection_export.py` - Экспорт результатов в формат dashboard и experiments
- `feature_selection_versioning.py` - Система версионирования и управления экспериментами
- `feature_selection_versioning_cli.py` - CLI утилита для управления версиями

## Использование

### Базовый пример подбора признаков

```python
from model_development import FeatureSelector, run_feature_selection_analysis
from scale import aggregate

# Загрузка данных
df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)

# Запуск анализа
results_df = run_feature_selection_analysis(
    predictions_dir="results/predictions",
    output_dir="experiments/feature_selection/my_experiment"
)
```

### Экспорт в формат experiments

```python
from model_development import export_to_experiment_format
from scale import spectral_analysis

# После подбора признаков и обучения модели
experiment_dir = export_to_experiment_format(
    selected_features=best_features,
    output_dir=Path("experiments/my_experiment"),
    method_name="positive_loadings",
    metrics=metrics,
    df_results=df_spectrum,
    analyzer=analyzer,
    use_relative_features=True
)
```

## Формат experiments

Результаты сохраняются в директорию `experiments/` в следующем формате:

```
experiments/my_experiment/
├── best_features_YYYYMMDD_HHMMSS.json  # Конфигурация признаков
├── results.csv                          # Результаты спектрального анализа
├── spectral_analyzer.pkl                 # Обученная модель
└── metadata.json                         # Метаданные эксперимента
```

Этот формат совместим с dashboard и позволяет загружать эксперименты через веб-интерфейс.

## Методы подбора признаков

- **positive_loadings** - Фильтрация по положительным loadings PC1
- **mutual_info** - Mutual Information
- **f_test** - F-test (ANOVA)
- **rfe** - Recursive Feature Elimination
- **rf_importance** - Random Forest importance
- **lasso** - Lasso regularization

## Версионирование

Используйте CLI утилиту для управления версиями экспериментов:

```bash
python -m model_development.feature_selection_versioning_cli list
python -m model_development.feature_selection_versioning_cli export <experiment_name>
python -m model_development.feature_selection_versioning_cli compare <exp1> <exp2>
```

