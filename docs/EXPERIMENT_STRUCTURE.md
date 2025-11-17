# Структура директории эксперимента

## Обзор

Эксперимент - это директория в `experiments/`, содержащая результаты анализа и конфигурацию для воспроизведения результатов.

## Два типа экспериментов

### 1. Эксперименты подбора признаков (feature_selection_*)

Создаются при запуске автоматизированного подбора признаков через `model_development/feature_selection_automated.py`.

**Обязательные файлы:**
- `best_features_YYYYMMDD_HHMMSS.json` - конфигурация лучшего метода с метриками
- `feature_selection_results_YYYYMMDD_HHMMSS.csv` - таблица сравнения всех методов

**Рекомендуемые файлы (для работы в dashboard):**
- `aggregated_data_YYYYMMDD_HHMMSS.csv` - агрегированные данные (абсолютные признаки: count, area)
- `relative_features_YYYYMMDD_HHMMSS.csv` - относительные признаки (нормализованные по Crypts)
- `all_features_YYYYMMDD_HHMMSS.csv` - все доступные признаки после фильтрации

**Дополнительные файлы:**
- `medical_report_YYYYMMDD_HHMMSS.md` - медицинский отчет с анализом результатов
- `ANALYSIS_PHASE*.md` - дополнительные аналитические отчеты (опционально)

**Пример структуры:**
```
experiments/feature_selection_quick/
├── best_features_20251116_143516.json          # ✅ Обязательно
├── feature_selection_results_20251116_143516.csv # ✅ Обязательно
├── aggregated_data_20251116_143516.csv         # ✅ Рекомендуется (для dashboard)
├── relative_features_20251116_143516.csv        # ✅ Рекомендуется (для dashboard)
├── all_features_20251116_143516.csv             # ✅ Рекомендуется (для dashboard)
└── medical_report_20251116_143516.md            # Опционально
```

### 2. Эксперименты из dashboard (experiment_*)

Создаются при сохранении результатов через веб-интерфейс dashboard.

**Обязательные файлы:**
- `results.csv` - результаты спектрального анализа (DataFrame с PC1, PC1_spectrum, PC1_mode и т.д.)
- `metadata.json` - метаданные эксперимента (настройки, время создания, количество образцов)

**Опциональные файлы:**
- `spectral_analyzer.pkl` - обученная модель SpectralAnalyzer (для инференса)
- `best_features_YYYYMMDD_HHMMSS.json` - конфигурация признаков (если были выбраны признаки)
- `comparison/` - директория с результатами сравнения методов (если проводилось сравнение)

**Пример структуры:**
```
experiments/experiment_20251108_220146/
├── results.csv                    # ✅ Обязательно
├── metadata.json                  # ✅ Обязательно
├── spectral_analyzer.pkl          # Опционально (для инференса)
├── best_features_20251108_220146.json  # Опционально
└── comparison/                    # Опционально
    ├── comparison.csv
    ├── statistics.csv
    ├── configs.json
    ├── comparison_plot.png
    ├── results/
    │   ├── pca_simple.csv
    │   └── spectral_p1_p99.csv
    └── models/
        ├── pca_simple.pkl
        └── spectral_p1_p99.pkl
```

## Формат файлов

### best_features_*.json

Конфигурация лучшего метода подбора признаков:

```json
{
  "method": "forward",
  "selected_features": [
    "Dysplasia_mean_relative_area",
    "Mild_relative_area",
    ...
  ],
  "metrics": {
    "score": 3.0783,
    "separation": 6.7901,
    "mean_pc1_norm_mod": 0.6802,
    "explained_variance": 0.5274,
    "mean_pc1_mod": 5.234,
    "mean_pc1_normal": -1.556
  },
  "timestamp": "20251116_143516"
}
```

### metadata.json

Метаданные эксперимента:

```json
{
  "timestamp": "2025-11-08T22:01:46.914744",
  "n_samples": 10,
  "settings": {
    "use_relative_features": true,
    "use_spectral_analysis": true,
    "percentile_low": 1.0,
    "percentile_high": 99.0
  },
  "source_experiment": "feature_selection_quick",
  "user_modified": false
}
```

### results.csv

Результаты спектрального анализа. Колонки:
- `image` - имя изображения
- `PC1` - первая главная компонента
- `PC1_spectrum` - спектральная оценка (0-1)
- `PC1_mode` - классификация (normal/mild/moderate/severe)
- Другие колонки в зависимости от метода

### aggregated_data_*.csv

Агрегированные данные из исходных предсказаний:
- Абсолютные признаки (count, area для каждого класса патологии)
- Исходные данные перед созданием относительных признаков

### relative_features_*.csv

Относительные признаки:
- Признаки, нормализованные по Crypts
- Формат: `{class}_relative_count`, `{class}_relative_area`, `{class}_mean_relative_area`

### all_features_*.csv

Все доступные признаки:
- Все признаки после фильтрации (без служебных колонок)
- Используется для подбора признаков

## Проверка наличия данных

Для работы в dashboard эксперимент должен содержать хотя бы один из следующих файлов:
- `aggregated_data_*.csv`
- `relative_features_*.csv`
- `all_features_*.csv`

Эксперименты без этих файлов не будут отображаться в списке доступных экспериментов (если включена проверка данных).

## Минимальная структура для работы в dashboard

Для использования эксперимента в dashboard достаточно:
1. `best_features_*.json` - для загрузки конфигурации признаков
2. Один из файлов данных: `aggregated_data_*.csv`, `relative_features_*.csv` или `all_features_*.csv` - для загрузки данных

Для инференса дополнительно требуется:
- `spectral_analyzer.pkl` - обученная модель

## Рекомендации

1. **Всегда сохраняйте данные** - включайте `aggregated_data`, `relative_features` и `all_features` при экспорте
2. **Сохраняйте модель** - `spectral_analyzer.pkl` необходим для инференса новых данных
3. **Используйте метаданные** - `metadata.json` помогает отслеживать настройки эксперимента
4. **Версионирование** - используйте timestamp в именах файлов для отслеживания версий

