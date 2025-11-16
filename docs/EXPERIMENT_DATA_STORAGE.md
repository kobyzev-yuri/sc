# Хранение данных в экспериментах

## Что сохраняется в эксперименте

При запуске подбора признаков теперь сохраняются **не только конфиги, но и агрегированные данные**:

### 1. Конфигурации и метрики

- ✅ `best_features_*.json` - лучший набор признаков с метриками
- ✅ `feature_selection_results_*.csv` - таблица сравнения методов
- ✅ `medical_report_*.md` - медицинский отчет
- ✅ `scale/cfg/feature_selection_config_relative.json` - конфигурация для dashboard

### 2. Агрегированные данные (НОВОЕ!)

- ✅ `aggregated_data_*.csv` - агрегированные данные из исходных предсказаний
  - Абсолютные признаки (count, area для каждого класса)
  - Исходные данные перед созданием относительных признаков
  
- ✅ `relative_features_*.csv` - относительные признаки
  - Признаки, нормализованные по Crypts
  - Все относительные признаки (relative_count, relative_area, mean_relative_area)
  
- ✅ `all_features_*.csv` - все доступные признаки
  - Все признаки после фильтрации (без служебных колонок)
  - Используется для подбора признаков

---

## Структура файлов эксперимента

```
experiments/
└── feature_selection_quick/
    ├── best_features_*.json              ← Конфигурация лучшего метода
    ├── feature_selection_results_*.csv   ← Сравнение методов
    ├── medical_report_*.md               ← Медицинский отчет
    ├── scale/cfg/feature_selection_config_relative.json  ← Конфигурация для dashboard
    │
    ├── aggregated_data_*.csv             ← Агрегированные данные (НОВОЕ!)
    ├── relative_features_*.csv          ← Относительные признаки (НОВОЕ!)
    └── all_features_*.csv                ← Все доступные признаки (НОВОЕ!)
```

---

## Зачем сохранять данные?

### ✅ Воспроизводимость результатов

С сохраненными данными можно:
- Воспроизвести результаты эксперимента
- Проверить метрики на тех же данных
- Сравнить разные методы на одинаковых данных

### ✅ Анализ без пересчета

Можно:
- Загрузить сохраненные данные для анализа
- Не нужно заново агрегировать предсказания
- Быстро проверить результаты

### ✅ Отслеживание изменений

Можно:
- Сравнить данные разных экспериментов
- Понять, как изменились данные со временем
- Отследить влияние изменений в предсказаниях

---

## Использование сохраненных данных

### Загрузка данных из эксперимента

```python
import pandas as pd
from pathlib import Path

experiment_dir = Path("experiments/feature_selection_quick")

# Загрузить агрегированные данные
df_aggregated = pd.read_csv(experiment_dir / "aggregated_data_*.csv")

# Загрузить относительные признаки
df_features = pd.read_csv(experiment_dir / "relative_features_*.csv")

# Загрузить все доступные признаки
df_all = pd.read_csv(experiment_dir / "all_features_*.csv")

# Загрузить конфигурацию
import json
with open(experiment_dir / "best_features_*.json") as f:
    config = json.load(f)
    selected_features = config['selected_features']
```

### Воспроизведение результатов

```python
from scale import pca_scoring

# Загрузить данные из эксперимента
df_all = pd.read_csv("experiments/feature_selection_quick/all_features_*.csv")

# Загрузить отобранные признаки
with open("experiments/feature_selection_quick/best_features_*.json") as f:
    config = json.load(f)
    selected_features = config['selected_features']

# Применить PCA с теми же признаками
pca_scorer = pca_scoring.PCAScorer()
df_pca = pca_scorer.fit_transform(df_all[['image'] + selected_features])

# Проверить метрики
print(f"PC1 range: {df_pca['PC1'].min():.4f} - {df_pca['PC1'].max():.4f}")
```

### Сравнение данных разных экспериментов

```python
import pandas as pd

# Загрузить данные из двух экспериментов
df1 = pd.read_csv("experiments/feature_selection_quick/all_features_*.csv")
df2 = pd.read_csv("experiments/feature_selection_no_paneth/all_features_*.csv")

# Сравнить количество признаков
print(f"Эксперимент 1: {len(df1.columns)} признаков")
print(f"Эксперимент 2: {len(df2.columns)} признаков")

# Сравнить образцы
print(f"Эксперимент 1: {len(df1)} образцов")
print(f"Эксперимент 2: {len(df2)} образцов")
```

---

## Размер файлов

### Типичные размеры:

- `aggregated_data_*.csv`: ~50-200 KB (зависит от числа образцов)
- `relative_features_*.csv`: ~100-400 KB
- `all_features_*.csv`: ~100-400 KB

**Общий размер эксперимента:** ~500 KB - 2 MB

---

## Рекомендации

### ✅ Сохранять данные:

- При важных экспериментах
- Для воспроизводимости результатов
- Для сравнения с другими экспериментами

### ⚠️ Можно не сохранять:

- При быстрых тестах (можно отключить через параметр)
- Если данные очень большие
- Если нужно сэкономить место

---

## Отключение сохранения данных

Если нужно сохранить только конфиги без данных:

```python
saved_files = feature_selection_export.export_complete_results(
    results_df=results_df,
    output_dir=output_dir,
    use_relative_features=True,
    auto_export_to_dashboard=False,
    df_aggregated=None,  # Не сохранять
    df_features=None,  # Не сохранять
    df_all_features=None,  # Не сохранять
)
```

---

## FAQ

### Q: Обязательно ли сохранять данные?

**A:** Нет, это опционально. Можно передать `None` для любого из параметров данных.

### Q: Можно ли загрузить данные из старого эксперимента?

**A:** Да, если файлы были сохранены. Просто загрузите CSV файлы из директории эксперимента.

### Q: Занимают ли данные много места?

**A:** Нет, обычно ~500 KB - 2 MB на эксперимент. Это приемлемо для большинства случаев.

### Q: Можно ли удалить данные, оставив только конфиги?

**A:** Да, можно удалить CSV файлы с данными, оставив только конфиги и отчеты.

---

## Заключение

Теперь эксперименты сохраняют **полную информацию**:
- ✅ Конфигурации признаков
- ✅ Метрики качества
- ✅ Агрегированные данные
- ✅ Относительные признаки
- ✅ Все доступные признаки

Это позволяет:
- Воспроизводить результаты
- Сравнивать эксперименты
- Анализировать данные без пересчета
- Отслеживать изменения


