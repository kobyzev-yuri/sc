# Работа с экспериментами

## Что такое эксперимент?

Эксперимент — это сохраненный набор результатов анализа WSI (Whole Slide Images), включающий:
- Результаты вычисления шкалы патологии (DataFrame с оценками)
- Обученные модели (SpectralAnalyzer, PCAScorer)
- Метаданные (настройки, время создания, количество образцов)
- Результаты сравнения методов (если проводилось сравнение)

## Куда сохраняются эксперименты?

Эксперименты сохраняются в директорию `experiments/` в корне проекта. Структура:

```
experiments/
├── experiment_YYYYMMDD_HHMMSS/          # Отдельный эксперимент
│   ├── results.csv                      # Основные результаты
│   ├── metadata.json                    # Метаданные эксперимента
│   ├── spectral_analyzer.pkl            # Обученная модель (если использовалась)
│   └── comparison/                     # Результаты сравнения методов (опционально)
│       ├── comparison.csv               # Таблица сравнения методов
│       ├── statistics.csv               # Статистика по методам
│       ├── configs.json                 # Конфигурации методов
│       ├── comparison_plot.png           # График сравнения
│       ├── results/                     # Результаты каждого метода
│       │   ├── pca_simple.csv
│       │   ├── spectral_p1_p99.csv
│       │   └── ...
│       └── models/                      # Сохраненные модели
│           ├── pca_simple.pkl
│           ├── spectral_p1_p99.pkl
│           └── ...
└── scale_comparison/                    # Именованный эксперимент сравнения
    └── ...
```

## Как настроить эксперимент на директорию с предиктами?

### Быстрый способ (рекомендуется)

Используйте готовый скрипт `test/run_experiment_from_predictions.py`:

```bash
# Базовый запуск на test/predictions
python test/run_experiment_from_predictions.py test/predictions

# Указать выходную директорию
python test/run_experiment_from_predictions.py test/predictions --output experiments/my_experiment

# Использовать PCA метод
python test/run_experiment_from_predictions.py test/predictions --method pca

# Спектральный анализ с кастомными процентилями
python test/run_experiment_from_predictions.py test/predictions --method spectral --percentiles 0.5 99.5

# Сравнение методов
python test/run_experiment_from_predictions.py test/predictions --method both --save-comparison
```

### Программно

```python
from pathlib import Path
from scale import aggregate
from scale import spectral_analysis
from scale.dashboard import create_experiment_dir, save_experiment

# Укажите директорию с предиктами
predictions_dir = Path("test/predictions")  # или "results/predictions"

# Загрузка предиктов
df = aggregate.load_predictions_batch(predictions_dir)

# Подготовка признаков
df_features = aggregate.create_relative_features(df)
df_features = aggregate.select_feature_columns(df_features)
df_features = df_features.fillna(0)

# Создание и запуск эксперимента
exp_dir = create_experiment_dir()
analyzer = spectral_analysis.SpectralAnalyzer()
analyzer.fit_pca(df_features)
df_pca = analyzer.transform_pca(df_features)
analyzer.fit_spectrum(df_pca, percentile_low=1.0, percentile_high=99.0)
df_results = analyzer.transform_spectrum(df_pca)

# Сохранение
save_experiment(
    exp_dir=exp_dir,
    df=df_results,
    analyzer=analyzer,
    metadata={"predictions_dir": str(predictions_dir)}
)
```

## Как сохранить эксперимент?

### 1. Через Streamlit Dashboard

В боковой панели:
1. Включите **"Использовать данные из results/predictions"** (или загрузите файлы)
2. Нажмите кнопку **"Сохранить эксперимент"**

Эксперимент будет сохранен с автоматически сгенерированным именем `experiment_YYYYMMDD_HHMMSS`.

**Примечание**: По умолчанию dashboard использует `results/predictions`. Чтобы использовать другую директорию (например, `test/predictions`), нужно либо:
- Изменить код в `scale/dashboard.py` (строка 302)
- Или использовать программный способ (см. выше)

### 2. Программно

```python
from pathlib import Path
from scale.dashboard import create_experiment_dir, save_experiment
from scale import spectral_analysis
import pandas as pd

# Создание директории эксперимента
exp_dir = create_experiment_dir()  # или create_experiment_dir("custom_dir")

# Подготовка данных
df_results = pd.DataFrame(...)  # Ваши результаты
analyzer = spectral_analysis.SpectralAnalyzer()
analyzer.fit_pca(df_features)
# ... обучение модели ...

# Сохранение эксперимента
save_experiment(
    exp_dir=exp_dir,
    df=df_results,
    analyzer=analyzer,  # опционально
    metadata={"description": "Мой эксперимент", "version": "1.0"}
)
```

### 3. Сохранение сравнения методов

```python
from scale.scale_comparison import ScaleComparison

comparison = ScaleComparison()
# ... добавление методов и вычисление результатов ...

# Сохранение всего сравнения
output_dir = Path("experiments/my_comparison")
comparison.save_results(output_dir)
```

## Как загрузить эксперимент?

### Загрузка метаданных и результатов

```python
import json
import pandas as pd
from pathlib import Path

exp_dir = Path("experiments/experiment_20251108_220146")

# Загрузка метаданных
with open(exp_dir / "metadata.json", "r") as f:
    metadata = json.load(f)
print(f"Эксперимент создан: {metadata['timestamp']}")
print(f"Настройки: {metadata.get('settings', {})}")

# Загрузка результатов
df_results = pd.read_csv(exp_dir / "results.csv")
print(df_results.head())
```

### Загрузка обученной модели

```python
from scale import spectral_analysis

exp_dir = Path("experiments/experiment_20251108_220146")

# Загрузка SpectralAnalyzer
analyzer = spectral_analysis.SpectralAnalyzer()
analyzer.load(exp_dir / "spectral_analyzer.pkl")

# Теперь можно использовать модель для новых данных
df_new_results = analyzer.transform_pca(df_new_features)
```

### Загрузка результатов сравнения методов

```python
import pandas as pd
import json
from pathlib import Path

comparison_dir = Path("experiments/scale_comparison")

# Загрузка сравнения
comparison_df = pd.read_csv(comparison_dir / "comparison.csv")
print(comparison_df.head())

# Загрузка статистики
stats_df = pd.read_csv(comparison_dir / "statistics.csv")
print(stats_df)

# Загрузка конфигураций
with open(comparison_dir / "configs.json", "r") as f:
    configs = json.load(f)
print(configs)

# Загрузка результатов конкретного метода
pca_results = pd.read_csv(comparison_dir / "results/pca_simple.csv")
```

## Как сравнить эксперименты?

### Сравнение двух экспериментов

```python
import pandas as pd
from pathlib import Path

# Загрузка результатов двух экспериментов
exp1_dir = Path("experiments/experiment_20251108_220146")
exp2_dir = Path("experiments/experiment_20251115_103022")

df1 = pd.read_csv(exp1_dir / "results.csv")
df2 = pd.read_csv(exp2_dir / "results.csv")

# Объединение для сравнения
comparison = pd.merge(
    df1[["image", "PC1_norm"]].rename(columns={"PC1_norm": "exp1_score"}),
    df2[["image", "PC1_norm"]].rename(columns={"PC1_norm": "exp2_score"}),
    on="image",
    how="outer"
)

# Вычисление различий
comparison["difference"] = comparison["exp1_score"] - comparison["exp2_score"]
print(comparison.describe())
```

### Сравнение нескольких экспериментов

```python
import pandas as pd
from pathlib import Path
import glob

# Поиск всех экспериментов
experiments = sorted(Path("experiments").glob("experiment_*"))

# Загрузка результатов всех экспериментов
all_results = {}
for exp_dir in experiments:
    if (exp_dir / "results.csv").exists():
        df = pd.read_csv(exp_dir / "results.csv")
        exp_name = exp_dir.name
        all_results[exp_name] = df[["image", "PC1_norm"]].rename(
            columns={"PC1_norm": exp_name}
        )

# Объединение всех результатов
comparison = None
for exp_name, df in all_results.items():
    if comparison is None:
        comparison = df
    else:
        comparison = pd.merge(comparison, df, on="image", how="outer")

print(comparison.head())
```

### Использование утилиты для сравнения

См. `test/compare_experiments.py` для готовой утилиты сравнения.

## Структура файлов эксперимента

### metadata.json

```json
{
  "timestamp": "2025-11-08T22:01:46.914744",
  "n_samples": 10,
  "settings": {
    "use_relative_features": true,
    "use_spectral_analysis": true,
    "percentile_low": 1.0,
    "percentile_high": 99.0
  }
}
```

### results.csv

CSV файл с колонками:
- `image` - имя изображения
- `PC1` - первая главная компонента
- `PC1_norm` - нормализованная оценка (0-1)
- `PC1_spectrum` - спектральная оценка (если использовался SpectralAnalyzer)
- `PC1_mode` - номер моды (если использовался SpectralAnalyzer)
- Другие колонки в зависимости от метода

### configs.json (для сравнения методов)

```json
{
  "pca_simple": {
    "method": "pca_scoring",
    "description": "Простая PCA нормализация (min-max на PC1)"
  },
  "spectral_p1_p99": {
    "method": "spectral_analysis",
    "percentile_low": 1.0,
    "percentile_high": 99.0,
    "use_gmm": false,
    "n_modes": 1,
    "description": "Спектральный анализ с процентилями [1.0, 99.0]"
  }
}
```

## Примеры использования

### Пример 1: Сохранение и загрузка простого эксперимента

```python
from pathlib import Path
from scale.dashboard import create_experiment_dir, save_experiment
from scale import spectral_analysis
import pandas as pd

# Создание и сохранение
exp_dir = create_experiment_dir()
df_results = pd.DataFrame({
    "image": ["img1", "img2", "img3"],
    "PC1": [1.5, 2.3, 0.8],
    "PC1_norm": [0.3, 0.7, 0.1]
})

save_experiment(exp_dir, df_results, metadata={"test": True})
print(f"Эксперимент сохранен: {exp_dir}")

# Загрузка
df_loaded = pd.read_csv(exp_dir / "results.csv")
print(df_loaded)
```

### Пример 2: Сравнение экспериментов с разными настройками

```python
from pathlib import Path
import pandas as pd

exp1 = Path("experiments/experiment_20251108_220146")
exp2 = Path("experiments/experiment_20251115_103022")

# Загрузка метаданных
import json
with open(exp1 / "metadata.json") as f:
    meta1 = json.load(f)
with open(exp2 / "metadata.json") as f:
    meta2 = json.load(f)

print("Настройки эксперимента 1:", meta1.get("settings"))
print("Настройки эксперимента 2:", meta2.get("settings"))

# Сравнение результатов
df1 = pd.read_csv(exp1 / "results.csv")
df2 = pd.read_csv(exp2 / "results.csv")

merged = pd.merge(
    df1[["image", "PC1_norm"]],
    df2[["image", "PC1_norm"]],
    on="image",
    suffixes=("_exp1", "_exp2")
)

merged["diff"] = merged["PC1_norm_exp1"] - merged["PC1_norm_exp2"]
print(f"Средняя разница: {merged['diff'].mean():.4f}")
print(f"Корреляция: {merged['PC1_norm_exp1'].corr(merged['PC1_norm_exp2']):.4f}")
```

## Рекомендации

1. **Именование**: Используйте описательные имена для экспериментов сравнения (например, `scale_comparison_v1`)
2. **Метаданные**: Всегда добавляйте метаданные с описанием эксперимента и параметрами
3. **Версионирование**: Сохраняйте версии моделей и параметров в метаданных
4. **Резервное копирование**: Регулярно делайте резервные копии важных экспериментов
5. **Документация**: Ведите README в директории экспериментов с описанием целей и результатов

