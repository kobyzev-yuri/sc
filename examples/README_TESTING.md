# Тестирование подходов к построению шкалы

Этот документ описывает, как тестировать и сравнивать разные подходы к построению шкалы оценки патологии.

## Структура

- `test_with_predictions.py` - простой тест с одним методом (spectral analysis)
- `test_scale_comparison.py` - расширенный тест с несколькими методами
- `run_experiments.py` - запуск экспериментов из конфигурационного файла
- `experiment_config.json` - конфигурация экспериментов

## Быстрый старт

### 1. Простой тест (один метод)

```bash
cd /mnt/ai/cnn/sc
python examples/test_with_predictions.py
```

Этот скрипт:
- Загружает предсказания из `results/predictions/`
- Применяет spectral analysis
- Сохраняет результаты в `experiments/test_run/`

### 2. Сравнение нескольких методов

```bash
python examples/test_scale_comparison.py
```

Этот скрипт тестирует:
- PCA Scoring (простая нормализация)
- Spectral Analysis с разными процентилями (1-99, 0.5-99.5, 5-95)
- Spectral Analysis с GMM

Результаты сохраняются в `experiments/scale_comparison/`:
- `comparison.csv` - сравнение всех методов
- `statistics.csv` - статистика по методам
- `configs.json` - конфигурации экспериментов
- `results/` - отдельные результаты каждого метода
- `models/` - сохраненные модели
- `comparison_plot.png` - график сравнения

### 3. Запуск из конфигурации

```bash
python examples/run_experiments.py
```

Или с кастомным конфигом:
```bash
python examples/run_experiments.py --config path/to/config.json
```

## Доступные методы

### 1. PCA Scoring (`pca_scoring`)

Простая нормализация PC1 в диапазон [0, 1] через min-max:
- `PC1_norm = (PC1 - PC1_min) / (PC1_max - PC1_min)`

**Плюсы:**
- Простота
- Быстрота
- Использует всю информацию из PC1

**Минусы:**
- Не учитывает структуру распределения
- Чувствителен к выбросам

### 2. Spectral Analysis (`spectral_analysis`)

Расширенный метод с учетом мод распределения:
- Использует процентили для буферных зон
- Выявляет стабильные состояния (моды) через KDE
- Опционально: GMM для моделирования состояний

**Параметры:**
- `percentile_low` - нижний процентиль (по умолчанию 1.0)
- `percentile_high` - верхний процентиль (по умолчанию 99.0)
- `use_gmm` - использовать ли GMM (по умолчанию False)

**Плюсы:**
- Учитывает структуру распределения
- Выявляет стабильные состояния
- Менее чувствителен к выбросам (благодаря процентилям)

**Минусы:**
- Более сложный
- Требует настройки процентилей

## Настройка экспериментов

Отредактируйте `experiment_config.json`:

```json
{
  "experiments": [
    {
      "name": "my_experiment",
      "method": "spectral_analysis",
      "description": "Мой эксперимент",
      "percentile_low": 2.0,
      "percentile_high": 98.0,
      "use_gmm": false,
      "enabled": true
    }
  ]
}
```

## Интерпретация результатов

### Сравнение методов

Файл `comparison.csv` содержит шкалы всех методов для каждого образца:
- `image` - имя образца
- `{method}_score` - шкала метода (0-1)
- `{method}_PC1` - исходное значение PC1

### Статистика

Файл `statistics.csv` содержит:
- `score_mean`, `score_std`, `score_min`, `score_max`, `score_median` - статистика шкалы
- `PC1_mean`, `PC1_std`, `PC1_min`, `PC1_max` - статистика PC1
- `n_modes` - число мод (для spectral analysis)
- `percentile_low`, `percentile_high` - используемые процентили

### Визуализация

График `comparison_plot.png` показывает:
1. Распределение шкал по методам (гистограммы)
2. Boxplot сравнение методов
3. Scatter plot сравнения двух методов
4. Корреляционная матрица между методами

## Рекомендации

1. **Начните с простого**: используйте `pca_simple` как baseline
2. **Экспериментируйте с процентилями**: попробуйте разные значения для spectral analysis
3. **Сравнивайте результаты**: смотрите на корреляцию между методами
4. **Анализируйте выбросы**: проверьте, какие образцы дают разные результаты
5. **Используйте GMM**: если нужно моделировать смесь состояний

## Добавление нового метода

Чтобы добавить новый метод, отредактируйте `scale/scale_comparison.py`:

1. Добавьте метод в класс `ScaleComparison`:
```python
def test_my_method(self, df_features, name="my_method", **kwargs):
    # Ваша реализация
    pass
```

2. Добавьте вызов в `test_scale_comparison.py` или `run_experiments.py`

3. Обновите `experiment_config.json` с новым методом


