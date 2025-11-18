# Автоматизированный подбор признаков для медицинской шкалы

## Цель

Найти оптимальный набор признаков, при котором образцы с заболеваниями (mod) получают **высокие значения PC1** (ближе к 1 на нормализованной шкале), что обеспечивает корректное разделение нормальных и патологических образцов.

## Методы автоматизированного поиска

### Метод 1: Forward Selection (Жадный алгоритм)

**Принцип:** Начинает с пустого набора и последовательно добавляет признаки, которые максимизируют целевую метрику.

**Преимущества:**
- Быстрый и эффективный
- Хорошо работает для большого числа признаков
- Гарантирует монотонное улучшение метрики

**Недостатки:**
- Может застревать в локальных оптимумах
- Не учитывает взаимодействия между признаками

**Использование:**
```python
from scale.feature_selection_automated import FeatureSelector, aggregate

df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_all = aggregate.select_all_feature_columns(df_features)

selector = FeatureSelector(df_all)
candidate_features = [c for c in df_all.columns if c != 'image']

features, metrics = selector.method_1_forward_selection(
    candidate_features,
    max_features=20,  # Максимальное число признаков
    min_improvement=0.01  # Минимальное улучшение для добавления
)
```

---

### Метод 2: Backward Elimination

**Принцип:** Начинает со всех признаков и последовательно удаляет наименее важные.

**Преимущества:**
- Учитывает взаимодействия между признаками
- Может найти лучшие комбинации, чем forward selection

**Недостатки:**
- Медленнее для большого числа признаков
- Может удалить важные признаки на ранних этапах

**Использование:**
```python
features, metrics = selector.method_2_backward_elimination(
    candidate_features,
    min_features=5,  # Минимальное число признаков
    min_improvement=0.01
)
```

---

### Метод 3: Фильтрация по положительным loadings PC1

**Принцип:** Использует только признаки с положительными loadings в PC1, что обеспечивает положительную корреляцию с патологией.

**Преимущества:**
- Быстрый и простой
- Гарантирует биологическую интерпретацию (положительная корреляция с патологией)
- Уже реализован в `aggregate.select_feature_columns()`

**Недостатки:**
- Может исключить важные признаки с отрицательными loadings
- Не оптимизирует разделение между группами

**Использование:**
```python
features, metrics = selector.method_3_positive_loadings_filter(
    candidate_features,
    min_loading=0.05  # Минимальный loading для включения
)
```

---

### Метод 4: Mutual Information (Взаимная информация)

**Принцип:** Использует взаимную информацию между признаками и метками классов для отбора наиболее информативных признаков.

**Преимущества:**
- Не требует предположений о распределении данных
- Учитывает нелинейные зависимости
- Хорошо работает для категориальных и непрерывных признаков

**Недостатки:**
- Может выбрать признаки, которые не оптимальны для PCA
- Не учитывает корреляции между признаками

**Использование:**
```python
features, metrics = selector.method_4_mutual_information(
    candidate_features,
    k=None  # None = автоматический выбор по медиане
)
```

---

### Метод 5: LASSO (L1-regularization)

**Принцип:** Использует LASSO для автоматического отбора признаков через регуляризацию. Признаки с нулевыми коэффициентами исключаются.

**Преимущества:**
- Автоматический отбор признаков
- Учитывает мультиколлинеарность
- Хорошо работает для большого числа признаков

**Недостатки:**
- Может выбрать слишком мало признаков
- Требует настройки параметра регуляризации

**Использование:**
```python
features, metrics = selector.method_5_lasso_selection(
    candidate_features,
    cv=5  # Число фолдов для кросс-валидации
)
```

---

### Метод 6: RFE (Recursive Feature Elimination)

**Принцип:** Использует важность признаков из модели (Random Forest) для рекурсивного удаления наименее важных признаков.

**Преимущества:**
- Учитывает взаимодействия между признаками
- Может использовать RFECV для автоматического выбора числа признаков
- Хорошо работает для нелинейных зависимостей

**Недостатки:**
- Медленнее других методов
- Зависит от выбора базовой модели

**Использование:**
```python
features, metrics = selector.method_6_rfe_selection(
    candidate_features,
    n_features=None  # None = автоматический выбор через RFECV
)
```

---

### Метод 7: Перебор комбинаций (Brute Force)

**Принцип:** Перебирает все возможные комбинации признаков и выбирает лучшую.

**Преимущества:**
- Находит глобальный оптимум (для малого числа признаков)
- Учитывает все возможные взаимодействия

**Недостатки:**
- Экспоненциальная сложность
- Практически неприменим для большого числа признаков

**Использование:**
```python
features, metrics = selector.method_7_brute_force_combinations(
    candidate_features,
    max_features=5,  # Максимальное число признаков в комбинации
    max_combinations=1000  # Максимальное число комбинаций для проверки
)
```

---

## Метрики качества

Все методы оцениваются по следующим метрикам:

1. **score** - Комплексная оценка:
   - 40% - разделение между группами (separation)
   - 30% - позиция mod образцов на нормализованной шкале (ближе к 1)
   - 30% - объясненная дисперсия PC1

2. **separation** - Разница между средними PC1 для mod и normal образцов
   - Чем больше, тем лучше разделение

3. **mean_pc1_norm_mod** - Среднее нормализованное PC1 для mod образцов
   - Цель: близко к 1.0

4. **explained_variance** - Доля объясненной дисперсии PC1
   - Чем больше, тем лучше

---

## Полный анализ и сравнение методов

### Запуск через командную строку:

```bash
python -m scale.feature_selection_automated results/predictions experiments/feature_selection
```

### Использование в коде:

```python
from scale.feature_selection_automated import run_feature_selection_analysis

results_df = run_feature_selection_analysis(
    predictions_dir="results/predictions",
    output_dir="experiments/feature_selection",
    methods=['forward', 'backward', 'positive_loadings', 'mutual_information', 'lasso', 'rfe']
)

# Результаты отсортированы по score (лучший первый)
best_method = results_df.iloc[0]
print(f"Лучший метод: {best_method['method']}")
print(f"Отобранные признаки: {best_method['features']}")
```

### Сравнение всех методов:

```python
from scale.feature_selection_automated import FeatureSelector, aggregate

df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_all = aggregate.select_all_feature_columns(df_features)

selector = FeatureSelector(df_all)
candidate_features = [c for c in df_all.columns if c != 'image']

results_df = selector.compare_all_methods(candidate_features)

# Результаты отсортированы по score
print(results_df[['method', 'n_features', 'score', 'separation', 'mean_pc1_norm_mod']])
```

---

## Рекомендации по выбору метода

### Для быстрого анализа:
- **Forward Selection** или **Positive Loadings Filter**

### Для точного подбора:
- **Backward Elimination** или **RFE**

### Для большого числа признаков:
- **LASSO** или **Mutual Information**

### Для малого числа признаков (< 10):
- **Brute Force** (перебор комбинаций)

### Комбинированный подход:
1. Использовать **Positive Loadings Filter** для предварительной фильтрации
2. Применить **Forward Selection** или **Backward Elimination** на отфильтрованных признаках
3. Сравнить результаты с другими методами

---

## Интерпретация результатов

### Хороший набор признаков должен:
1. ✅ **Разделять группы**: `separation > 2.0`
2. ✅ **Позиционировать mod образцы высоко**: `mean_pc1_norm_mod > 0.7`
3. ✅ **Объяснять дисперсию**: `explained_variance > 0.3`
4. ✅ **Иметь высокий score**: `score > 1.0`

### Пример хорошего результата:
```
method: forward
n_features: 12
score: 1.45
separation: 3.2
mean_pc1_norm_mod: 0.85
explained_variance: 0.42
```

---

## Сохранение и использование результатов

Результаты автоматически сохраняются в:
- `feature_selection_results_TIMESTAMP.csv` - таблица сравнения методов
- `best_features_TIMESTAMP.json` - лучший набор признаков с метриками

### Использование сохраненного набора признаков:

```python
import json
from scale import aggregate, pca_scoring

# Загрузка конфигурации
with open("experiments/feature_selection/best_features_TIMESTAMP.json") as f:
    config = json.load(f)

selected_features = config['selected_features']

# Применение к новым данным
df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)

# Использование отобранных признаков
df_selected = df_features[['image'] + selected_features]

# PCA анализ
pca_scorer = pca_scoring.PCAScorer()
df_pca = pca_scorer.fit_transform(df_selected)
```

---

## Примеры использования

### Пример 1: Быстрый подбор признаков

```python
from scale.feature_selection_automated import FeatureSelector, aggregate

# Загрузка данных
df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_all = aggregate.select_all_feature_columns(df_features)

# Создание селектора
selector = FeatureSelector(df_all)
candidate_features = [c for c in df_all.columns if c != 'image']

# Быстрый подбор через Forward Selection
features, metrics = selector.method_1_forward_selection(
    candidate_features,
    max_features=15
)

print(f"Отобрано признаков: {len(features)}")
print(f"Score: {metrics['score']:.4f}")
print(f"Разделение: {metrics['separation']:.4f}")
print(f"Mod образцы (норм.): {metrics['mean_pc1_norm_mod']:.4f}")
```

### Пример 2: Сравнение нескольких методов

```python
from scale.feature_selection_automated import FeatureSelector, aggregate

df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_all = aggregate.select_all_feature_columns(df_features)

selector = FeatureSelector(df_all)
candidate_features = [c for c in df_all.columns if c != 'image']

# Сравнение методов
results_df = selector.compare_all_methods(
    candidate_features,
    methods=['forward', 'backward', 'positive_loadings', 'lasso']
)

# Вывод результатов
print(results_df[['method', 'n_features', 'score', 'separation', 'mean_pc1_norm_mod']])
```

### Пример 3: Комбинированный подход

```python
from scale.feature_selection_automated import FeatureSelector, aggregate

df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_all = aggregate.select_all_feature_columns(df_features)

selector = FeatureSelector(df_all)
candidate_features = [c for c in df_all.columns if c != 'image']

# Шаг 1: Предварительная фильтрация по положительным loadings
filtered_features, _ = selector.method_3_positive_loadings_filter(
    candidate_features,
    min_loading=0.05
)

print(f"После фильтрации: {len(filtered_features)} признаков")

# Шаг 2: Точный подбор через Forward Selection
final_features, metrics = selector.method_1_forward_selection(
    filtered_features,
    max_features=20
)

print(f"Финальный набор: {len(final_features)} признаков")
print(f"Score: {metrics['score']:.4f}")
```

---

## Заключение

Автоматизированный подбор признаков позволяет найти оптимальный набор признаков для построения медицинской шкалы, где образцы с заболеваниями (mod) получают высокие значения PC1. Рекомендуется:

1. Начать с **Forward Selection** или **Positive Loadings Filter** для быстрого анализа
2. Сравнить результаты с другими методами через `compare_all_methods()`
3. Выбрать метод с наилучшим score и метриками
4. Сохранить и использовать отобранный набор признаков для дальнейшего анализа







