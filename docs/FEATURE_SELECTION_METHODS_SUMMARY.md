# Краткая сводка методов автоматизированного подбора признаков

## Цель

Найти оптимальный набор признаков для построения медицинской шкалы на базе PCA, где образцы с заболеваниями (mod) получают **высокие значения PC1** (ближе к 1 на нормализованной шкале).

---

## Доступные методы

| Метод | Скорость | Точность | Рекомендация |
|-------|----------|----------|--------------|
| **1. Forward Selection** | ⚡⚡⚡ Быстрый | ⭐⭐⭐ Хорошая | ✅ Рекомендуется для начала |
| **2. Backward Elimination** | ⚡⚡ Средний | ⭐⭐⭐⭐ Отличная | ✅ Для точного подбора |
| **3. Positive Loadings Filter** | ⚡⚡⚡ Очень быстрый | ⭐⭐ Базовая | ✅ Быстрая предварительная фильтрация |
| **4. Mutual Information** | ⚡⚡ Средний | ⭐⭐⭐ Хорошая | ✅ Для нелинейных зависимостей |
| **5. LASSO** | ⚡⚡ Средний | ⭐⭐⭐ Хорошая | ✅ Для большого числа признаков |
| **6. RFE** | ⚡ Медленный | ⭐⭐⭐⭐ Отличная | ✅ Для точного подбора |
| **7. Brute Force** | ⚡⚡⚡⚡ Очень медленный | ⭐⭐⭐⭐⭐ Идеальная | ⚠️ Только для малого числа признаков |

---

## Быстрый старт

### Вариант 1: Один метод (рекомендуется для начала)

```python
from scale.feature_selection_automated import FeatureSelector, aggregate

# Загрузка данных
df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_all = aggregate.select_all_feature_columns(df_features)

# Создание селектора
selector = FeatureSelector(df_all)
candidate_features = [c for c in df_all.columns if c != 'image']

# Подбор признаков через Forward Selection
features, metrics = selector.method_1_forward_selection(
    candidate_features,
    max_features=20
)

print(f"Отобрано признаков: {len(features)}")
print(f"Score: {metrics['score']:.4f}")
print(f"Разделение mod/normal: {metrics['separation']:.4f}")
print(f"Mod образцы (норм.): {metrics['mean_pc1_norm_mod']:.4f}")
```

### Вариант 2: Сравнение всех методов (рекомендуется для точного подбора)

```python
from scale.feature_selection_automated import run_feature_selection_analysis

results_df = run_feature_selection_analysis(
    predictions_dir="results/predictions",
    output_dir="experiments/feature_selection",
    methods=['forward', 'backward', 'positive_loadings', 'mutual_information', 'lasso', 'rfe']
)

# Результаты автоматически сохраняются в output_dir
# Лучший метод - первый в отсортированной таблице
best_method = results_df.iloc[0]
print(f"Лучший метод: {best_method['method']}")
print(f"Отобранные признаки: {best_method['features']}")
```

### Вариант 3: Комбинированный подход

```python
from scale.feature_selection_automated import FeatureSelector, aggregate

df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_all = aggregate.select_all_feature_columns(df_features)

selector = FeatureSelector(df_all)
candidate_features = [c for c in df_all.columns if c != 'image']

# Шаг 1: Предварительная фильтрация
filtered_features, _ = selector.method_3_positive_loadings_filter(
    candidate_features,
    min_loading=0.05
)

# Шаг 2: Точный подбор
final_features, metrics = selector.method_1_forward_selection(
    filtered_features,
    max_features=20
)
```

---

## Метрики качества

Все методы оцениваются по комплексной метрике **score**, которая учитывает:

- **40%** - Разделение между группами (separation)
- **30%** - Позиция mod образцов на нормализованной шкале (ближе к 1)
- **30%** - Объясненная дисперсия PC1

### Хорошие результаты:
- ✅ `separation > 2.0` - хорошее разделение между группами
- ✅ `mean_pc1_norm_mod > 0.7` - mod образцы позиционируются высоко
- ✅ `explained_variance > 0.3` - PC1 объясняет значительную часть дисперсии
- ✅ `score > 1.0` - хорошая комплексная оценка

---

## Рекомендации

### Для быстрого анализа:
1. Используйте **Forward Selection** или **Positive Loadings Filter**

### Для точного подбора:
1. Используйте **compare_all_methods()** для сравнения всех методов
2. Выберите метод с наилучшим score
3. Проверьте метрики разделения и позиционирования mod образцов

### Для большого числа признаков (> 30):
1. Начните с **LASSO** или **Mutual Information**
2. Затем примените **Forward Selection** на отфильтрованных признаках

### Для малого числа признаков (< 10):
1. Используйте **Brute Force** для перебора всех комбинаций
2. Или **Backward Elimination** для точного подбора

---

## Примеры использования

### Пример 1: Быстрый подбор (Forward Selection)

```python
features, metrics = selector.method_1_forward_selection(
    candidate_features,
    max_features=15,
    min_improvement=0.01
)
```

### Пример 2: Точный подбор (Backward Elimination)

```python
features, metrics = selector.method_2_backward_elimination(
    candidate_features,
    min_features=5,
    min_improvement=0.01
)
```

### Пример 3: Фильтрация по положительным loadings

```python
features, metrics = selector.method_3_positive_loadings_filter(
    candidate_features,
    min_loading=0.05
)
```

### Пример 4: Mutual Information

```python
features, metrics = selector.method_4_mutual_information(
    candidate_features,
    k=None  # Автоматический выбор
)
```

### Пример 5: LASSO

```python
features, metrics = selector.method_5_lasso_selection(
    candidate_features,
    cv=5
)
```

### Пример 6: RFE

```python
features, metrics = selector.method_6_rfe_selection(
    candidate_features,
    n_features=None  # Автоматический выбор через RFECV
)
```

---

## Сохранение и использование результатов

Результаты автоматически сохраняются в:
- `feature_selection_results_TIMESTAMP.csv` - таблица сравнения методов
- `best_features_TIMESTAMP.json` - лучший набор признаков с метриками

### Использование сохраненного набора:

```python
import json
from scale import aggregate, pca_scoring

# Загрузка конфигурации
with open("experiments/feature_selection/best_features_TIMESTAMP.json") as f:
    config = json.load(f)

selected_features = config['selected_features']

# Применение к данным
df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_selected = df_features[['image'] + selected_features]

# PCA анализ
pca_scorer = pca_scoring.PCAScorer()
df_pca = pca_scorer.fit_transform(df_selected)
```

---

## Дополнительная информация

Подробное описание методов и примеры использования см. в:
- `docs/FEATURE_SELECTION_AUTOMATED.md` - полная документация
- `scale/feature_selection_automated.py` - исходный код


