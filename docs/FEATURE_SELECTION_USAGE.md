# Использование выбора признаков с положительными loadings

## Обзор

По умолчанию система теперь использует только признаки с **положительными loadings** в PC1, что значительно улучшает качество шкалы патологии.

## Преимущества

- ✅ **Объясненная дисперсия PC1**: 30.39% → **70.16%** (более чем в 2 раза!)
- ✅ **Корректная классификация**: Патологические образцы лучше разделяются
- ✅ **Автоматический отбор**: Система сама выбирает оптимальные признаки

## Использование в коде

### По умолчанию (положительные loadings)

```python
from scale import aggregate

# Автоматически использует только положительные loadings
df_features = aggregate.select_feature_columns(df)
```

### Все признаки (старый подход)

```python
# Использовать все признаки без фильтрации
df_features = aggregate.select_feature_columns(df, use_positive_loadings=False)
```

### Настройка параметров

```python
# Изменить минимальный loading
df_features = aggregate.select_feature_columns(df, min_loading=0.1)

# Включить Paneth (не рекомендуется)
df_features = aggregate.select_feature_columns(df, exclude_paneth=False)
```

## Использование в дашборде

В Streamlit дашборде доступны настройки:

1. **Режим выбора признаков**:
   - "Только положительные loadings (рекомендуется)" - по умолчанию
   - "Все признаки" - старый подход

2. **Параметры** (при выборе положительных loadings):
   - Минимальный loading (по умолчанию 0.05)
   - Исключить Paneth признаки (по умолчанию включено)

## Рекомендуемый набор признаков

При использовании положительных loadings (min_loading=0.05, exclude_paneth=True) включаются:

- `Dysplasia_mean_relative_area`
- `Enterocytes_mean_relative_area`
- `Enterocytes_relative_area`
- `EoE_mean_relative_area`
- `Meta_mean_relative_area`
- `Meta_relative_area`
- `Mild_mean_relative_area`
- `Neutrophils_mean_relative_area`
- `Plasma Cells_mean_relative_area`
- `Plasma Cells_relative_area`
- `Plasma Cells_relative_count`

**Итого: 11 признаков** (вместо 30)

## Обратная совместимость

Старая функция `select_feature_columns()` теперь является оберткой:
- По умолчанию использует положительные loadings
- Можно переключить на все признаки через параметр `use_positive_loadings=False`

Для прямого доступа к старой функции используйте:
```python
df_features = aggregate.select_all_feature_columns(df)
```

## Когда использовать каждый режим

### Положительные loadings (рекомендуется)
- ✅ Для построения шкалы патологии
- ✅ Когда нужна максимальная объясненная дисперсия
- ✅ Для корректной классификации патологических образцов

### Все признаки
- ⚠️ Для экспериментов и сравнения
- ⚠️ Когда нужны все доступные признаки
- ⚠️ Для анализа влияния отдельных признаков

## Технические детали

### Как работает отбор

1. Сначала выбираются все доступные признаки через `select_all_feature_columns()`
2. Обучается PCA на полном наборе данных
3. Извлекаются loadings первой главной компоненты (PC1)
4. Фильтруются только признаки с `loading > min_loading`
5. При необходимости исключаются Paneth признаки

### Важно

- Отбор признаков требует обучения PCA на **полном наборе данных**
- Для новых данных используйте **те же признаки**, что были отобраны при обучении
- Не меняйте набор признаков между обучением и применением модели

## Примеры

### Базовое использование

```python
from scale import aggregate, pca_scoring, spectral_analysis

# Загрузка данных
df = aggregate.load_predictions_batch("predictions/")

# Создание относительных признаков
df_features = aggregate.create_relative_features(df)

# Отбор признаков (автоматически использует положительные loadings)
df_selected = aggregate.select_feature_columns(df_features)

# PCA анализ
pca_scorer = pca_scoring.PCAScorer()
df_pca = pca_scorer.fit_transform(df_selected)

# Спектральный анализ
analyzer = spectral_analysis.SpectralAnalyzer()
analyzer.fit_spectrum(df_pca, pc1_column="PC1")
df_result = analyzer.transform_to_spectrum(df_pca, pc1_column="PC1")
```

### Сравнение режимов

```python
# Режим 1: Положительные loadings
df_positive = aggregate.select_feature_columns(df_features, use_positive_loadings=True)
pca_positive = pca_scoring.PCAScorer()
df_pca_positive = pca_positive.fit_transform(df_positive)
print(f"Объясненная дисперсия: {pca_positive.pca.explained_variance_ratio_[0]:.2%}")

# Режим 2: Все признаки
df_all = aggregate.select_feature_columns(df_features, use_positive_loadings=False)
pca_all = pca_scoring.PCAScorer()
df_pca_all = pca_all.fit_transform(df_all)
print(f"Объясненная дисперсия: {pca_all.pca.explained_variance_ratio_[0]:.2%}")
```

## Дополнительная информация

- Подробный план подбора признаков: [docs/FEATURE_SELECTION_PLAN.md](FEATURE_SELECTION_PLAN.md)
- Анализ признаков: `python test/analyze_feature_selection.py`
- Тест положительных loadings: `python test/test_positive_loadings_features.py`








