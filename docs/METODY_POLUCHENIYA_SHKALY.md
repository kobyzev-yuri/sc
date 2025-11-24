# Методы получения шкалы патологии

## Обзор

Система использует несколько методов для построения шкалы оценки патологии от 0 (норма) до 1 (максимальная патология):

1. **PCA (Principal Component Analysis)** - основной метод снижения размерности
2. **Спектральный анализ** - расширенный метод с выявлением мод распределения
3. **Отбор признаков** - Forward Selection и Mutual Information для выбора оптимальных признаков

---

## PCA метод

### Принцип работы

PCA находит главные компоненты - направления максимальной вариации в данных. Первая главная компонента (PC1) используется как основа для шкалы патологии.

### Алгоритм

1. **Стандартизация данных:**
   ```
   X_scaled = (X - μ) / σ
   ```
   Приводит все признаки к одному масштабу.

2. **Применение PCA:**
   - Находит направление максимальной вариации (PC1)
   - Вычисляет loadings (веса признаков)

3. **Вычисление PC1 для WSI:**
   ```
   PC1(wsi) = loading₁ × X_scaled[1] + loading₂ × X_scaled[2] + ... + loadingₙ × X_scaled[n]
   ```

4. **Нормализация в шкалу 0-1:**
   ```
   PC1_norm(wsi) = (PC1(wsi) - PC1_min) / (PC1_max - PC1_min)
   ```

### Важность признаков (Loadings)

Loadings показывают вклад каждого признака в PC1:
- **Большое абсолютное значение** → признак сильно влияет на PC1
- **Положительный loading** → увеличение признака увеличивает патологию
- **Отрицательный loading** → увеличение признака уменьшает патологию

### Реализация

```python
from scale import pca_scoring

# Обучение
scorer = pca_scoring.PCAScorer()
scorer.fit(df_features, feature_columns)

# Применение
df_results = scorer.transform(df_features)
# df_results содержит колонку 'PC1_norm' со шкалой 0-1
```

---

## Метрики качества

### Score (комплексная метрика)

Формула:
```
score = 0.4 × separation + 0.3 × mean_pc1_norm_mod + 0.3 × explained_variance
```

**Компоненты:**

1. **Separation (разделение классов):**
   ```
   separation = mean_pc1_mod - mean_pc1_normal
   ```
   - Показывает, насколько хорошо разделяются патологические и нормальные образцы
   - Целевое значение: > 6.0 (хорошо), > 7.0 (отлично)

2. **mean_pc1_norm_mod (позиция патологических образцов):**
   ```
   mean_pc1_norm_mod = mean((pc1_mod - pc1_min) / (pc1_max - pc1_min))
   ```
   - Показывает, где находятся патологические образцы на шкале 0-1
   - Целевое значение: > 0.70 (хорошо), > 0.85 (отлично)

3. **explained_variance (объясненная дисперсия):**
   ```
   explained_variance = λ₁ / Σλᵢ
   ```
   - Показывает, какая доля вариации объясняется PC1
   - Целевое значение: > 0.30 (хорошо), > 0.50 (отлично)

**Диапазоны Score:**
- Плохо: score < 2.0
- Удовлетворительно: 2.0 ≤ score < 3.0
- Хорошо: 3.0 ≤ score < 3.5
- Отлично: score ≥ 3.5

---

## Отбор признаков

### Forward Selection

**Принцип:** Жадный алгоритм, который последовательно добавляет признаки, максимизирующие Score.

**Алгоритм:**
1. Начинает с пустого набора признаков
2. На каждом шаге пробует каждый оставшийся признак
3. Выбирает признак, который дает наибольшее улучшение Score
4. Останавливается, когда улучшение < min_improvement или достигнуто max_features

**Преимущества:**
- ✅ Учитывает взаимодействие признаков
- ✅ Оптимизирует целевую метрику (Score)
- ✅ Учитывает специфику PCA

**Недостатки:**
- ❌ Вычислительно затратный
- ❌ Может застрять в локальном оптимуме

### Mutual Information

**Принцип:** Измеряет количество информации, которую признак содержит о классе (патология vs норма).

**Формула:**
```
MI(X, Y) = Σ Σ P(x, y) × log(P(x, y) / (P(x) × P(y)))
```

**Алгоритм:**
1. Вычисляет MI для всех признаков
2. Сортирует по убыванию MI
3. Выбирает топ-k признаков

**Преимущества:**
- ✅ Быстрый (один проход)
- ✅ Не зависит от модели
- ✅ Учитывает нелинейные зависимости

**Недостатки:**
- ❌ Не учитывает взаимодействие признаков
- ❌ Не оптимизирует целевую метрику

### Рекомендации

**Комбинированный подход:**
1. **Этап 1:** Mutual Information → отфильтровать до топ-20-25 признаков
2. **Этап 2:** Forward Selection → выбрать финальные 15-20 признаков

Это сочетает скорость MI и точность Forward Selection!

---

## Признаки

### Абсолютные признаки

Исходные количественные характеристики:
- `{class}_count` - количество объектов класса
- `{class}_area` - общая площадь объектов класса

**22 признака:** 11 классов × 2 типа (count, area)

### Относительные признаки

Нормализованные значения относительно Crypts:
- `{class}_relative_count = {class}_count / Crypts_count`
- `{class}_relative_area = {class}_area / Crypts_area`
- `{class}_mean_relative_area = {class}_relative_area / {class}_count`

**30 признаков:** 10 классов патологий × 3 типа признаков

### Рекомендация

Используйте **относительные признаки** - они более устойчивы к вариациям размера образцов и лучше разделяют патологические и нормальные образцы.

---

## Спектральный анализ

Расширенный метод, который:
- Выявляет моды (пики) в распределении PC1
- Классифицирует образцы по модам (normal/mild/moderate/severe)
- Создает более интерпретируемую шкалу

**Реализация:**
```python
from scale import spectral_analysis

analyzer = spectral_analysis.SpectralAnalyzer()
analyzer.fit_pca(df_features)
df_pca = analyzer.transform_pca(df_features)
analyzer.fit_spectrum(df_pca, percentile_low=1.0, percentile_high=99.0)
df_results = analyzer.transform_spectrum(df_pca)
```

---

## Примеры использования

### Пример 1: Базовый PCA анализ

```python
from scale import aggregate, pca_scoring
import pandas as pd

# Загрузка данных
df = aggregate.load_predictions_batch("results/predictions")
df_features = aggregate.create_relative_features(df)
df_features = aggregate.select_feature_columns(df_features)

# Обучение PCA
scorer = pca_scoring.PCAScorer()
scorer.fit(df_features, df_features.columns.tolist())

# Применение
df_results = scorer.transform(df_features)
print(df_results[['image', 'PC1_norm']].head())
```

### Пример 2: Отбор признаков с Forward Selection

```python
from model_development.feature_selection_automated import FeatureSelector

selector = FeatureSelector(df_features, mod_samples, normal_samples)
selected_features, metrics = selector.method_1_forward_selection(
    candidate_features=df_features.columns.tolist(),
    max_features=20,
    min_improvement=0.01
)

print(f"Выбрано признаков: {len(selected_features)}")
print(f"Score: {metrics['score']:.4f}")
print(f"Separation: {metrics['separation']:.4f}")
```

---

## Рекомендации

1. **Начните с относительных признаков** - они более устойчивы
2. **Используйте Forward Selection** для финального отбора признаков
3. **Оценивайте качество по Score** - комплексная метрика учитывает все аспекты
4. **Сохраняйте обученные модели** для инференса новых данных
5. **Документируйте выбранные признаки** для воспроизводимости

