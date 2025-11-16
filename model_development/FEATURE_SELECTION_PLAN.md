# План поиска лучших признаков для медицинской шкалы

## Текущее состояние

### Проведенные эксперименты:

1. **feature_selection_quick** (forward selection)
   - Score: 3.078
   - Separation: 6.790
   - Mod (норм. PC1): 0.680
   - Объясненная дисперсия: 0.527
   - Признаков: 15 (включая Paneth)

2. **feature_selection_no_paneth** (forward selection без Paneth)
   - Score: 2.936
   - Separation: 6.458
   - Mod (норм. PC1): 0.663
   - Объясненная дисперсия: 0.513
   - Признаков: 15 (без Paneth)

### Доступные методы подбора признаков:

1. **forward** - Forward Selection (последовательное добавление признаков)
2. **backward** - Backward Elimination (последовательное удаление признаков)
3. **positive_loadings** - Фильтрация по положительным loadings PC1
4. **mutual_information** - Mutual Information (взаимная информация)
5. **lasso** - Lasso regularization (L1 регуляризация)
6. **rfe** - Recursive Feature Elimination (рекурсивное удаление признаков)

## План дальнейших экспериментов

### Этап 1: Базовое сравнение всех методов (приоритет: ВЫСОКИЙ)

**Цель:** Сравнить все доступные методы на одних и тех же данных

**Эксперименты:**
- `feature_selection_all_methods` - сравнение всех 6 методов
- `feature_selection_all_methods_no_paneth` - сравнение всех методов без Paneth

**Ожидаемый результат:** Определение лучшего базового метода

**Команда:**
```bash
python3 -m model_development.feature_selection_automated \
    results/predictions \
    experiments/feature_selection_all_methods
```

### Этап 2: Вариации параметров для лучших методов (приоритет: ВЫСОКИЙ)

**Цель:** Найти оптимальные параметры для методов, показавших лучшие результаты

#### 2.1. Forward Selection - вариации параметров:
- `feature_selection_forward_min_improvement_001` - min_improvement=0.01 (более строгий отбор)
- `feature_selection_forward_min_improvement_005` - min_improvement=0.05 (менее строгий отбор)
- `feature_selection_forward_min_improvement_0001` - min_improvement=0.001 (более мягкий отбор)

#### 2.2. Positive Loadings - вариации порога:
- `feature_selection_positive_loadings_001` - min_loading=0.01 (больше признаков)
- `feature_selection_positive_loadings_005` - min_loading=0.05 (текущий)
- `feature_selection_positive_loadings_01` - min_loading=0.1 (меньше признаков, только важные)
- `feature_selection_positive_loadings_02` - min_loading=0.2 (только самые важные)

#### 2.3. Mutual Information - вариации числа признаков:
- `feature_selection_mutual_info_k5` - k=5 признаков
- `feature_selection_mutual_info_k10` - k=10 признаков
- `feature_selection_mutual_info_k15` - k=15 признаков
- `feature_selection_mutual_info_k20` - k=20 признаков
- `feature_selection_mutual_info_auto` - автоматический выбор k

#### 2.4. Lasso - вариации альфа:
- `feature_selection_lasso_alpha_001` - alpha=0.01 (меньше регуляризации)
- `feature_selection_lasso_alpha_01` - alpha=0.1 (текущий)
- `feature_selection_lasso_alpha_1` - alpha=1.0 (больше регуляризации)

#### 2.5. RFE - вариации числа признаков:
- `feature_selection_rfe_n5` - n_features=5
- `feature_selection_rfe_n10` - n_features=10
- `feature_selection_rfe_n15` - n_features=15
- `feature_selection_rfe_n20` - n_features=20

### Этап 3: Комбинированные методы (приоритет: СРЕДНИЙ)

**Цель:** Комбинировать лучшие методы для улучшения результатов

#### 3.1. Двухэтапный отбор:
- `feature_selection_positive_then_forward` - сначала positive_loadings, затем forward selection
- `feature_selection_mutual_then_forward` - сначала mutual_info, затем forward selection
- `feature_selection_lasso_then_forward` - сначала lasso, затем forward selection

#### 3.2. Исключение проблемных признаков:
- `feature_selection_no_paneth_no_surface` - без Paneth и Surface epithelium
- `feature_selection_no_paneth_no_muscularis` - без Paneth и Muscularis mucosae
- `feature_selection_only_pathology` - только патологические признаки (без структурных)

### Этап 4: Анализ типов признаков (приоритет: СРЕДНИЙ)

**Цель:** Понять, какие типы признаков наиболее важны

#### 4.1. По типу признака:
- `feature_selection_only_relative_count` - только relative_count признаки
- `feature_selection_only_relative_area` - только relative_area признаки
- `feature_selection_only_mean_relative_area` - только mean_relative_area признаки
- `feature_selection_count_and_area` - relative_count + relative_area (без mean)

#### 4.2. По классам патологии:
- `feature_selection_mild_dysplasia_moderate` - только Mild, Dysplasia, Moderate
- `feature_selection_inflammatory` - только воспалительные (Neutrophils, EoE, Plasma Cells)
- `feature_selection_structural_changes` - только структурные изменения (Meta, Enterocytes)

### Этап 5: Оптимизация по метрикам (приоритет: НИЗКИЙ)

**Цель:** Найти набор признаков, оптимизированный под конкретные метрики

#### 5.1. Оптимизация по Separation:
- `feature_selection_max_separation` - максимизация separation
- `feature_selection_separation_threshold_7` - separation > 7.0

#### 5.2. Оптимизация по Mod позиции:
- `feature_selection_mod_position_08` - mod (норм. PC1) > 0.8
- `feature_selection_mod_position_085` - mod (норм. PC1) > 0.85

#### 5.3. Оптимизация по объясненной дисперсии:
- `feature_selection_explained_variance_06` - explained_variance > 0.6

## Рекомендуемый порядок выполнения

### Фаза 1: Быстрый скрининг (1-2 дня)
1. ✅ Этап 1: Базовое сравнение всех методов
2. ✅ Этап 2.1: Вариации Forward Selection (самый перспективный метод)

### Фаза 2: Детальная оптимизация (3-5 дней)
3. ✅ Этап 2.2-2.5: Вариации параметров для лучших методов
4. ✅ Этап 3: Комбинированные методы

### Фаза 3: Углубленный анализ (опционально)
5. ✅ Этап 4: Анализ типов признаков
6. ✅ Этап 5: Оптимизация по метрикам

## Критерии оценки результатов

### Основные метрики (приоритет):
1. **Score** (комплексная оценка) - главный критерий
   - Хорошие значения: > 3.0
   - Отличные значения: > 3.5

2. **Separation** (разделение групп)
   - Хорошие значения: > 6.0
   - Отличные значения: > 7.0

3. **Mod (норм. PC1)** (позиция патологических образцов)
   - Хорошие значения: > 0.70
   - Отличные значения: > 0.85

4. **Объясненная дисперсия**
   - Хорошие значения: > 0.50
   - Отличные значения: > 0.60

### Дополнительные критерии:
- **Число признаков** - меньше признаков лучше (проще интерпретация)
- **Стабильность** - результаты должны быть воспроизводимыми
- **Интерпретируемость** - признаки должны иметь биологический смысл

## Скрипты для автоматизации

См. `run_comprehensive_feature_selection.py` для автоматического запуска всех экспериментов.

