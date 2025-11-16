# Объяснение методов отбора признаков: Forward Selection и Mutual Information

## Forward Selection (Последовательный отбор признаков)

### Принцип работы

Forward Selection - это **жадный алгоритм**, который начинает с пустого набора признаков и последовательно добавляет признаки, которые максимизируют целевую метрику.

### Алгоритм (пошагово)

```
1. Начальное состояние:
   - selected = [] (пустой список)
   - remaining = [все кандидатные признаки]
   - best_score = -∞

2. Для каждого шага:
   a. Для каждого признака в remaining:
      - Добавить признак к selected
      - Вычислить метрики (score, separation, mod_norm, etc.)
      - Если score лучше текущего best_score, запомнить этот признак
   
   b. Выбрать признак с лучшим score
   
   c. Проверить условие остановки:
      - Если улучшение score < min_improvement → остановиться
      - Если достигнуто max_features → остановиться
   
   d. Добавить лучший признак к selected
   e. Удалить его из remaining

3. Вернуть selected и метрики
```

### Код реализации

```python
def method_1_forward_selection(
    self,
    candidate_features: List[str],
    max_features: Optional[int] = None,
    min_improvement: float = 0.01,
) -> Tuple[List[str], Dict[str, float]]:
    selected = []
    remaining = candidate_features.copy()
    best_score = -np.inf
    
    while len(selected) < max_features and remaining:
        best_feature = None
        best_new_score = best_score
        
        # Пробуем каждый оставшийся признак
        for feature in remaining:
            test_features = selected + [feature]
            metrics = evaluate_feature_set(
                self.df, test_features, 
                self.mod_samples, self.normal_samples
            )
            
            if metrics['score'] > best_new_score:
                best_new_score = metrics['score']
                best_feature = feature
        
        # Проверяем условие остановки
        if best_feature is None or (best_new_score - best_score) < min_improvement:
            break
        
        # Добавляем лучший признак
        selected.append(best_feature)
        remaining.remove(best_feature)
        best_score = best_new_score
    
    return selected, best_metrics
```

### Как вычисляется метрика для каждого признака?

Для каждого кандидатного признака вызывается `evaluate_feature_set()`, которая:

1. **Создает PCA модель** на текущем наборе признаков (selected + новый признак)
2. **Вычисляет PC1** для всех образцов
3. **Нормализует PC1** в диапазон [0, 1]
4. **Вычисляет метрики:**
   - `separation` = mean(PC1_mod) - mean(PC1_normal)
   - `mean_pc1_norm_mod` = среднее нормализованное PC1 для патологических образцов
   - `explained_variance` = доля дисперсии, объясняемая PC1
   - `score` = комбинация всех метрик

### Пример работы Forward Selection

**Шаг 1:** Пробуем все 32 признака по отдельности
- `Mild_relative_count` → score = 2.5
- `Neutrophils_relative_count` → score = 2.8
- `Moderate_relative_count` → score = 2.3
- ...
- **Лучший:** `Neutrophils_relative_count` (score = 2.8)
- **Добавляем:** `Neutrophils_relative_count`

**Шаг 2:** Пробуем оставшиеся 31 признак вместе с `Neutrophils_relative_count`
- `[Neutrophils_relative_count, Mild_relative_count]` → score = 3.1
- `[Neutrophils_relative_count, Moderate_relative_count]` → score = 2.9
- ...
- **Лучший:** `Mild_relative_count` (score = 3.1)
- **Добавляем:** `Mild_relative_count`

**Шаг 3:** Пробуем оставшиеся 30 признаков вместе с уже выбранными
- ...
- И так далее, пока улучшение score < min_improvement

### Преимущества Forward Selection

✅ **Учитывает взаимодействие признаков** - каждый признак оценивается в контексте уже выбранных  
✅ **Оптимизирует целевую метрику** - выбирает признаки, которые максимизируют score  
✅ **Учитывает специфику задачи** - метрика учитывает разделение патологических и нормальных образцов  

### Недостатки Forward Selection

❌ **Жадный алгоритм** - может застрять в локальном оптимуме  
❌ **Вычислительно затратный** - на каждом шаге нужно пересчитывать PCA для всех кандидатов  
❌ **Порядок важен** - первый выбранный признак влияет на все последующие  

---

## Mutual Information (Взаимная информация)

### Принцип работы

Mutual Information измеряет **количество информации**, которую один признак содержит о классе (патология vs норма). Это статистическая мера зависимости между признаком и меткой класса.

### Математическая формула

```
MI(X, Y) = Σ Σ P(x, y) * log(P(x, y) / (P(x) * P(y)))
```

где:
- X - признак
- Y - метка класса (mod/normal)
- P(x, y) - совместная вероятность
- P(x), P(y) - маргинальные вероятности

### Алгоритм (пошагово)

```
1. Создать метки классов:
   y = [1 если mod, 0 если normal] для каждого образца

2. Для каждого признака:
   - Вычислить Mutual Information с метками классов
   - Получить MI score

3. Отсортировать признаки по MI score (от большего к меньшему)

4. Выбрать топ-k признаков (или все с MI > порога)

5. Вычислить метрики на выбранных признаках
```

### Код реализации

```python
def method_4_mutual_information(
    self,
    candidate_features: List[str],
    k: Optional[int] = None,
) -> Tuple[List[str], Dict[str, float]]:
    # Создаем метки классов
    y = (self.df['sample_type'] == 'mod').astype(int).values
    
    # Вычисляем mutual information для всех признаков
    X = self.df[candidate_features].fillna(0).values
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Сортируем по MI score
    mi_df = pd.DataFrame({
        'feature': candidate_features,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Выбираем топ-k признаков
    if k is None:
        # Автоматически: признаки с MI > медианы
        threshold = mi_df['mi_score'].median()
        selected_features = mi_df[mi_df['mi_score'] > threshold]['feature'].tolist()
    else:
        selected_features = mi_df.head(k)['feature'].tolist()
    
    # Вычисляем метрики на выбранных признаках
    metrics = evaluate_feature_set(
        self.df, selected_features, 
        self.mod_samples, self.normal_samples
    )
    
    return selected_features, metrics
```

### Как работает mutual_info_classif?

Функция из sklearn:
1. **Дискретизирует** непрерывные признаки (если нужно)
2. **Вычисляет энтропию** H(X) и H(Y)
3. **Вычисляет совместную энтропию** H(X, Y)
4. **Вычисляет MI** = H(X) + H(Y) - H(X, Y)

### Пример работы Mutual Information

**Шаг 1:** Вычисляем MI для всех признаков
```
Mild_relative_count:      MI = 0.45
Neutrophils_relative_count: MI = 0.52
Moderate_relative_count:    MI = 0.38
Paneth_relative_count:      MI = 0.31
...
```

**Шаг 2:** Сортируем по убыванию MI
```
1. Neutrophils_relative_count: 0.52
2. Mild_relative_count:         0.45
3. Moderate_relative_count:      0.38
4. Paneth_relative_count:       0.31
...
```

**Шаг 3:** Выбираем топ-k (например, k=16)
- Берем первые 16 признаков с наибольшим MI

**Шаг 4:** Вычисляем метрики на этих 16 признаках
- Создаем PCA модель
- Вычисляем score, separation, mod_norm, etc.

### Преимущества Mutual Information

✅ **Быстрый** - вычисляется один раз для всех признаков  
✅ **Не зависит от модели** - не требует обучения PCA/модели  
✅ **Учитывает нелинейные зависимости** - может обнаружить сложные связи  
✅ **Независим от масштаба** - не требует нормализации  

### Недостатки Mutual Information

❌ **Не учитывает взаимодействие признаков** - каждый признак оценивается независимо  
❌ **Не оптимизирует целевую метрику** - выбирает признаки с высокой MI, но не обязательно лучшие для PCA  
❌ **Может выбрать избыточные признаки** - если два признака сильно коррелируют, оба могут иметь высокий MI  

---

## Сравнение методов

### Forward Selection vs Mutual Information

| Критерий | Forward Selection | Mutual Information |
|----------|------------------|-------------------|
| **Подход** | Жадный алгоритм | Статистический фильтр |
| **Оценка признаков** | В контексте уже выбранных | Независимо для каждого |
| **Оптимизация** | Целевая метрика (score) | Mutual Information |
| **Скорость** | Медленнее (пересчет PCA на каждом шаге) | Быстрее (один проход) |
| **Учет взаимодействий** | ✅ Да | ❌ Нет |
| **Риск переобучения** | Средний | Низкий |
| **Результаты (Фаза 1)** | Score: 3.2644 (24 признака) | Score: 2.9617 (16 признаков) |

### Почему Forward Selection показал лучший результат?

1. **Учитывает взаимодействие признаков:**
   - Forward Selection пробует каждый признак вместе с уже выбранными
   - Это позволяет найти комбинации признаков, которые работают лучше вместе

2. **Оптимизирует целевую метрику:**
   - Forward Selection максимизирует score, который учитывает:
     - Separation (разделение классов)
     - Mod позицию (положение патологических образцов)
     - Объясненную дисперсию
   - Mutual Information максимизирует только MI, которая может не коррелировать с целевой метрикой

3. **Учитывает специфику PCA:**
   - Forward Selection пересчитывает PCA на каждом шаге
   - Это позволяет видеть, как добавление признака влияет на PC1
   - Mutual Information не учитывает, как признаки будут работать в PCA

### Пример: Почему Forward лучше?

**Сценарий:** Есть два признака A и B

**Mutual Information:**
- MI(A, class) = 0.5
- MI(B, class) = 0.5
- Оба признака имеют одинаковый MI
- Выбирает оба, но не знает, что они могут быть избыточными

**Forward Selection:**
- Шаг 1: Пробует A → score = 2.5
- Шаг 1: Пробует B → score = 2.5
- Выбирает A (первый)
- Шаг 2: Пробует [A, B] → score = 2.6 (небольшое улучшение)
- Если улучшение < min_improvement, не добавляет B
- Результат: только A (избегает избыточности)

---

## Рекомендации по использованию

### Когда использовать Forward Selection?

✅ Когда важна **точность** и можно потратить время на вычисления  
✅ Когда признаки могут **взаимодействовать** друг с другом  
✅ Когда нужно **оптимизировать конкретную метрику** (score, separation, etc.)  

### Когда использовать Mutual Information?

✅ Когда нужна **быстрая** предварительная фильтрация признаков  
✅ Когда есть **много признаков** и нужно быстро отсеять неважные  
✅ Когда признаки **независимы** друг от друга  
✅ Когда можно использовать как **первый этап**, затем Forward Selection  

### Комбинированный подход

**Двухэтапный отбор:**
1. **Этап 1:** Mutual Information - отфильтровать до топ-20 признаков
2. **Этап 2:** Forward Selection - выбрать финальные 15-20 признаков из отфильтрованных

Это сочетает скорость MI и точность Forward Selection!

