# Тайминг в WSIPredictor

## Обзор

В `WSIPredictor` добавлен встроенный тайминг для анализа производительности. Тайминг использует `time.perf_counter()` для точного измерения времени выполнения различных этапов обработки.

## Использование

### Включение тайминга

```python
from scale import wsi, predict, model_config

# Создаем WSI объект
wsi_obj = wsi.WSI(wsi_path, num_sections=6)

# Создаем предиктор с включенным таймингом
predictor = predict.WSIPredictor(
    wsi_obj,
    model_configs,
    postprocess_settings,
    overlap_ratio=0.8,
    enable_timing=True  # Включаем тайминг
)

# При выполнении предикта будут выводиться сообщения о времени
predictions = predictor.predict_section(0)
```

### Пример вывода

```
[TIMER] _predict_section(index=0): 45.234 s
[TIMER] _postprocess_predictions(Mild): 0.123 s
[TIMER] _postprocess_predictions(Moderate): 0.456 s
[TIMER] _postprocess_predictions(Granulomas): 0.234 s
...
```

## Что измеряется

### Основные этапы

1. **`_predict_section(index=N)`** - время обработки секции (включая все модели и окна)
2. **`_postprocess_predictions(ClassName)`** - время постпроцессинга для каждого класса

### Для подсекций

При использовании `predict_section_via_subsections` тайминг показывает:
- Время обработки каждой подсекции (если включен)
- Время постпроцессинга для каждого класса

## Пример использования в тестах

```python
# test/test_predict_via_subsections.py
predictor = predict.WSIPredictor(
    wsi_obj,
    model_configs,
    postprocess_settings,
    overlap_ratio=0.2,
    enable_timing=True  # Включаем тайминг для детального анализа
)

# Тест 1: Обычный метод
print("Тест 1: predict_section")
predictions1 = predictor.predict_section(0)
# Вывод: [TIMER] _predict_section(index=0): 45.234 s
#        [TIMER] _postprocess_predictions(Mild): 0.123 s
#        ...

# Тест 2: Через подсекции
print("Тест 2: predict_section_via_subsections")
predictions2 = predictor.predict_section_via_subsections(0)
# Вывод: [TIMER] _predict_section(index=0): 18.567 s  (быстрее!)
#        [TIMER] _postprocess_predictions(Mild): 0.123 s
#        ...
```

## Анализ производительности

### Сравнение методов

Тайминг помогает сравнить производительность разных методов:

```python
# Обычный метод
predictor1 = predict.WSIPredictor(..., enable_timing=True)
predictions1 = predictor1.predict_section(0)
# [TIMER] _predict_section(index=0): 45.234 s

# Через подсекции
predictor2 = predict.WSIPredictor(..., enable_timing=True)
predictions2 = predictor2.predict_section_via_subsections(0)
# [TIMER] _predict_section(index=0): 18.567 s  # Ускорение ~2.4x
```

### Анализ узких мест

Тайминг показывает, где тратится больше всего времени:

```
[TIMER] _predict_section(index=0): 45.234 s
[TIMER] _postprocess_predictions(Mild): 0.123 s
[TIMER] _postprocess_predictions(Moderate): 0.456 s
[TIMER] _postprocess_predictions(Granulomas): 12.345 s  # Медленно!
```

Это помогает оптимизировать конкретные классы или модели.

## Отключение тайминга

По умолчанию тайминг отключен (`enable_timing=False`). Для отключения просто не указывайте параметр или установите `False`:

```python
predictor = predict.WSIPredictor(
    wsi_obj,
    model_configs,
    postprocess_settings,
    enable_timing=False  # Тайминг отключен (по умолчанию)
)
```

## Технические детали

- Использует `time.perf_counter()` для максимальной точности
- Формат вывода: `[TIMER] название: время s`
- Время выводится с точностью до миллисекунд (3 знака после запятой)
- Тайминг не влияет на производительность (overhead минимален)

## Интеграция с тестами

Все тестовые скрипты обновлены для использования тайминга:

- `test/test_predict_via_subsections.py` - основной тест сравнения
- `test/debug_parallel_subsections.py` - отладка параллелизации
- `predict_subsections_batch.py` - пакетная обработка
- `predict_subsections_batch_colab.py` - для Colab

Все они используют `enable_timing=True` для детального анализа производительности.





