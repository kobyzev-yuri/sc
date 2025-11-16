# Параллельная обработка подсекций для эффективного использования GPU

## Проблема

При обработке подсекций последовательно GPU A100 используется неэффективно - загрузка низкая, так как подсекции обрабатываются одна за другой.

## Решение

Реализована двухуровневая оптимизация для максимальной загрузки GPU:

1. **Параллельная обработка подсекций** через ThreadPoolExecutor
2. **Batch processing патчей** внутри каждой подсекции

## Как это работает

### Уровень 1: Параллельная обработка подсекций

```python
# Несколько подсекций обрабатываются параллельно
with ThreadPoolExecutor(max_workers=4) as executor:
    # Подсекция 0-0 обрабатывается в потоке 1
    # Подсекция 0-1 обрабатывается в потоке 2
    # Подсекция 0-2 обрабатывается в потоке 3
    # Подсекция 0-3 обрабатывается в потоке 4
    # Все одновременно используют GPU A100
```

**Преимущества:**
- YOLO модели потокобезопасны для работы с GPU
- Несколько подсекций обрабатываются одновременно
- GPU загружается более эффективно

### Уровень 2: Batch processing патчей

```python
# Внутри каждой подсекции патчи собираются в батчи
batch_regions = [patch1, patch2, patch3, ..., patch8]  # batch_size=8
batch_preds = model.predict(batch_regions)  # Обрабатываем батч целиком
```

**Преимущества:**
- GPU обрабатывает несколько патчей одновременно
- Лучшая утилизация памяти GPU
- Меньше overhead на вызовы модели

## Использование

### Базовое использование (параллелизация включена по умолчанию)

```python
from scale import wsi, predict, model_config

# Создаем WSI
wsi_obj = wsi.WSI(
    wsi_path="./wsi/image.tiff",
    num_sections=6,
    num_subsections=None
)

# Создаем предиктор (параллелизация включена по умолчанию)
predictor = predict.WSIPredictor(
    wsi_obj,
    model_configs,
    postprocess_settings,
    overlap_ratio=0.8,
    parallel_subsections=True,  # По умолчанию True
    max_workers=None  # Автоматически: min(количество_подсекций, 4)
)

# Обрабатываем с параллелизацией
predictions = predictor.predict_section_via_subsections(section_index=0)
```

### Настройка количества воркеров

```python
# Для A100 оптимально 4-8 воркеров
predictor = predict.WSIPredictor(
    wsi_obj,
    model_configs,
    postprocess_settings,
    overlap_ratio=0.8,
    parallel_subsections=True,
    max_workers=6  # Явно указываем количество потоков
)
```

### Отключение параллелизации

```python
# Если нужно последовательное выполнение
predictor = predict.WSIPredictor(
    wsi_obj,
    model_configs,
    postprocess_settings,
    overlap_ratio=0.8,
    parallel_subsections=False  # Отключаем параллелизацию
)
```

## Параметры

### `parallel_subsections: bool = True`
- Включает/выключает параллельную обработку подсекций
- По умолчанию включено для лучшей загрузки GPU

### `max_workers: Optional[int] = None`
- Количество потоков для параллельной обработки
- По умолчанию: `min(количество_подсекций, 4)`
- Для A100 рекомендуется 4-8 воркеров

### `batch_size: int = 8` (внутренний параметр)
- Размер батча для обработки патчей внутри подсекции
- По умолчанию 8 патчей на батч
- Можно настроить в зависимости от размера патчей и памяти GPU

## Производительность

### Ожидаемое ускорение

- **Параллельная обработка подсекций**: 2-4x ускорение (зависит от количества подсекций)
- **Batch processing патчей**: 1.5-2x ускорение (зависит от размера патчей)
- **Общее ускорение**: 3-8x по сравнению с последовательной обработкой

### Мониторинг загрузки GPU

```bash
# Во время обработки проверяйте загрузку GPU
watch -n 1 nvidia-smi
```

Ожидаемая загрузка GPU A100:
- **Без параллелизации**: 20-40%
- **С параллелизацией**: 60-90%

## Ограничения

1. **Память GPU**: При большом количестве воркеров может не хватить памяти
   - Решение: уменьшить `max_workers` или `batch_size`

2. **Количество подсекций**: Если подсекция одна, параллелизация не поможет
   - Решение: используйте больше подсекций или обрабатывайте несколько секций

3. **Размер патчей**: Большие патчи могут ограничить размер батча
   - Решение: уменьшите `batch_size` в коде

## Рекомендации для A100

1. **Количество воркеров**: 4-6 для оптимальной загрузки
2. **Batch size**: 8-16 патчей (зависит от размера патча)
3. **Мониторинг**: Следите за использованием памяти GPU через `nvidia-smi`

## Пример использования

```python
import time
from scale import wsi, predict, model_config

# Загружаем модели
model_configs = model_config.create_model_configs()
postprocess_settings = model_config.get_postprocess_settings()

# Создаем WSI
wsi_obj = wsi.WSI(
    "./wsi/image.tiff",
    num_sections=6,
    num_subsections=None
)

# Создаем предиктор с параллелизацией
predictor = predict.WSIPredictor(
    wsi_obj,
    model_configs,
    postprocess_settings,
    overlap_ratio=0.8,
    parallel_subsections=True,
    max_workers=4  # 4 потока для A100
)

# Обрабатываем с измерением времени
start = time.time()
predictions = predictor.predict_section_via_subsections(section_index=0)
elapsed = time.time() - start

print(f"Время обработки: {elapsed:.2f} сек")
print(f"Загрузка GPU должна быть 60-90%")
```

## Технические детали

### ThreadPoolExecutor vs ProcessPoolExecutor

Используется **ThreadPoolExecutor**, так как:
- YOLO модели потокобезопасны для работы с GPU
- Потоки разделяют один GPU эффективнее процессов
- Меньше overhead на создание процессов

### Batch processing

Патчи собираются в батчи размером 8 (по умолчанию):
- YOLO эффективно обрабатывает батчи
- Лучшая утилизация GPU памяти
- Меньше вызовов модели = меньше overhead

### Потокобезопасность

- YOLO модели потокобезопасны для работы с GPU
- Не требуется lock для синхронизации
- Каждый поток обрабатывает свою подсекцию независимо





