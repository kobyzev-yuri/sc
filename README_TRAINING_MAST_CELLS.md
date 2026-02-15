# Руководство по обучению моделей для обнаружения мастоцитов

## Обзор

Это руководство описывает два практических подхода к улучшению обнаружения мастоцитов на основе рекомендаций Gemini 3 Pro:

1. **Knowledge Base / RAG для Gemini** - быстрый подход без обучения модели
2. **Файнтюнинг Vision модели (Qwen-VL) с LoRA** - долгосрочное решение с обучением

## ПОДХОД 1: Knowledge Base / RAG для Gemini

### Преимущества
- ✅ Быстрое внедрение (дни)
- ✅ Не требует GPU для обучения
- ✅ Легко обновлять
- ✅ Работает с существующим Gemini API

### Установка

```bash
# Установка зависимостей
pip install chromadb sentence-transformers

# Или добавьте в requirements.txt:
# chromadb>=0.4.0
# sentence-transformers>=2.2.0
```

### Шаг 1: Создание и пополнение Knowledge Base

```bash
# Создать и пополнить базу знаний из результатов анализа Gemini
python train_knowledge_base.py --action populate --kb_path ./mast_cells_kb

# Или по отдельности:
python train_knowledge_base.py --action add_explicit --kb_path ./mast_cells_kb
python train_knowledge_base.py --action add_implicit --kb_path ./mast_cells_kb
```

### Шаг 2: Использование Knowledge Base для анализа

```bash
# Анализ одного изображения с использованием KB
python analyze_with_kb.py \
    --image MAST_GEMINI/01_no.png \
    --kb_path ./mast_cells_kb \
    --query "Find all mast cells, including implicit ones" \
    --n_examples 5

# Анализ всех _no изображений
python analyze_with_kb.py \
    --kb_path ./mast_cells_kb \
    --n_examples 5

# С фильтрацией по типу мастоцитов
python analyze_with_kb.py \
    --image MAST_GEMINI/02_no.png \
    --filter_cell_type implicit \
    --n_examples 3
```

### Шаг 3: Поиск в Knowledge Base

```bash
# Поиск похожих примеров
python train_knowledge_base.py \
    --action search \
    --query "mast cell with central nucleus and pink cytoplasm" \
    --n_results 5

# Просмотр всех примеров
python train_knowledge_base.py --action list
```

### Структура Knowledge Base

База знаний хранит:
- Морфологические признаки мастоцитов
- Инсайты от Gemini
- Координаты и типы клеток
- Уровни сложности и уверенности
- Пути к изображениям

### Как это работает

1. **Поиск похожих примеров**: При запросе система ищет в базе знаний примеры, похожие на текущий случай
2. **Формирование контекста**: Найденные примеры добавляются в промпт для Gemini
3. **Улучшенный анализ**: Gemini использует контекст для более точного обнаружения

---

## ПОДХОД 2: Файнтюнинг Qwen-VL с LoRA

### Преимущества
- ✅ Полный контроль над моделью
- ✅ Модель "запоминает" паттерны
- ✅ Не зависит от внешних API
- ✅ Можно использовать локально

### Требования

- **GPU**: Минимум 16GB VRAM (для Qwen2-VL-2B) или 24GB+ (для 7B)
- **Диск**: ~20GB свободного места
- **Время обучения**: 2-6 часов (зависит от размера датасета)

### Установка

```bash
# Установка зависимостей
pip install torch>=2.0.0 transformers>=4.35.0 peft>=0.6.0 accelerate>=0.24.0
pip install bitsandbytes>=0.41.0  # для 8-bit training (опционально)
```

### Шаг 1: Подготовка данных

```python
# prepare_training_data.py
# (Скрипт описан в docs/TRAINING_APPROACHES_FOR_MAST_CELLS.md)

# Создает JSONL файл с данными для обучения
python prepare_training_data.py
```

Формат данных (JSONL):
```json
{
  "id": "mast_cell_01_no",
  "conversations": [
    {
      "from": "user",
      "value": [
        {"type": "image", "image": "data:image/png;base64,..."},
        {"type": "text", "text": "Find all mast cells..."}
      ]
    },
    {
      "from": "assistant",
      "value": "Мастоцит #1: координаты (x: 195, y: 190)..."
    }
  ]
}
```

### Шаг 2: Обучение модели

```bash
# train_qwen_vl_lora.py
# (Скрипт описан в docs/TRAINING_APPROACHES_FOR_MAST_CELLS.md)

python train_qwen_vl_lora.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --train_file mast_cells_training_data.jsonl \
    --output_dir ./qwen_vl_mast_cells_lora \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4
```

### Шаг 3: Использование обученной модели

```python
# use_trained_model.py
from use_trained_model import TrainedMastCellsDetector
from pathlib import Path

detector = TrainedMastCellsDetector(
    base_model="Qwen/Qwen2-VL-2B-Instruct",
    lora_path="./qwen_vl_mast_cells_lora"
)

result = detector.detect_mast_cells(Path("MAST_GEMINI/01_no.png"))
print(result)
```

---

## Сравнение подходов

| Критерий | Knowledge Base | Файнтюнинг LoRA |
|----------|---------------|-----------------|
| **Сложность** | Низкая | Средняя-Высокая |
| **Ресурсы** | Минимальные | GPU 16GB+ |
| **Время внедрения** | Дни | Недели |
| **Точность** | Зависит от KB | Может быть выше |
| **Обновляемость** | Легко | Нужно переобучать |
| **Стоимость** | Низкая | Средняя |

## Рекомендации

### Для быстрого старта:
1. ✅ **Начните с Knowledge Base** - быстрее внедрить
2. ✅ Пополните базу знаний результатами анализа Gemini
3. ✅ Используйте `analyze_with_kb.py` для улучшенного анализа

### Для долгосрочного решения:
1. ⏳ Параллельно готовьте данные для файнтюнинга
2. ⏳ Когда наберется 50-100 примеров, запустите обучение
3. ⏳ Используйте оба подхода: KB для контекста, обученная модель для точности

### Гибридный подход (рекомендуется):
1. Knowledge Base для контекста и быстрых ответов
2. Обученная модель для финальной детекции
3. Объединить результаты обоих подходов

---

## Структура файлов

```
sc/
├── train_knowledge_base.py          # Управление Knowledge Base
├── analyze_with_kb.py                # Анализ с использованием KB
├── prepare_training_data.py         # Подготовка данных для файнтюнинга
├── train_qwen_vl_lora.py            # Обучение Qwen-VL с LoRA
├── use_trained_model.py              # Использование обученной модели
├── mast_cells_kb/                    # Директория Knowledge Base
│   └── (ChromaDB файлы)
├── qwen_vl_mast_cells_lora/         # Обученная модель (после обучения)
└── docs/
    └── TRAINING_APPROACHES_FOR_MAST_CELLS.md  # Подробное руководство
```

---

## Следующие шаги

1. ✅ Создать Knowledge Base и пополнить её
2. ✅ Протестировать анализ с KB
3. ⏳ Собрать больше данных (50-100 примеров)
4. ⏳ Подготовить данные для файнтюнинга
5. ⏳ Настроить инфраструктуру для обучения (GPU)
6. ⏳ Обучить Qwen-VL с LoRA
7. ⏳ Сравнить результаты обоих подходов

---

## Полезные команды

```bash
# Просмотр статистики Knowledge Base
python train_knowledge_base.py --action list

# Поиск примеров неявных мастоцитов
python train_knowledge_base.py \
    --action search \
    --query "implicit mast cell with blurred boundaries" \
    --n_results 5

# Анализ с фокусом на неявные мастоциты
python analyze_with_kb.py \
    --image MAST_GEMINI/03_no.png \
    --filter_cell_type implicit \
    --n_examples 5
```

---

## Troubleshooting

### Knowledge Base не находит примеры
- Проверьте, что база знаний создана: `python train_knowledge_base.py --action list`
- Попробуйте более общий запрос
- Увеличьте `n_results`

### Ошибки при обучении Qwen-VL
- Проверьте наличие GPU: `nvidia-smi`
- Уменьшите `batch_size` если не хватает памяти
- Используйте меньшую модель (2B вместо 7B)

### Gemini API ошибки
- Проверьте API ключ в `config.env`
- Убедитесь, что баланс достаточен
- Проверьте размер изображений (уменьшите если нужно)

---

## Дополнительные ресурсы

- [ChromaDB документация](https://docs.trychroma.com/)
- [Qwen-VL GitHub](https://github.com/QwenLM/Qwen2-VL)
- [PEFT (LoRA) документация](https://huggingface.co/docs/peft/)
- [Подробное руководство](docs/TRAINING_APPROACHES_FOR_MAST_CELLS.md)

