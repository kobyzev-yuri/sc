# Рабочий процесс подбора признаков и визуализации в dashboard

## Обзор

Система автоматизированного подбора признаков интегрирована с dashboard для удобного представления результатов медикам. Все результаты автоматически экспортируются в формат, совместимый с dashboard.

---

## Рабочий процесс

### Шаг 1: Подбор признаков

Запустите автоматизированный подбор признаков:

```bash
# Быстрый тест нескольких методов
python3 test_feature_selection_quick.py results/predictions

# Полный анализ всех методов
python3 -m scale.feature_selection_automated results/predictions experiments/feature_selection
```

**Что происходит:**
- ✅ Тестируются различные методы подбора признаков
- ✅ Результаты сравниваются по метрикам качества
- ✅ Лучший набор признаков автоматически экспортируется в dashboard
- ✅ Создается медицинский отчет с интерпретацией результатов

### Шаг 2: Автоматический экспорт в dashboard

После завершения подбора признаков:

**Автоматически создаются файлы:**
1. **`scale/feature_selection_config_relative.json`** - конфигурация для dashboard
   - При следующем запуске dashboard эти признаки будут автоматически загружены
   
2. **`experiments/feature_selection_*/medical_report_*.md`** - отчет для медиков
   - Содержит метрики, интерпретацию и рекомендации
   
3. **`experiments/feature_selection_*/feature_selection_results_*.csv`** - таблица сравнения методов
   
4. **`experiments/feature_selection_*/best_features_*.json`** - лучший набор признаков

### Шаг 3: Визуализация в dashboard

Запустите dashboard:

```bash
streamlit run scale/dashboard.py
```

**Что происходит:**
- ✅ Dashboard автоматически загружает отобранные признаки из `feature_selection_config_relative.json`
- ✅ Все визуализации строятся с использованием этих признаков
- ✅ Медики могут видеть результаты в удобном интерактивном формате
- ✅ Можно вручную изменить набор признаков через интерфейс dashboard

---

## Структура файлов

```
sc/
├── scale/
│   ├── feature_selection_config_relative.json  ← Автоматически обновляется
│   └── feature_selection_config_absolute.json
│
├── experiments/
│   └── feature_selection_*/
│       ├── feature_selection_results_*.csv      ← Таблица сравнения методов
│       ├── best_features_*.json                  ← Лучший набор признаков
│       ├── medical_report_*.md                   ← Отчет для медиков
│       └── feature_selection_config_relative.json  ← Копия для dashboard
│
└── docs/
    └── FEATURE_SELECTION_WORKFLOW.md  ← Эта инструкция
```

---

## Примеры использования

### Пример 1: Быстрый тест и визуализация

```bash
# 1. Запустить быстрый тест
python3 test_feature_selection_quick.py results/predictions

# 2. Запустить dashboard (признаки уже загружены автоматически)
streamlit run scale/dashboard.py

# 3. В dashboard:
#    - Выбрать директорию с предсказаниями
#    - Включить "Использовать относительные признаки"
#    - Признаки уже будут выбраны автоматически!
#    - Нажать "Запустить анализ"
```

### Пример 2: Полный анализ с сохранением

```bash
# 1. Запустить полный анализ
python3 -m scale.feature_selection_automated \
    results/predictions \
    experiments/feature_selection_full

# 2. Результаты сохранены в experiments/feature_selection_full/
#    - CSV с результатами всех методов
#    - JSON с лучшим набором признаков
#    - Markdown отчет для медиков
#    - Конфигурация для dashboard

# 3. Запустить dashboard
streamlit run scale/dashboard.py
```

### Пример 3: Ручной экспорт результатов

```python
from scale import feature_selection_export
import pandas as pd

# Загрузить результаты
results_df = pd.read_csv("experiments/feature_selection/results.csv")

# Экспортировать в dashboard
saved_files = feature_selection_export.export_complete_results(
    results_df=results_df,
    output_dir="experiments/export",
    use_relative_features=True,
    auto_export_to_dashboard=True,
)

print(f"Конфигурация для dashboard: {saved_files['dashboard_config']}")
print(f"Медицинский отчет: {saved_files['medical_report']}")
```

---

## Преимущества этого подхода

### ✅ Для исследователей:
- Автоматический экспорт результатов в dashboard
- Не нужно вручную копировать признаки
- Все результаты сохраняются с метаданными

### ✅ Для медиков:
- Удобная визуализация в dashboard
- Медицинский отчет с интерпретацией метрик
- Возможность коллективного анализа результатов

### ✅ Для команды:
- Единый формат хранения результатов
- Возможность сравнения разных методов
- История экспериментов сохраняется

---

## Рекомендации

### Перед презентацией медикам:

1. **Запустите подбор признаков:**
   ```bash
   python3 test_feature_selection_quick.py results/predictions
   ```

2. **Проверьте медицинский отчет:**
   ```bash
   cat experiments/feature_selection_quick/medical_report_*.md
   ```

3. **Запустите dashboard и проверьте визуализацию:**
   ```bash
   streamlit run scale/dashboard.py
   ```

4. **Подготовьте презентацию:**
   - Используйте медицинский отчет как основу
   - Покажите визуализации из dashboard
   - Объясните метрики качества

### Для коллективного анализа:

1. **Сохраните результаты в общей директории:**
   ```bash
   python3 -m scale.feature_selection_automated \
       results/predictions \
       experiments/shared/feature_selection_$(date +%Y%m%d)
   ```

2. **Поделитесь медицинским отчетом:**
   - Отправьте `medical_report_*.md` коллегам
   - Или покажите в dashboard

3. **Обсудите результаты:**
   - Используйте dashboard для интерактивного анализа
   - Сравните разные методы через CSV файлы

---

## Часто задаваемые вопросы

### Q: Как изменить набор признаков в dashboard после подбора?

**A:** В dashboard есть интерфейс для выбора признаков. Вы можете:
- Вручную выбрать/снять признаки через чекбоксы
- Нажать "Сохранить конфигурацию" для сохранения изменений

### Q: Можно ли использовать несколько наборов признаков?

**A:** Да! Вы можете:
- Сохранить разные конфигурации в разные файлы
- Переименовать `feature_selection_config_relative.json` перед новым подбором
- Загружать нужную конфигурацию в dashboard вручную

### Q: Как сравнить результаты разных экспериментов?

**A:** Используйте CSV файлы с результатами:
```bash
# Сравнить результаты
diff experiments/feature_selection_1/results.csv \
      experiments/feature_selection_2/results.csv
```

### Q: Что делать, если dashboard не загружает признаки?

**A:** Проверьте:
1. Существует ли файл `scale/feature_selection_config_relative.json`
2. Правильный ли формат JSON
3. Есть ли выбранные признаки в данных

---

## Дополнительная информация

- **Полная документация методов:** `docs/FEATURE_SELECTION_AUTOMATED.md`
- **Краткая сводка методов:** `docs/FEATURE_SELECTION_METHODS_SUMMARY.md`
- **Исходный код:** `scale/feature_selection_automated.py` и `scale/feature_selection_export.py`


