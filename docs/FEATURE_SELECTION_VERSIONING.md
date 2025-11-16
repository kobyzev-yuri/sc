# Система версионирования результатов подбора признаков

## Проблема

При запуске нового теста подбора признаков текущие хорошие результаты могут быть перезаписаны, что нежелательно.

## Решение

Система версионирования позволяет:
- ✅ Сохранять каждый эксперимент отдельно
- ✅ Сравнивать разные эксперименты
- ✅ Выбирать, какой эксперимент экспортировать в dashboard
- ✅ Создавать резервные копии перед экспортом
- ✅ Не перезаписывать dashboard автоматически

---

## Использование

### 1. Запуск подбора признаков

Результаты автоматически сохраняются в отдельную директорию:

```bash
# Быстрый тест
python3 test_feature_selection_quick.py results/predictions

# Полный анализ
python3 -m scale.feature_selection_automated results/predictions experiments/feature_selection
```

**Важно:** Конфигурация dashboard НЕ обновляется автоматически!

### 2. Просмотр списка экспериментов

```bash
# Показать все эксперименты
python3 -m scale.feature_selection_versioning_cli list

# Показать только завершенные
python3 -m scale.feature_selection_versioning_cli list --status completed

# Показать эксперименты с определенными тегами
python3 -m scale.feature_selection_versioning_cli list --tags forward_selection
```

### 3. Сравнение экспериментов

```bash
# Сравнить два эксперимента
python3 -m scale.feature_selection_versioning_cli compare \
    experiment_20251116_134939 \
    experiment_20251116_135951
```

### 4. Экспорт эксперимента в dashboard

```bash
# Экспортировать конкретный эксперимент
python3 -m scale.feature_selection_versioning_cli export experiment_20251116_134939

# Экспортировать без создания резервной копии (не рекомендуется)
python3 -m scale.feature_selection_versioning_cli export experiment_20251116_134939 --no-backup
```

**Что происходит:**
- ✅ Создается резервная копия текущей конфигурации dashboard
- ✅ Конфигурация dashboard обновляется признаками из эксперимента
- ✅ При следующем запуске dashboard будут использованы эти признаки

### 5. Найти лучший эксперимент

```bash
# Найти лучший по score
python3 -m scale.feature_selection_versioning_cli best

# Найти лучший по separation
python3 -m scale.feature_selection_versioning_cli best --metric separation

# Найти лучший и автоматически экспортировать его
python3 -m scale.feature_selection_versioning_cli best --export
```

---

## Структура файлов

```
experiments/
└── feature_selection/
    ├── experiments_metadata.json          ← Метаданные всех экспериментов
    ├── experiment_20251116_134939/        ← Эксперимент 1
    │   ├── feature_selection_results_*.csv
    │   ├── best_features_*.json
    │   ├── medical_report_*.md
    │   └── feature_selection_config_relative.json
    └── experiment_20251116_135951/        ← Эксперимент 2
        ├── feature_selection_results_*.csv
        ├── best_features_*.json
        └── ...

scale/
├── feature_selection_config_relative.json     ← Текущая конфигурация dashboard
└── feature_selection_config_relative_backup_*.json  ← Резервные копии
```

---

## Примеры использования

### Пример 1: Сохранение хорошего результата

```bash
# 1. Запустить подбор признаков
python3 test_feature_selection_quick.py results/predictions

# Результаты сохранены в experiments/feature_selection_quick/
# Dashboard НЕ обновлен (безопасно!)

# 2. Проверить результаты
cat experiments/feature_selection_quick/medical_report_*.md

# 3. Если результаты хорошие - экспортировать в dashboard
python3 -m scale.feature_selection_versioning_cli export feature_selection_quick
```

### Пример 2: Сравнение разных подходов

```bash
# 1. Запустить тест с Paneth
python3 test_feature_selection_quick.py results/predictions
# Результаты в: experiments/feature_selection_quick/

# 2. Запустить тест без Paneth
python3 test_without_paneth.py results/predictions
# Результаты в: experiments/feature_selection_no_paneth/

# 3. Сравнить результаты
python3 -m scale.feature_selection_versioning_cli compare \
    feature_selection_quick \
    feature_selection_no_paneth

# 4. Выбрать лучший и экспортировать
python3 -m scale.feature_selection_versioning_cli best --export
```

### Пример 3: Работа с несколькими экспериментами

```bash
# 1. Просмотреть все эксперименты
python3 -m scale.feature_selection_versioning_cli list

# 2. Сравнить несколько экспериментов
python3 -m scale.feature_selection_versioning_cli compare \
    exp1 exp2 exp3

# 3. Найти лучший по метрике
python3 -m scale.feature_selection_versioning_cli best --metric mod_norm

# 4. Экспортировать лучший
python3 -m scale.feature_selection_versioning_cli best --export
```

---

## Безопасность

### ✅ Защита от потери данных:

1. **Автоматическое сохранение:** Каждый эксперимент сохраняется в отдельную директорию
2. **Резервные копии:** При экспорте создается резервная копия текущей конфигурации
3. **Нет автоматического экспорта:** Dashboard не обновляется автоматически
4. **Явный экспорт:** Нужно явно указать, какой эксперимент экспортировать

### Резервные копии:

Резервные копии сохраняются в:
```
scale/feature_selection_config_relative_backup_TIMESTAMP.json
```

Чтобы вернуться к предыдущей версии:
```bash
# Найти резервную копию
ls -lt scale/feature_selection_config_relative_backup_*.json

# Восстановить (вручную)
cp scale/feature_selection_config_relative_backup_TIMESTAMP.json \
   scale/cfg/feature_selection_config_relative.json
```

---

## Рекомендации

### Рабочий процесс:

1. **Запустить тест:**
   ```bash
   python3 test_feature_selection_quick.py results/predictions
   ```

2. **Проверить результаты:**
   ```bash
   cat experiments/feature_selection_quick/medical_report_*.md
   ```

3. **Если результаты хорошие - экспортировать:**
   ```bash
   python3 -m scale.feature_selection_versioning_cli export feature_selection_quick
   ```

4. **Если хотите попробовать другой подход:**
   - Запустить новый тест (результаты сохранятся отдельно)
   - Сравнить с предыдущим
   - Выбрать лучший и экспортировать

### Перед презентацией медикам:

1. Найти лучший эксперимент:
   ```bash
   python3 -m scale.feature_selection_versioning_cli best
   ```

2. Экспортировать его в dashboard:
   ```bash
   python3 -m scale.feature_selection_versioning_cli best --export
   ```

3. Запустить dashboard:
   ```bash
   streamlit run scale/dashboard.py
   ```

---

## FAQ

### Q: Что делать, если случайно экспортировал неправильный эксперимент?

**A:** Используйте резервную копию:
```bash
# Найти последнюю резервную копию
ls -lt scale/feature_selection_config_relative_backup_*.json | head -1

# Восстановить
cp scale/feature_selection_config_relative_backup_TIMESTAMP.json \
   scale/cfg/feature_selection_config_relative.json
```

### Q: Как удалить старые эксперименты?

**A:** Удалите директорию эксперимента:
```bash
rm -rf experiments/feature_selection/experiment_NAME
```

### Q: Можно ли переименовать эксперимент?

**A:** Да, просто переименуйте директорию и обновите метаданные вручную (или удалите старую запись).

### Q: Как сравнить все эксперименты сразу?

**A:** Используйте команду `list` для просмотра всех экспериментов с метриками.

---

## Дополнительная информация

- **Исходный код:** `scale/feature_selection_versioning.py`
- **CLI утилита:** `scale/feature_selection_versioning_cli.py`
- **Документация по подбору признаков:** `docs/FEATURE_SELECTION_AUTOMATED.md`


