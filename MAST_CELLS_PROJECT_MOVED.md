# Проект анализа мастоцитов перенесен

## ⚠️ ВАЖНО: Проект переехал

Все файлы, связанные с анализом мастоцитов через Gemini и Knowledge Base, были перенесены в отдельный проект.

## Новое расположение

**Проект:** `/mnt/ai/cnn/mast/`

**GitHub:** Будет создан отдельный репозиторий `mast` (после очистки текущего репозитория)

## Что было перенесено

### Скрипты:
- `analyze_mast_cells_gemini.py` → `../mast/`
- `analyze_mast_cells_coordinates_gemini.py` → `../mast/`
- `analyze_with_kb.py` → `../mast/`
- `train_knowledge_base.py` → `../mast/`
- `add_datasets_to_kb.py` → `../mast/`

### Документация:
- `README_MAST_CELLS_ANALYSIS.md` → `../mast/`
- `README_MAST_CELLS_COORDINATES_ANALYSIS.md` → `../mast/`
- `README_TRAINING_MAST_CELLS.md` → `../mast/`
- `MAST_CELLS_COORDINATES_ANALYSIS_SUMMARY.md` → `../mast/`
- `docs/GEMINI_DATASET_REQUIREMENTS.md` → `../mast/docs/`
- `docs/TRAINING_APPROACHES_FOR_MAST_CELLS.md` → `../mast/docs/`
- `docs/KNOWLEDGE_BASE_ARCHITECTURE.md` → `../mast/docs/`
- `docs/KB_QUICK_GUIDE.md` → `../mast/docs/`
- `docs/DATASETS_FOR_KB.md` → `../mast/docs/`

### Данные и результаты:
- `MAST_GEMINI/` → `../mast/data/MAST_GEMINI/`
- `mast_cells_analysis_result.txt` → `../mast/results/`
- `mast_cells_coordinates_analysis_result.*` → `../mast/results/`

## Причина переноса

Проект `sc` фокусируется на построении шкалы патологии (0-1) для анализа WSI изображений.

Проект `mast` фокусируется на обнаружении мастоцитов через Gemini 3 Pro и Knowledge Base (RAG).

Разделение проектов улучшает:
- Организацию кода
- Поддержку и развитие
- Независимое версионирование
- Четкость целей каждого проекта

## Связь между проектами

Проекты могут взаимодействовать:
- Результаты анализа мастоцитов могут использоваться в проекте `sc` для построения шкалы
- Методы из `sc` могут быть применены к данным мастоцитов

Но каждый проект имеет свою четкую область ответственности.

## Дата переноса

**2025-02-15**

---

**Для работы с проектом мастоцитов перейдите в:** `/mnt/ai/cnn/mast/`

