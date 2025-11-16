"""
Скрипт для пакетной обработки WSI файлов через подсекции.
Аналог обычного predict_first_section, но использует подсекции для ускорения.
"""

import os
import json
from pathlib import Path

from scale import wsi, predict, model_config, domain


# Настройки путей (измените под вашу структуру)
RESULTS_ROOT = "./scale_results"
PREDS = os.path.join(RESULTS_ROOT, "predictions")
WSI_IMGS_DIR = "./wsi"

# Параметры обработки
NUM_SECTIONS = 6  # Количество секций для поиска (или None для автоматического)
NUM_SUBSECTIONS = None  # Количество подсекций (None для автоматического определения)
OVERLAP_RATIO = 0.8  # Перекрытие для скользящего окна

# Создаем директории
os.makedirs(PREDS, exist_ok=True)

# Загружаем модели и настройки один раз
print("⏳ Загружаем модели и настройки...")
model_configs = model_config.create_model_configs()
postprocess_settings = model_config.get_postprocess_settings()
print(f"✅ Загружено моделей: {len(model_configs)}")
print()

# Обрабатываем каждый WSI файл
for wsi_name in os.listdir(WSI_IMGS_DIR):
    # Пропускаем не-TIFF файлы
    if not wsi_name.lower().endswith(('.tiff', '.tif')):
        continue
    
    print(f"📁 Обрабатываем: {wsi_name}")
    wsi_path = os.path.join(WSI_IMGS_DIR, wsi_name)
    
    try:
        # Создаем WSI объект с автоматическим определением подсекций
        print("  ⏳ Создаем WSI объект...")
        wsi_img = wsi.WSI(
            wsi_path,
            num_sections=NUM_SECTIONS,
            num_subsections=NUM_SUBSECTIONS
        )
        
        # Проверяем наличие подсекций для секции 0
        subsection_bounds = wsi_img.extract_subsection_bounds(0)
        print(f"  ✅ Найдено подсекций: {len(subsection_bounds)}")
        
        # Создаем предиктор с параллелизацией для лучшей загрузки GPU
        predictor = predict.WSIPredictor(
            wsi_img,
            model_configs,
            postprocess_settings,
            overlap_ratio=OVERLAP_RATIO,
            parallel_subsections=True,  # Включаем параллельную обработку подсекций
            max_workers=4,  # 4 потока для оптимальной загрузки A100
            enable_timing=True  # Включаем тайминг для анализа производительности
        )
        
        # Получаем предсказания через подсекции (быстрее!)
        print("  ⏳ Получаем predictions через подсекции...")
        preds = predictor.predict_first_section_via_subsections()
        
        # Сохраняем результаты
        wsi_name_no_ext = wsi_name.split(".")[0]
        preds_filename = wsi_name_no_ext + ".json"
        preds_path = os.path.join(PREDS, preds_filename)
        
        domain.predictions_to_json(preds, preds_path)
        
        # Подсчитываем количество предиктов
        total_preds = sum(len(preds_list) for preds_list in preds.values())
        print(f"  ✅ Сохранено {total_preds} predictions в {preds_filename}")
        print(f"  ✅ {wsi_name} done!")
        
    except Exception as e:
        print(f"  ❌ Ошибка при обработке {wsi_name}: {e}")
        import traceback
        traceback.print_exc()
    
    print()

print("="*60)
print("✅ ВСЕ ФАЙЛЫ ОБРАБОТАНЫ!")
print("="*60)

