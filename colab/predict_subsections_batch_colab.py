"""
Скрипт для пакетной обработки WSI файлов через подсекции.
Точный аналог обычного predict_first_section, но использует подсекции для ускорения.
Адаптирован для Google Colab (можно легко изменить пути для локального использования).
"""

import os
import json

from scale import wsi, predict, model_config, domain


# Настройки путей (адаптировано из вашего примера)
RESULTS_ROOT = "/content/drive/MyDrive/scale_results"
PREDS = os.path.join(RESULTS_ROOT, "predictions")
WSI_IMGS_DIR = "/content/wsi"

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
    print(wsi_name)
    
    # Пропускаем не-TIFF файлы
    if not wsi_name.lower().endswith(('.tiff', '.tif')):
        continue
    
    wsi_path = os.path.join(WSI_IMGS_DIR, wsi_name)
    
    try:
        # Создаем WSI объект с автоматическим определением подсекций
        wsi_img = wsi.WSI(
            wsi_path,
            num_sections=NUM_SECTIONS,
            num_subsections=NUM_SUBSECTIONS
        )
        
        # Создаем предиктор с параллелизацией для лучшей загрузки GPU A100
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
        # Используем новый метод вместо predict_first_section()
        preds = predictor.predict_first_section_via_subsections()
        
        # Сохраняем результаты (аналогично вашему примеру)
        wsi_name_no_ext = wsi_name.split(".")[0]
        preds_filename = wsi_name_no_ext + ".json"
        domain.predictions_to_json(preds, os.path.join(PREDS, preds_filename))
        
        print(wsi_name, "done!")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка при обработке {wsi_name}: {e}")
        import traceback
        traceback.print_exc()
        print()

print("="*60)
print("✅ ВСЕ ФАЙЛЫ ОБРАБОТАНЫ!")
print("="*60)

