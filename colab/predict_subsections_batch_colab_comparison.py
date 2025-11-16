"""
Скрипт для сравнения производительности обычного метода и метода через подсекции на Colab.
Обрабатывает WSI файлы обоими методами и показывает ускорение.
Адаптирован для Google Colab.
"""

import os
import json
import time

from scale import wsi, predict, model_config, domain


# Настройки путей (адаптировано для Colab)
RESULTS_ROOT = "/content/drive/MyDrive/scale_results"
PREDS_STANDARD = os.path.join(RESULTS_ROOT, "predictions_standard")
PREDS_SUBSECTIONS = os.path.join(RESULTS_ROOT, "predictions_subsections")
WSI_IMGS_DIR = "/content/wsi"

# Параметры обработки
NUM_SECTIONS = 6  # Количество секций для поиска (или None для автоматического)
NUM_SUBSECTIONS = None  # Количество подсекций (None для автоматического определения)
OVERLAP_RATIO = 0.8  # Перекрытие для скользящего окна

# Создаем директории
os.makedirs(PREDS_STANDARD, exist_ok=True)
os.makedirs(PREDS_SUBSECTIONS, exist_ok=True)

# Загружаем модели и настройки один раз
print("⏳ Загружаем модели и настройки...")
model_configs = model_config.create_model_configs()
postprocess_settings = model_config.get_postprocess_settings()
print(f"✅ Загружено моделей: {len(model_configs)}")
print()

# Статистика
total_files = 0
total_time_standard = 0
total_time_subsections = 0

# Обрабатываем каждый WSI файл
for wsi_name in os.listdir(WSI_IMGS_DIR):
    # Пропускаем не-TIFF файлы
    if not wsi_name.lower().endswith(('.tiff', '.tif')):
        continue
    
    total_files += 1
    print(f"📁 Обрабатываем: {wsi_name}")
    wsi_path = os.path.join(WSI_IMGS_DIR, wsi_name)
    wsi_name_no_ext = wsi_name.split(".")[0]
    
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
        
        # Метод 1: Обычный predict_first_section
        print("  ⏳ Метод 1: Обычный predict_first_section...")
        predictor_standard = predict.WSIPredictor(
            wsi_img,
            model_configs,
            postprocess_settings,
            overlap_ratio=OVERLAP_RATIO,
            enable_timing=True  # Включаем тайминг
        )
        
        start_time = time.time()
        preds_standard = predictor_standard.predict_first_section()
        time_standard = time.time() - start_time
        total_time_standard += time_standard
        
        preds_filename = wsi_name_no_ext + ".json"
        domain.predictions_to_json(
            preds_standard,
            os.path.join(PREDS_STANDARD, preds_filename)
        )
        total_preds_standard = sum(len(preds_list) for preds_list in preds_standard.values())
        print(f"     ✅ Время: {time_standard:.2f} сек ({time_standard/60:.1f} мин), Predictions: {total_preds_standard}")
        
        # Метод 2: Через подсекции
        print("  ⏳ Метод 2: predict_first_section_via_subsections...")
        predictor_subsections = predict.WSIPredictor(
            wsi_img,
            model_configs,
            postprocess_settings,
            overlap_ratio=OVERLAP_RATIO,
            parallel_subsections=True,  # Включаем параллельную обработку подсекций
            max_workers=4,  # 4 потока для оптимальной загрузки A100
            enable_timing=True  # Включаем тайминг
        )
        
        start_time = time.time()
        preds_subsections = predictor_subsections.predict_first_section_via_subsections()
        time_subsections = time.time() - start_time
        total_time_subsections += time_subsections
        
        domain.predictions_to_json(
            preds_subsections,
            os.path.join(PREDS_SUBSECTIONS, preds_filename)
        )
        total_preds_subsections = sum(len(preds_list) for preds_list in preds_subsections.values())
        print(f"     ✅ Время: {time_subsections:.2f} сек ({time_subsections/60:.1f} мин), Predictions: {total_preds_subsections}")
        
        # Сравнение
        speedup = time_standard / time_subsections if time_subsections > 0 else 0
        pred_diff = abs(total_preds_standard - total_preds_subsections)
        pred_diff_pct = (pred_diff / total_preds_standard * 100) if total_preds_standard > 0 else 0
        
        print(f"  📊 Ускорение: {speedup:.2f}x")
        print(f"  📊 Разница в predictions: {pred_diff} ({pred_diff_pct:.1f}%)")
        print(f"  💾 Сэкономлено времени: {time_standard - time_subsections:.2f} сек ({(time_standard - time_subsections)/60:.1f} мин)")
        print(f"  ✅ {wsi_name} done!")
        
    except Exception as e:
        print(f"  ❌ Ошибка при обработке {wsi_name}: {e}")
        import traceback
        traceback.print_exc()
    
    print()

# Итоговая статистика
print("="*60)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*60)
print(f"Обработано файлов: {total_files}")
print(f"Общее время обычного метода: {total_time_standard:.2f} сек ({total_time_standard/60:.1f} мин)")
print(f"Общее время через подсекции: {total_time_subsections:.2f} сек ({total_time_subsections/60:.1f} мин)")
if total_time_subsections > 0:
    overall_speedup = total_time_standard / total_time_subsections
    print(f"Общее ускорение: {overall_speedup:.2f}x")
    time_saved = total_time_standard - total_time_subsections
    print(f"Сэкономлено времени: {time_saved:.2f} сек ({time_saved/60:.1f} мин)")
print("="*60)




