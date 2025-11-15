"""
Скрипт для предсказания на WSI с использованием ONNX моделей (стандартный метод).

Использование на Google Colab:
1. Загрузите WSI файлы в MyDrive/sc/wsi/
2. Убедитесь что ONNX модели находятся в MyDrive/sc/models/
3. Запустите скрипт
4. Результаты сохранятся в MyDrive/sc/results/predictions/
"""

import sys
from pathlib import Path
import json

# Установка зависимостей
print("=" * 60)
print("Установка зависимостей")
print("=" * 60)

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "cucim", "opencv-python-headless", "scikit-image", "scikit-learn", "shapely"])

print("✅ Зависимости установлены\n")

# Импорты
from google.colab import drive
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Добавляем путь к проекту (если нужно)
# sys.path.insert(0, '/content/drive/MyDrive/sc')

# ============================================
# Конфигурация путей
# ============================================
print("=" * 60)
print("Настройка путей")
print("=" * 60)

# Монтируем Google Drive
drive.mount('/content/drive')

# Пути
DRIVE_SC_PATH = Path('/content/drive/MyDrive/sc')
WSI_DIR = DRIVE_SC_PATH / 'wsi'
MODELS_DIR = DRIVE_SC_PATH / 'models'
RESULTS_DIR = DRIVE_SC_PATH / 'results' / 'predictions'

# Создаем директории если нужно
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ WSI директория: {WSI_DIR}")
print(f"✅ Модели директория: {MODELS_DIR}")
print(f"✅ Результаты директория: {RESULTS_DIR}\n")

# ============================================
# Конфигурация моделей
# ============================================
print("=" * 60)
print("Загрузка конфигурации моделей")
print("=" * 60)

# Конфигурация моделей (адаптировано из model_config.py)
MODEL_CONFIGS = [
    {
        'name': 'Mild',
        'model_file': 'nn_seg_data_outputs_mild_train.onnx',
        'window_size': 514,
        'min_conf': 0.25,
        'classes': {0: 'Mild'}
    },
    {
        'name': 'Moderate',
        'model_file': 'moderate_seg_train6_acc2.onnx',
        'window_size': 514,
        'min_conf': 0.25,
        'classes': {0: 'Moderate'}
    },
    {
        'name': 'Dysplasia',
        'model_file': 'nn_det2_data_outputs_dysplasia_ibd_seg_train3.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'Dysplasia'}
    },
    {
        'name': 'Meta',
        'model_file': 'nn_det2_data_outputs_meta_train4.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'Meta'}
    },
    {
        'name': 'Plasma Cells',
        'model_file': 'nn_det2_data_outputs_plasma-transformed_train3.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'Plasma Cells'}
    },
    {
        'name': 'Neutrophils',
        'model_file': 'nn_det2_data_outputs_neutrophils_train7.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'Neutrophils'}
    },
    {
        'name': 'EoE',
        'model_file': 'nn_det2_data_outputs_eoe_train6.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'EoE'}
    },
    {
        'name': 'Enterocytes',
        'model_file': 'nn_det2_data_outputs_enterocytes_train2.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'Enterocytes'}
    },
    {
        'name': 'Granulomas',
        'model_file': 'nn_det2_data_outputs_gran_train5.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'Granulomas'}
    },
    {
        'name': 'Paneth',
        'model_file': 'nn_det2_data_outputs_paneth_train5.onnx',
        'window_size': 640,
        'min_conf': 0.25,
        'classes': {0: 'Paneth'}
    },
]

# Загружаем модели
print("⏳ Загрузка ONNX моделей...")
loaded_models = []
for config in MODEL_CONFIGS:
    model_path = MODELS_DIR / config['model_file']
    if model_path.exists():
        try:
            model = YOLO(str(model_path))
            loaded_models.append({
                'model': model,
                'config': config
            })
            print(f"✅ Загружена: {config['name']}")
        except Exception as e:
            print(f"❌ Ошибка загрузки {config['name']}: {e}")
    else:
        print(f"⚠️  Файл не найден: {model_path}")

print(f"\n✅ Загружено моделей: {len(loaded_models)}/{len(MODEL_CONFIGS)}\n")

# ============================================
# Загрузка WSI файлов
# ============================================
print("=" * 60)
print("Поиск WSI файлов")
print("=" * 60)

wsi_files = list(WSI_DIR.glob('*.tiff')) + list(WSI_DIR.glob('*.tif'))
if not wsi_files:
    print(f"❌ Не найдено WSI файлов в {WSI_DIR}")
    print("   Загрузите .tiff или .tif файлы в MyDrive/sc/wsi/")
else:
    print(f"✅ Найдено {len(wsi_files)} WSI файлов:")
    for wsi_file in wsi_files:
        print(f"   - {wsi_file.name}")

# ============================================
# Предсказание (стандартный метод)
# ============================================
if wsi_files and loaded_models:
    print("\n" + "=" * 60)
    print("Начало предсказания (стандартный метод)")
    print("=" * 60)
    
    from cucim.clara import CuImage
    import cv2
    
    for wsi_file in wsi_files:
        print(f"\n📁 Обработка: {wsi_file.name}")
        
        try:
            # Загружаем WSI
            wsi = CuImage(str(wsi_file))
            wsi_size = wsi.resolutions["level_dimensions"][0]
            print(f"   Размер WSI: {wsi_size[0]}x{wsi_size[1]}")
            
            # Получаем первую секцию (упрощенно - весь WSI)
            # В реальности нужно использовать extract_biopsy_bound, но для Colab упрощаем
            window_size = 640
            overlap_ratio = 0.5
            stride = int(window_size * (1 - overlap_ratio))
            
            all_predictions = defaultdict(list)
            
            # Обрабатываем окна
            print(f"   ⏳ Обработка окон (размер: {window_size}, stride: {stride})...")
            window_count = 0
            
            for y in range(0, wsi_size[1], stride):
                for x in range(0, wsi_size[0], stride):
                    size_x = min(window_size, wsi_size[0] - x)
                    size_y = min(window_size, wsi_size[1] - y)
                    
                    # Читаем регион
                    region = wsi.read_region(
                        location=(x, y),
                        size=(size_x, size_y),
                        level=0,
                    )
                    region_bgr = cv2.cvtColor(np.asarray(region)[..., :3], cv2.COLOR_RGB2BGR)
                    
                    # Предсказание для каждого окна
                    for model_wrapper in loaded_models:
                        config = model_wrapper['config']
                        if size_x == window_size and size_y == window_size:
                            # Только полные окна
                            try:
                                preds = model_wrapper['model'].predict([region_bgr], verbose=False, conf=config['min_conf'])
                                
                                # Парсим предсказания YOLO
                                for pred in preds:
                                    boxes = pred.boxes.xyxy.cpu().numpy()
                                    confs = pred.boxes.conf.cpu().numpy()
                                    cls_indexes = pred.boxes.cls.cpu().numpy().astype(int)
                                    
                                    for box, conf, cls_idx in zip(boxes, confs, cls_indexes):
                                        cls_name = config['classes'].get(cls_idx, f"Class_{cls_idx}")
                                        
                                        # Добавляем смещение координат
                                        pred_with_offset = {
                                            'box': [
                                                float(box[0]) + x,
                                                float(box[1]) + y,
                                                float(box[2]) + x,
                                                float(box[3]) + y
                                            ],
                                            'conf': float(conf)
                                        }
                                        all_predictions[cls_name].append(pred_with_offset)
                            except Exception as e:
                                # Пропускаем ошибки предсказания
                                continue
                    
                    window_count += 1
                    if window_count % 100 == 0:
                        print(f"      Обработано окон: {window_count}")
            
            print(f"   ✅ Обработано окон: {window_count}")
            
            # Сохраняем результаты
            wsi_name = wsi_file.stem
            output_file = RESULTS_DIR / f"{wsi_name}.json"
            
            # Конвертируем в формат JSON
            output_data = {}
            for cls_name, preds in all_predictions.items():
                output_data[cls_name] = preds
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"   ✅ Результаты сохранены: {output_file}")
            print(f"   📊 Всего предсказаний: {sum(len(preds) for preds in all_predictions.values())}")
            
        except Exception as e:
            print(f"   ❌ Ошибка при обработке {wsi_file.name}: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ГОТОВО!")
print("=" * 60)
print(f"\n📁 Результаты сохранены в: {RESULTS_DIR}")
