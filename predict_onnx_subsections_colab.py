"""
Скрипт для предсказания на WSI с использованием ONNX моделей (метод через подсекции).

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
from cucim.clara import CuImage
import cv2
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans

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

MODEL_CONFIGS = [
    {
        'name': 'Mild',
        'model_file': 'nn_seg_data_outputs_mild_train.onnx',
        'window_size': 514,
        'min_conf': 0.6,
        'classes': {0: 'Mild'}
    },
    {
        'name': 'Moderate',
        'model_file': 'moderate_seg_train6_acc2.onnx',
        'window_size': 514,
        'min_conf': 0.7,
        'classes': {0: 'Moderate'}
    },
    {
        'name': 'Dysplasia',
        'model_file': 'nn_det2_data_outputs_dysplasia_ibd_seg_train3.onnx',
        'window_size': 640,
        'min_conf': 0.2,
        'classes': {0: 'Dysplasia'}
    },
    {
        'name': 'Meta',
        'model_file': 'nn_det2_data_outputs_meta_train4.onnx',
        'window_size': 640,
        'min_conf': 0.2,
        'classes': {0: 'Meta'}
    },
    {
        'name': 'Plasma Cells',
        'model_file': 'nn_det2_data_outputs_plasma-transformed_train3.onnx',
        'window_size': 640,
        'min_conf': 0.3,
        'classes': {0: 'Plasma Cells'}
    },
    {
        'name': 'Neutrophils',
        'model_file': 'nn_det2_data_outputs_neutrophils_train7.onnx',
        'window_size': 640,
        'min_conf': 0.3,
        'classes': {0: 'Neutrophils'}
    },
    {
        'name': 'EoE',
        'model_file': 'nn_det2_data_outputs_eoe_train6.onnx',
        'window_size': 640,
        'min_conf': 0.4,
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
        'min_conf': 0.2,
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
# Вспомогательные функции
# ============================================
def extract_biopsy_bound(wsi, section_index=0):
    """Упрощенное извлечение границ биопсии."""
    # Получаем thumbnail для определения границ
    thumb = wsi.read_region(
        location=(0, 0),
        size=wsi.resolutions["level_dimensions"][-1],  # Самый низкий уровень
        level=len(wsi.resolutions["level_dimensions"]) - 1
    )
    thumb_gray = cv2.cvtColor(np.asarray(thumb)[..., :3], cv2.COLOR_RGB2GRAY)
    
    # Бинаризация
    _, binary = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Берем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Масштабируем обратно к уровню 0
        scale_factor = wsi.resolutions["level_dimensions"][0][0] / wsi.resolutions["level_dimensions"][-1][0]
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        w = int(w * scale_factor)
        h = int(h * scale_factor)
        
        return {'x': x, 'y': y, 'w': w, 'h': h}
    else:
        # Если не нашли контуры, используем весь WSI
        size = wsi.resolutions["level_dimensions"][0]
        return {'x': 0, 'y': 0, 'w': size[0], 'h': size[1]}

def extract_subsection_bounds(wsi, section_index=0, num_subsections=None):
    """Упрощенное извлечение подсекций через кластеризацию."""
    bound = extract_biopsy_bound(wsi, section_index)
    
    # Получаем thumbnail области биопсии
    thumb_size = (1024, 1024)
    thumb = wsi.read_region(
        location=(bound['x'], bound['y']),
        size=(min(bound['w'], thumb_size[0]), min(bound['h'], thumb_size[1])),
        level=0
    )
    thumb_gray = cv2.cvtColor(np.asarray(thumb)[..., :3], cv2.COLOR_RGB2GRAY)
    
    # Бинаризация
    _, binary = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Находим связанные компоненты
    labeled = label(binary)
    regions = regionprops(labeled)
    
    if not regions:
        # Если не нашли регионы, возвращаем одну подсекцию (весь bound)
        return [{
            'x': bound['x'],
            'y': bound['y'],
            'w': bound['w'],
            'h': bound['h']
        }]
    
    # Берем центроиды регионов
    centroids = np.array([r.centroid for r in regions if r.area > 100])
    
    if len(centroids) == 0:
        return [{
            'x': bound['x'],
            'y': bound['y'],
            'w': bound['w'],
            'h': bound['h']
        }]
    
    # Кластеризация для определения подсекций
    if num_subsections is None:
        # Автоматическое определение количества подсекций
        n_clusters = min(len(centroids), 4)  # Максимум 4 подсекции
    else:
        n_clusters = min(num_subsections, len(centroids))
    
    if n_clusters <= 1:
        return [{
            'x': bound['x'],
            'y': bound['y'],
            'w': bound['w'],
            'h': bound['h']
        }]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(centroids)
    
    # Создаем bounding boxes для каждой подсекции
    subsection_bounds = []
    for i in range(n_clusters):
        cluster_points = centroids[kmeans.labels_ == i]
        if len(cluster_points) > 0:
            min_x = int(cluster_points[:, 1].min())
            min_y = int(cluster_points[:, 0].min())
            max_x = int(cluster_points[:, 1].max())
            max_y = int(cluster_points[:, 0].max())
            
            # Масштабируем обратно к уровню 0
            scale_x = bound['w'] / thumb_size[0]
            scale_y = bound['h'] / thumb_size[1]
            
            subsection_bounds.append({
                'x': bound['x'] + int(min_x * scale_x),
                'y': bound['y'] + int(min_y * scale_y),
                'w': int((max_x - min_x) * scale_x),
                'h': int((max_y - min_y) * scale_y)
            })
    
    return subsection_bounds if subsection_bounds else [bound]

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
# Предсказание (метод через подсекции)
# ============================================
if wsi_files and loaded_models:
    print("\n" + "=" * 60)
    print("Начало предсказания (метод через подсекции)")
    print("=" * 60)
    
    for wsi_file in wsi_files:
        print(f"\n📁 Обработка: {wsi_file.name}")
        
        try:
            # Загружаем WSI
            wsi = CuImage(str(wsi_file))
            wsi_size = wsi.resolutions["level_dimensions"][0]
            print(f"   Размер WSI: {wsi_size[0]}x{wsi_size[1]}")
            
            # Извлекаем подсекции
            print("   ⏳ Извлечение подсекций...")
            subsection_bounds = extract_subsection_bounds(wsi, section_index=0, num_subsections=None)
            print(f"   ✅ Найдено подсекций: {len(subsection_bounds)}")
            
            all_predictions = defaultdict(list)
            window_size = 640
            overlap_ratio = 0.8
            stride = int(window_size * (1 - overlap_ratio))
            
            # Обрабатываем каждую подсекцию
            for sub_idx, sub_bound in enumerate(subsection_bounds):
                print(f"   ⏳ Обработка подсекции {sub_idx + 1}/{len(subsection_bounds)}...")
                
                x_start, y_start = sub_bound['x'], sub_bound['y']
                w, h = sub_bound['w'], sub_bound['h']
                
                window_count = 0
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        size_x = min(window_size, w - x)
                        size_y = min(window_size, h - y)
                        
                        # Читаем регион
                        region = wsi.read_region(
                            location=(x_start + x, y_start + y),
                            size=(size_x, size_y),
                            level=0,
                        )
                        region_bgr = cv2.cvtColor(np.asarray(region)[..., :3], cv2.COLOR_RGB2BGR)
                        
                        # Предсказание для каждого окна
                        for model_wrapper in loaded_models:
                            config = model_wrapper['config']
                            if size_x == window_size and size_y == window_size:
                                # Только полные окна
                                preds = model_wrapper['model'].predict([region_bgr], verbose=False, conf=config['min_conf'])
                                
                                # Парсим предсказания
                                for pred in preds:
                                    boxes = pred.boxes.xyxy.cpu().numpy()
                                    confs = pred.boxes.conf.cpu().numpy()
                                    cls_indexes = pred.boxes.cls.cpu().numpy().astype(int)
                                    
                                    for box, conf, cls_idx in zip(boxes, confs, cls_indexes):
                                        cls_name = config['classes'].get(cls_idx, f"Class_{cls_idx}")
                                        
                                        # Добавляем смещение координат
                                        pred_with_offset = {
                                            'box': [
                                                float(box[0]) + x_start + x,
                                                float(box[1]) + y_start + y,
                                                float(box[2]) + x_start + x,
                                                float(box[3]) + y_start + y
                                            ],
                                            'conf': float(conf)
                                        }
                                        all_predictions[cls_name].append(pred_with_offset)
                        
                        window_count += 1
                        if window_count % 50 == 0:
                            print(f"      Обработано окон: {window_count}")
                
                print(f"   ✅ Подсекция {sub_idx + 1} обработана ({window_count} окон)")
            
            # Простой NMS для удаления дубликатов (упрощенный)
            print("   ⏳ Применение NMS...")
            final_predictions = defaultdict(list)
            
            for cls_name, preds in all_predictions.items():
                if not preds:
                    continue
                
                # Простой фильтр по IoU (упрощенный)
                # В реальности нужен более сложный NMS
                boxes = np.array([p['box'] for p in preds])
                confs = np.array([p['conf'] for p in preds])
                
                # Сортируем по уверенности
                sorted_indices = np.argsort(confs)[::-1]
                keep = []
                used = set()
                
                for idx in sorted_indices:
                    if idx in used:
                        continue
                    keep.append(idx)
                    box = boxes[idx]
                    
                    # Помечаем пересекающиеся боксы
                    for other_idx, other_box in enumerate(boxes):
                        if other_idx == idx or other_idx in used:
                            continue
                        
                        # Простой IoU
                        x1 = max(box[0], other_box[0])
                        y1 = max(box[1], other_box[1])
                        x2 = min(box[2], other_box[2])
                        y2 = min(box[3], other_box[3])
                        
                        if x2 > x1 and y2 > y1:
                            intersection = (x2 - x1) * (y2 - y1)
                            area1 = (box[2] - box[0]) * (box[3] - box[1])
                            area2 = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
                            union = area1 + area2 - intersection
                            
                            if union > 0:
                                iou = intersection / union
                                if iou > 0.5:  # Порог IoU
                                    used.add(other_idx)
                
                # Сохраняем отфильтрованные предсказания
                for idx in keep:
                    final_predictions[cls_name].append(preds[idx])
            
            # Сохраняем результаты
            wsi_name = wsi_file.stem
            output_file = RESULTS_DIR / f"{wsi_name}.json"
            
            # Конвертируем в формат JSON
            output_data = {}
            for cls_name, preds in final_predictions.items():
                output_data[cls_name] = preds
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"   ✅ Результаты сохранены: {output_file}")
            print(f"   📊 Всего предсказаний: {sum(len(preds) for preds in final_predictions.values())}")
            
        except Exception as e:
            print(f"   ❌ Ошибка при обработке {wsi_file.name}: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ГОТОВО!")
print("=" * 60)
print(f"\n📁 Результаты сохранены в: {RESULTS_DIR}")

