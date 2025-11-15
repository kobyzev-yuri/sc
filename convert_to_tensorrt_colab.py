"""
Скрипт для конвертации YOLO моделей в TensorRT на Google Colab.

Использование:
1. Откройте Google Colab: https://colab.research.google.com/
2. Включите GPU: Runtime → Change runtime type → GPU
3. Скопируйте и выполните этот скрипт
4. Загрузите ваши .pt модели (через Drive или интерфейс Colab)
5. Запустите конвертацию
6. Скачайте .engine файлы
"""

# ============================================
# ШАГ 1: Установка TensorRT
# ============================================
print("=" * 60)
print("Установка TensorRT на Google Colab")
print("=" * 60)

# Установка через pip (работает на Colab)
!pip install nvidia-pyindex -q
!pip install nvidia-tensorrt -q

# Проверка установки
try:
    import tensorrt as trt
    print(f"✅ TensorRT установлен, версия: {trt.__version__}")
except ImportError:
    print("❌ Ошибка установки TensorRT")
    print("Попробуйте альтернативный метод:")
    !sudo apt-get update -q
    !sudo apt-get install -y python3-libnvinfer-dev -q
    !pip install nvidia-pyindex -q
    !pip install nvidia-tensorrt -q
    
    import tensorrt as trt
    print(f"✅ TensorRT установлен, версия: {trt.__version__}")

# ============================================
# ШАГ 2: Установка Ultralytics
# ============================================
print("\n" + "=" * 60)
print("Установка Ultralytics")
print("=" * 60)

!pip install ultralytics -q
print("✅ Ultralytics установлен")

# ============================================
# ШАГ 3: Проверка GPU
# ============================================
print("\n" + "=" * 60)
print("Проверка GPU")
print("=" * 60)

import torch
if torch.cuda.is_available():
    print(f"✅ CUDA доступна")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA версия: {torch.version.cuda}")
else:
    print("❌ CUDA не доступна!")
    print("   Убедитесь что GPU включен: Runtime → Change runtime type → GPU")

# ============================================
# ШАГ 4: Загрузка моделей
# ============================================
print("\n" + "=" * 60)
print("Загрузка моделей")
print("=" * 60)

# Вариант 1: Через Google Drive
USE_DRIVE = True  # Измените на False если хотите загружать через интерфейс

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Укажите путь к вашим моделям в Drive
    DRIVE_MODELS_PATH = '/content/drive/MyDrive/models'  # Измените на ваш путь
    
    import os
    if os.path.exists(DRIVE_MODELS_PATH):
        !cp -r {DRIVE_MODELS_PATH} /content/models
        print(f"✅ Модели скопированы из Drive: {DRIVE_MODELS_PATH}")
    else:
        print(f"⚠️  Путь не найден: {DRIVE_MODELS_PATH}")
        print("   Загрузите модели через интерфейс Colab (Files → Upload)")
        print("   Или измените DRIVE_MODELS_PATH на правильный путь")
else:
    print("📁 Загрузите .pt файлы через интерфейс Colab:")
    print("   Files → Upload → выберите .pt файлы")
    print("   Затем укажите путь к моделям в переменной MODELS_DIR ниже")

# ============================================
# ШАГ 5: Конвертация моделей
# ============================================
print("\n" + "=" * 60)
print("Конвертация моделей в TensorRT")
print("=" * 60)

from ultralytics import YOLO
from pathlib import Path
import os

# Укажите путь к моделям
MODELS_DIR = Path('/content/models')  # Измените если нужно

# Находим все .pt файлы
pt_files = list(MODELS_DIR.glob('*.pt'))

if not pt_files:
    print(f"❌ Не найдено .pt файлов в {MODELS_DIR}")
    print("   Убедитесь что модели загружены правильно")
else:
    print(f"✅ Найдено {len(pt_files)} моделей для конвертации:")
    for pt_file in pt_files:
        print(f"   - {pt_file.name}")
    
    print("\n⏳ Начинаю конвертацию...")
    
    converted = []
    failed = []
    
    for pt_file in pt_files:
        try:
            print(f"\n📦 Конвертирую {pt_file.name}...")
            model = YOLO(str(pt_file))
            
            engine_path = model.export(
                format='engine',
                imgsz=640,  # Измените если нужно
                batch=1,    # Измените если нужно
                half=True,  # FP16 для ускорения
                verbose=True
            )
            
            converted.append(engine_path)
            print(f"✅ Успешно: {engine_path}")
            
        except Exception as e:
            failed.append((pt_file.name, str(e)))
            print(f"❌ Ошибка при конвертации {pt_file.name}: {e}")
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ КОНВЕРТАЦИИ")
    print("=" * 60)
    print(f"✅ Успешно: {len(converted)}")
    print(f"❌ Ошибок: {len(failed)}")
    
    if failed:
        print("\nОшибки:")
        for name, error in failed:
            print(f"   - {name}: {error}")

# ============================================
# ШАГ 6: Скачивание результатов
# ============================================
print("\n" + "=" * 60)
print("Скачивание результатов")
print("=" * 60)

# Создаем архив с .engine файлами
import zipfile
from pathlib import Path

engine_files = list(MODELS_DIR.glob('*.engine'))

if engine_files:
    zip_path = '/content/tensorrt_engines.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for engine_file in engine_files:
            zipf.write(engine_file, engine_file.name)
    
    print(f"✅ Создан архив: {zip_path}")
    print(f"   Содержит {len(engine_files)} .engine файлов")
    
    # Скачивание
    from google.colab import files
    files.download(zip_path)
    print("✅ Архив скачан!")
    
    # Также можно сохранить в Drive
    if USE_DRIVE:
        DRIVE_OUTPUT_PATH = '/content/drive/MyDrive/tensorrt_engines'
        !mkdir -p {DRIVE_OUTPUT_PATH}
        !cp /content/models/*.engine {DRIVE_OUTPUT_PATH}/
        print(f"✅ Файлы также сохранены в Drive: {DRIVE_OUTPUT_PATH}")
else:
    print("⚠️  Не найдено .engine файлов для скачивания")

print("\n" + "=" * 60)
print("✅ ГОТОВО!")
print("=" * 60)
print("\n💡 Рекомендации:")
print("   - Сохраните .engine файлы в безопасное место")
print("   - Помните: .engine файлы специфичны для GPU архитектуры")
print("   - Используйте их на системе с совместимой GPU или продолжайте использовать PyTorch модели")

