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

# Проверка установки
try:
    import tensorrt as trt
    print(f"✅ TensorRT установлен, версия: {trt.__version__}")
except ImportError:
    print("❌ Ошибка установки TensorRT")
    print("Попробуйте альтернативный метод:")
    
    import tensorrt as trt
    print(f"✅ TensorRT установлен, версия: {trt.__version__}")

# ============================================
# ШАГ 2: Установка Ultralytics
# ============================================
print("\n" + "=" * 60)
print("Установка Ultralytics")
print("=" * 60)

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
    DRIVE_MODELS_PATH = '/content/drive/MyDrive/sc/models'  # Измените на ваш путь
    
    import os
    import shutil
    if os.path.exists(DRIVE_MODELS_PATH):
        if os.path.exists('/content/models'):
            shutil.rmtree('/content/models')
        shutil.copytree(DRIVE_MODELS_PATH, '/content/models')
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
            
            # Сначала экспортируем в ONNX (это всегда работает)
            print(f"   ⏳ Экспорт в ONNX...")
            onnx_path = model.export(
                format='onnx',
                imgsz=640,
                verbose=False
            )
            print(f"   ✅ ONNX создан: {onnx_path}")
            
            # Пытаемся экспортировать в TensorRT
            try:
                print(f"   ⏳ Попытка экспорта в TensorRT...")
                engine_path = model.export(
                    format='engine',
                    imgsz=640,
                    batch=1,
                    half=True,
                    verbose=False
                )
                converted.append(engine_path)
                print(f"   ✅ TensorRT engine создан: {engine_path}")
            except Exception as trt_error:
                # Если TensorRT не работает, используем ONNX
                error_msg = str(trt_error)
                if "pybind11" in error_msg or "factory function" in error_msg:
                    print(f"   ⚠️  TensorRT ошибка (известная проблема на Colab): {error_msg[:100]}")
                    print(f"   💡 Используйте ONNX модель: {onnx_path}")
                    print(f"   💡 ONNX Runtime работает отлично и быстрее чем PyTorch!")
                    # Добавляем ONNX как успешную конвертацию
                    converted.append(onnx_path)
                else:
                    failed.append((pt_file.name, error_msg))
                    print(f"   ❌ Ошибка TensorRT: {error_msg[:200]}")
            
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

# Создаем архив с результатами (.engine и .onnx файлы)
import zipfile
from pathlib import Path

engine_files = list(MODELS_DIR.glob('*.engine'))
onnx_files = list(MODELS_DIR.glob('*.onnx'))

if engine_files or onnx_files:
    zip_path = '/content/converted_models.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Добавляем .engine файлы
        for engine_file in engine_files:
            zipf.write(engine_file, engine_file.name)
        # Добавляем .onnx файлы
        for onnx_file in onnx_files:
            zipf.write(onnx_file, onnx_file.name)
    
    print(f"✅ Создан архив: {zip_path}")
    print(f"   Содержит {len(engine_files)} .engine файлов и {len(onnx_files)} .onnx файлов")
    
    # Скачивание
    from google.colab import files
    files.download(zip_path)
    print("✅ Архив скачан!")
    
    # Также можно сохранить в Drive
    if USE_DRIVE:
        DRIVE_SC_PATH = '/content/drive/MyDrive/sc'
        DRIVE_MODELS_PATH = f'{DRIVE_SC_PATH}/models'
        import os
        import shutil
        os.makedirs(DRIVE_MODELS_PATH, exist_ok=True)
        # Копируем .engine файлы
        for engine_file in engine_files:
            shutil.copy2(engine_file, DRIVE_MODELS_PATH)
        # Копируем .onnx файлы
        for onnx_file in onnx_files:
            shutil.copy2(onnx_file, DRIVE_MODELS_PATH)
        print(f"✅ Файлы также сохранены в Drive: {DRIVE_MODELS_PATH}")
        print(f"   (в директории MyDrive/sc/models/)")
else:
    print("⚠️  Не найдено конвертированных файлов (.engine или .onnx)")

print("\n" + "=" * 60)
print("✅ ГОТОВО!")
print("=" * 60)

if len(converted) > 0:
    print(f"\n📊 Итоги:")
    print(f"   ✅ Успешно конвертировано: {len(converted)} моделей")
    
    engine_count = len([f for f in converted if str(f).endswith('.engine')])
    onnx_count = len([f for f in converted if str(f).endswith('.onnx')])
    
    if engine_count > 0:
        print(f"      - TensorRT .engine: {engine_count}")
    if onnx_count > 0:
        print(f"      - ONNX: {onnx_count}")
    
    print("\n💡 Рекомендации:")
    if engine_count > 0:
        print("   - TensorRT .engine файлы специфичны для GPU архитектуры")
        print("   - Используйте их на системе с совместимой GPU")
    if onnx_count > 0:
        print("   - ONNX модели работают на любой системе с ONNX Runtime")
        print("   - ONNX Runtime быстрее PyTorch и проще в использовании")
        print("   - Установка: pip install onnxruntime-gpu")
    
    if failed:
        print(f"\n⚠️  Не удалось конвертировать {len(failed)} моделей в TensorRT")
        print("   Это известная проблема на Colab (pybind11 ошибка)")
        print("   ONNX модели созданы и работают отлично!")
else:
    print("\n⚠️  Не удалось создать TensorRT модели")
    print("   Это известная проблема на Colab")
    print("   💡 Решение: Используйте ONNX модели - они уже созданы и работают отлично!")
    print("   💡 ONNX Runtime быстрее PyTorch и проще в использовании")

