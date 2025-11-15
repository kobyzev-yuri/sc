#!/usr/bin/env python3
"""
Пример использования TensorRT оптимизированных YOLO моделей.

Этот скрипт демонстрирует как использовать TensorRT модели для ускорения инференса.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from scale.model_config import (
    create_model_configs,
    create_tensorrt_model_configs,
    get_postprocess_settings,
)


def test_tensorrt_models():
    """Тестирует загрузку TensorRT моделей."""
    print("=" * 60)
    print("Тестирование TensorRT моделей")
    print("=" * 60)
    print()
    
    # Проверяем наличие TensorRT
    try:
        import tensorrt as trt
        print(f"✓ TensorRT установлен, версия: {trt.__version__}")
    except ImportError:
        print("⚠ TensorRT не установлен, будет использован fallback на PyTorch")
    print()
    
    # Создаем конфигурации с TensorRT (с fallback на PyTorch)
    print("Создание конфигураций моделей с TensorRT...")
    try:
        model_configs = create_tensorrt_model_configs(fallback_to_pt=True)
        print(f"✓ Успешно создано {len(model_configs)} конфигураций моделей")
        
        # Проверяем какие модели используют TensorRT
        tensorrt_count = 0
        pytorch_count = 0
        
        for config in model_configs:
            if hasattr(config.model, 'is_tensorrt') and config.model.is_tensorrt:
                tensorrt_count += 1
            else:
                pytorch_count += 1
        
        print(f"  - TensorRT моделей: {tensorrt_count}")
        print(f"  - PyTorch моделей (fallback): {pytorch_count}")
        print()
        
        if tensorrt_count == 0:
            print("⚠ Внимание: Все модели используют PyTorch fallback.")
            print("  Возможно, нужно конвертировать модели в TensorRT формат:")
            print("  python convert_to_tensorrt.py")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка при создании конфигураций: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_configs():
    """Сравнивает обычные и TensorRT конфигурации."""
    print("=" * 60)
    print("Сравнение конфигураций")
    print("=" * 60)
    print()
    
    # Обычные модели
    print("Создание конфигураций с PyTorch моделями...")
    configs_pt = create_model_configs(use_tensorrt=False)
    print(f"✓ Создано {len(configs_pt)} конфигураций")
    print()
    
    # TensorRT модели
    print("Создание конфигураций с TensorRT моделями...")
    configs_trt = create_tensorrt_model_configs(fallback_to_pt=True)
    print(f"✓ Создано {len(configs_trt)} конфигураций")
    print()
    
    # Проверяем что количество совпадает
    if len(configs_pt) == len(configs_trt):
        print("✓ Количество конфигураций совпадает")
    else:
        print(f"⚠ Количество конфигураций не совпадает: {len(configs_pt)} vs {len(configs_trt)}")
    print()


def main():
    """Главная функция."""
    print()
    print("TensorRT Model Test Script")
    print()
    
    # Тест 1: Загрузка TensorRT моделей
    success = test_tensorrt_models()
    
    if not success:
        print("\n⚠ Тест не прошел. Проверьте установку TensorRT и наличие .engine файлов.")
        return
    
    # Тест 2: Сравнение конфигураций
    compare_configs()
    
    print("=" * 60)
    print("Тесты завершены")
    print("=" * 60)
    print()
    print("Для использования TensorRT моделей в предсказаниях:")
    print("  from scale.model_config import create_tensorrt_model_configs")
    print("  model_configs = create_tensorrt_model_configs()")
    print()


if __name__ == '__main__':
    main()

