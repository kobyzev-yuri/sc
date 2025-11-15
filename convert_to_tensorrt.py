#!/usr/bin/env python3
"""
Скрипт для конвертации YOLO моделей в формат TensorRT.

TensorRT оптимизирует модели для конкретной GPU архитектуры, что может дать
значительное ускорение инференса (обычно 2-5x быстрее чем PyTorch).

ВАЖНО:
- TensorRT engine файлы специфичны для конкретной GPU и версии TensorRT/CUDA
- Модели нужно конвертировать на той же системе, где они будут использоваться
- Требуется NVIDIA GPU с поддержкой TensorRT

Использование:
    python convert_to_tensorrt.py [--models-dir models/] [--batch-size 1] [--imgsz 640]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def check_tensorrt_installation() -> bool:
    """Проверяет наличие TensorRT."""
    try:
        import tensorrt as trt
        return True
    except ImportError:
        return False


def convert_model_to_tensorrt(
    model_path: Path,
    output_dir: Optional[Path] = None,
    batch_size: int = 1,
    imgsz: int = 640,
    half: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Конвертирует YOLO модель в TensorRT формат.
    
    Args:
        model_path: Путь к .pt файлу модели
        output_dir: Директория для сохранения .engine файла (по умолчанию рядом с моделью)
        batch_size: Размер батча для оптимизации (может влиять на производительность)
        imgsz: Размер изображения для оптимизации
        half: Использовать FP16 точность (быстрее, но может быть менее точно)
        verbose: Выводить подробную информацию
        
    Returns:
        Путь к созданному .engine файлу
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.suffix != '.pt':
        raise ValueError(f"Expected .pt file, got {model_path.suffix}")
    
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Converting {model_path.name} to TensorRT...")
        print(f"  Input: {model_path}")
        print(f"  Output dir: {output_dir}")
        print(f"  Batch size: {batch_size}, Image size: {imgsz}, FP16: {half}")
    
    # Проверяем наличие TensorRT
    if not check_tensorrt_installation():
        if verbose:
            print("  ⚠️  TensorRT не установлен. Ultralytics попытается установить его автоматически...")
            print("  💡 Это может занять несколько минут. Если процесс зависнет, установите TensorRT вручную:")
            print("     pip install tensorrt-cu12")
            print("  ⏳ Ожидание установки TensorRT...")
    
    # Загружаем модель
    model = YOLO(str(model_path))
    
    # Экспортируем в TensorRT
    # Формат будет автоматически определен как 'engine'
    try:
        exported_path = model.export(
            format='engine',
            imgsz=imgsz,
            batch=batch_size,
            half=half,
            verbose=verbose,
        )
        
        # Перемещаем в нужную директорию если нужно
        exported_path_obj = Path(exported_path)
        if output_dir != exported_path_obj.parent:
            target_path = output_dir / exported_path_obj.name
            if target_path.exists():
                target_path.unlink()
            exported_path_obj.rename(target_path)
            exported_path = str(target_path)
        
        if verbose:
            print(f"✓ Successfully converted to: {exported_path}")
        
        return Path(exported_path)
        
    except Exception as e:
        print(f"✗ Error converting {model_path.name}: {e}", file=sys.stderr)
        raise


def convert_all_models(
    models_dir: Path,
    batch_size: int = 1,
    imgsz: int = 640,
    half: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """
    Конвертирует все .pt модели в директории в TensorRT формат.
    
    Args:
        models_dir: Директория с моделями
        batch_size: Размер батча для оптимизации
        imgsz: Размер изображения для оптимизации
        half: Использовать FP16 точность
        verbose: Выводить подробную информацию
        
    Returns:
        Список путей к созданным .engine файлам
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Проверяем TensorRT заранее
    if verbose:
        print("Checking TensorRT installation...")
        if check_tensorrt_installation():
            try:
                import tensorrt as trt
                print(f"✅ TensorRT установлен, версия: {trt.__version__}")
            except:
                print("✅ TensorRT доступен")
        else:
            print("⚠️  TensorRT не установлен")
            print("   Ultralytics попытается установить его автоматически при первой конвертации")
            print("   Это может занять несколько минут и часто не работает.")
            print()
            print("   💡 Рекомендуется установить TensorRT вручную одним из способов:")
            print("   1. Через NVIDIA PyIndex (рекомендуется):")
            print("      pip install nvidia-pyindex")
            print("      pip install nvidia-tensorrt")
            print()
            print("   2. Через conda (если используете conda):")
            print("      conda install -c nvidia tensorrt")
            print()
            print("   3. Через официальный пакет NVIDIA:")
            print("      Скачайте с https://developer.nvidia.com/tensorrt")
            print("      И установите wheel файл из python/ директории")
        print()
    
    # Находим все .pt файлы
    pt_files = list(models_dir.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {models_dir}")
        return []
    
    if verbose:
        print(f"Found {len(pt_files)} model(s) to convert")
        print()
    
    converted = []
    failed = []
    
    for pt_file in pt_files:
        try:
            engine_path = convert_model_to_tensorrt(
                pt_file,
                output_dir=models_dir,
                batch_size=batch_size,
                imgsz=imgsz,
                half=half,
                verbose=verbose,
            )
            converted.append(engine_path)
            if verbose:
                print()
        except Exception as e:
            failed.append((pt_file, str(e)))
            if verbose:
                print()
    
    # Итоговая статистика
    if verbose:
        print("=" * 60)
        print(f"Conversion complete:")
        print(f"  Successfully converted: {len(converted)}")
        print(f"  Failed: {len(failed)}")
        
        if failed:
            print("\nFailed models:")
            for model_path, error in failed:
                print(f"  - {model_path.name}: {error}")
    
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO models to TensorRT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all models in default models/ directory
  python convert_to_tensorrt.py
  
  # Convert models in custom directory
  python convert_to_tensorrt.py --models-dir /path/to/models
  
  # Convert with specific batch size and image size
  python convert_to_tensorrt.py --batch-size 32 --imgsz 640
  
  # Convert single model
  python convert_to_tensorrt.py --model models/nn_det2_data_outputs_meta_train4.pt
        """
    )
    
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path(__file__).parent / 'models',
        help='Directory containing .pt model files (default: models/)'
    )
    
    parser.add_argument(
        '--model',
        type=Path,
        default=None,
        help='Single model file to convert (if not specified, converts all .pt files)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for TensorRT optimization (default: 1)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for TensorRT optimization (default: 640)'
    )
    
    parser.add_argument(
        '--no-half',
        action='store_true',
        help='Use FP32 instead of FP16 (slower but more accurate)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        if args.model:
            # Конвертируем одну модель
            convert_model_to_tensorrt(
                args.model,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                half=not args.no_half,
                verbose=not args.quiet,
            )
        else:
            # Конвертируем все модели
            convert_all_models(
                args.models_dir,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                half=not args.no_half,
                verbose=not args.quiet,
            )
    except KeyboardInterrupt:
        print("\nConversion interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

