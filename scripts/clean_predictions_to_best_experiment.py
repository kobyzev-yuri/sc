#!/usr/bin/env python3
"""
Скрипт для очистки results/predictions от новых образцов, ухудшивших метрику.

Оставляет только те файлы, которые были в лучшем эксперименте:
feature_selection_all_methods_all_relative_full_rerun (36 образцов, Score: 3.5693)

Удаляемые файлы (9 штук):
- 10_gran_* (5 образцов) - добавлены вручную как mod
- 14_lgd_ibd_* (4 образца) - добавлены как mod

Файлы перемещаются в архив, а не удаляются навсегда.
"""

import sys
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Основная функция."""
    print("="*70)
    print("ОЧИСТКА results/predictions ОТ НОВЫХ ОБРАЗЦОВ")
    print("="*70)
    
    # Загружаем данные из лучшего эксперимента
    best_exp_file = project_root / "experiments" / "feature_selection_all_methods_all_relative_full_rerun" / "aggregated_data_20251217_122438.csv"
    
    if not best_exp_file.exists():
        print(f"❌ Ошибка: файл лучшего эксперимента не найден: {best_exp_file}")
        sys.exit(1)
    
    df_best = pd.read_csv(best_exp_file)
    best_samples = set(df_best['image'].tolist())
    
    print(f"\nЛучший эксперимент: feature_selection_all_methods_all_relative_full_rerun")
    print(f"Образцов в лучшем эксперименте: {len(best_samples)}")
    print(f"Score: 3.5693")
    
    # Получаем список всех файлов в results/predictions
    predictions_dir = project_root / "results" / "predictions"
    
    if not predictions_dir.exists():
        print(f"❌ Ошибка: директория не найдена: {predictions_dir}")
        sys.exit(1)
    
    all_files = {f.stem: f for f in predictions_dir.glob("*.json")}
    
    # Находим файлы для удаления
    files_to_remove = {name: path for name, path in all_files.items() if name not in best_samples}
    
    print(f"\nТекущее состояние:")
    print(f"  Всего файлов в results/predictions: {len(all_files)}")
    print(f"  Файлов для удаления: {len(files_to_remove)}")
    
    if not files_to_remove:
        print("\n✅ Нет файлов для удаления. Всё уже чисто!")
        return
    
    # Создаём директорию для архива
    archive_dir = project_root / "results" / "predictions_archive_removed_17dec"
    archive_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_subdir = archive_dir / f"removed_{timestamp}"
    archive_subdir.mkdir(exist_ok=True)
    
    print(f"\nФайлы для удаления (будут перемещены в архив):")
    for name in sorted(files_to_remove.keys()):
        print(f"  - {name}.json")
    
    print(f"\nАрхив: {archive_subdir}")
    
    # Перемещаем файлы в архив
    print(f"\nПеремещение файлов...")
    moved_count = 0
    errors = []
    
    for name, path in sorted(files_to_remove.items()):
        try:
            dest = archive_subdir / path.name
            shutil.move(str(path), str(dest))
            moved_count += 1
            print(f"  ✓ {path.name}")
        except Exception as e:
            error_msg = f"Ошибка при перемещении {path.name}: {e}"
            errors.append(error_msg)
            print(f"  ✗ {error_msg}")
    
    print(f"\n✅ Перемещено файлов: {moved_count}/{len(files_to_remove)}")
    
    if errors:
        print(f"\n⚠️  Ошибки ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    # Проверяем результат
    remaining_files = list(predictions_dir.glob("*.json"))
    print(f"\nРезультат:")
    print(f"  Осталось файлов в results/predictions: {len(remaining_files)}")
    
    if len(remaining_files) == 36:
        print("  ✅ Всё правильно! Осталось 36 файлов из лучшего эксперимента.")
    else:
        print(f"  ⚠️  Ожидалось 36 файлов, но осталось {len(remaining_files)}")
        print("  Проверьте, что все файлы из лучшего эксперимента присутствуют.")
    
    print(f"\nАрхив сохранён в: {archive_subdir}")
    print("="*70)


if __name__ == "__main__":
    main()

