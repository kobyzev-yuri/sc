#!/usr/bin/env python3
"""
Утилита для архивации экспериментов без данных.

Перемещает эксперименты, у которых нет данных (aggregated_data, relative_features, all_features),
в директорию experiments/archive/.
"""

import shutil
from pathlib import Path
from typing import List
from datetime import datetime


def has_data(exp_dir: Path) -> bool:
    """
    Проверяет, есть ли данные в эксперименте.
    
    Args:
        exp_dir: Путь к директории эксперимента
        
    Returns:
        True если есть данные, False иначе
    """
    aggregated_files = list(exp_dir.glob("aggregated_data_*.csv"))
    relative_files = list(exp_dir.glob("relative_features_*.csv"))
    all_features_files = list(exp_dir.glob("all_features_*.csv"))
    
    return bool(aggregated_files or relative_files or all_features_files)


def archive_experiments_without_data(
    experiments_dir: Path = Path("experiments"),
    archive_dir: Path = None,
    dry_run: bool = False
) -> List[str]:
    """
    Архивирует эксперименты без данных.
    
    Args:
        experiments_dir: Базовая директория с экспериментами
        archive_dir: Директория для архива (по умолчанию experiments/archive)
        dry_run: Если True, только показывает что будет заархивировано, не перемещает
        
    Returns:
        Список имен заархивированных экспериментов
    """
    if archive_dir is None:
        archive_dir = experiments_dir / "archive"
    
    experiments_dir = Path(experiments_dir)
    archive_dir = Path(archive_dir)
    
    archived = []
    
    # Сканируем все директории в experiments
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Пропускаем архивную директорию
        if exp_dir.name == "archive":
            continue
        
        # Пропускаем, если это не директория эксперимента (нет best_features_*.json)
        json_files = list(exp_dir.glob("best_features_*.json"))
        if not json_files:
            continue
        
        # Проверяем наличие данных
        if not has_data(exp_dir):
            archived.append(exp_dir.name)
            
            if not dry_run:
                # Создаем директорию архива, если её нет
                archive_dir.mkdir(parents=True, exist_ok=True)
                
                # Перемещаем эксперимент в архив
                archive_path = archive_dir / exp_dir.name
                
                # Если уже существует, добавляем timestamp
                if archive_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_path = archive_dir / f"{exp_dir.name}_{timestamp}"
                
                shutil.move(str(exp_dir), str(archive_path))
                print(f"✓ Перемещен в архив: {exp_dir.name} -> {archive_path.name}")
            else:
                print(f"  Будет заархивирован: {exp_dir.name}")
    
    return archived


def main():
    """Главная функция для CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Архивация экспериментов без данных")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Директория с экспериментами (по умолчанию: experiments)"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Директория для архива (по умолчанию: experiments/archive)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать что будет заархивировано, не перемещать"
    )
    
    args = parser.parse_args()
    
    print("🔍 Поиск экспериментов без данных...")
    print()
    
    archived = archive_experiments_without_data(
        experiments_dir=args.experiments_dir,
        archive_dir=args.archive_dir,
        dry_run=args.dry_run
    )
    
    print()
    if args.dry_run:
        print(f"📋 Найдено экспериментов для архивации: {len(archived)}")
        if archived:
            print("\nСписок:")
            for exp_name in archived:
                print(f"  - {exp_name}")
            print("\nЗапустите без --dry-run для архивации.")
    else:
        print(f"✅ Заархивировано экспериментов: {len(archived)}")
        if archived:
            print("\nЗаархивированные эксперименты:")
            for exp_name in archived:
                print(f"  - {exp_name}")


if __name__ == "__main__":
    main()

