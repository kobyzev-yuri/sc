#!/usr/bin/env python3
"""
CLI утилита для управления версиями результатов подбора признаков.

Позволяет:
- Просматривать список экспериментов
- Сравнивать эксперименты
- Экспортировать конкретный эксперимент в dashboard
- Находить лучший эксперимент по метрикам
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Добавляем путь к проекту для импорта
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model_development.feature_selection_versioning import (
    FeatureSelectionVersionManager,
    list_all_experiments,
    export_experiment_to_dashboard,
)


def cmd_list(args):
    """Команда: список экспериментов"""
    manager = FeatureSelectionVersionManager(args.experiments_dir)
    df = manager.list_experiments(status=args.status, tags=args.tags)
    
    if len(df) == 0:
        print("Эксперименты не найдены.")
        return
    
    print("\n" + "="*80)
    print("СПИСОК ЭКСПЕРИМЕНТОВ")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)


def cmd_compare(args):
    """Команда: сравнение экспериментов"""
    manager = FeatureSelectionVersionManager(args.experiments_dir)
    df = manager.compare_experiments(args.experiments)
    
    if len(df) == 0:
        print("Эксперименты не найдены.")
        return
    
    print("\n" + "="*80)
    print("СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    
    # Выделяем лучший по score
    if len(df) > 0:
        best = df.loc[df['score'].idxmax()]
        print(f"\n🏆 Лучший по score: {best['experiment']} (score={best['score']:.4f})")


def cmd_export(args):
    """Команда: экспорт эксперимента в dashboard"""
    try:
        manager = FeatureSelectionVersionManager(args.experiments_dir)
        dashboard_path = manager.export_to_dashboard(
            args.experiment,
            backup_current=args.backup,
        )
        
        print("\n" + "="*80)
        print("ЭКСПОРТ ЗАВЕРШЕН")
        print("="*80)
        print(f"✓ Эксперимент '{args.experiment}' экспортирован в dashboard")
        print(f"✓ Конфигурация: {dashboard_path}")
        if args.backup:
            print("✓ Резервная копия предыдущей конфигурации создана")
        print("\n💡 При следующем запуске dashboard будут использованы признаки из этого эксперимента")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


def cmd_best(args):
    """Команда: найти лучший эксперимент"""
    manager = FeatureSelectionVersionManager(args.experiments_dir)
    
    best_exp = manager.get_best_experiment(metric=args.metric)
    
    if best_exp is None:
        print("Завершенные эксперименты не найдены.")
        return
    
    print("\n" + "="*80)
    print(f"ЛУЧШИЙ ЭКСПЕРИМЕНТ (по {args.metric})")
    print("="*80)
    
    exp_data = manager.metadata[best_exp]
    print(f"\nЭксперимент: {best_exp}")
    print(f"Метод: {exp_data.get('best_method', 'unknown')}")
    print(f"Score: {exp_data.get('metrics', {}).get('score', 0):.4f}")
    print(f"Separation: {exp_data.get('metrics', {}).get('separation', 0):.4f}")
    print(f"Mod (норм. PC1): {exp_data.get('metrics', {}).get('mean_pc1_norm_mod', 0):.4f}")
    print(f"Объясненная дисперсия: {exp_data.get('metrics', {}).get('explained_variance', 0):.4f}")
    print(f"Количество признаков: {exp_data.get('n_features', 0)}")
    
    if args.export:
        print("\n" + "="*80)
        print("ЭКСПОРТ В DASHBOARD")
        print("="*80)
        manager.export_to_dashboard(best_exp, backup_current=True)
        print(f"\n✓ Лучший эксперимент экспортирован в dashboard")


def main():
    parser = argparse.ArgumentParser(
        description="Утилита для управления версиями результатов подбора признаков",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Показать список всех экспериментов
  python3 -m scale.feature_selection_versioning_cli list

  # Показать только завершенные эксперименты
  python3 -m scale.feature_selection_versioning_cli list --status completed

  # Сравнить два эксперимента
  python3 -m scale.feature_selection_versioning_cli compare exp1 exp2

  # Экспортировать эксперимент в dashboard
  python3 -m scale.feature_selection_versioning_cli export experiment_20251116_134939

  # Найти лучший эксперимент и экспортировать его
  python3 -m scale.feature_selection_versioning_cli best --export
        """
    )
    
    parser.add_argument(
        '--experiments-dir',
        type=Path,
        default=Path("experiments/feature_selection"),
        help='Директория с экспериментами (по умолчанию: experiments/feature_selection)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Команда list
    parser_list = subparsers.add_parser('list', help='Показать список экспериментов')
    parser_list.add_argument('--status', choices=['completed', 'in_progress'], help='Фильтр по статусу')
    parser_list.add_argument('--tags', nargs='+', help='Фильтр по тегам')
    parser_list.set_defaults(func=cmd_list)
    
    # Команда compare
    parser_compare = subparsers.add_parser('compare', help='Сравнить эксперименты')
    parser_compare.add_argument('experiments', nargs='+', help='Имена экспериментов для сравнения')
    parser_compare.set_defaults(func=cmd_compare)
    
    # Команда export
    parser_export = subparsers.add_parser('export', help='Экспортировать эксперимент в dashboard')
    parser_export.add_argument('experiment', help='Имя эксперимента для экспорта')
    parser_export.add_argument('--no-backup', dest='backup', action='store_false', 
                             help='Не создавать резервную копию текущей конфигурации')
    parser_export.set_defaults(func=cmd_export)
    
    # Команда best
    parser_best = subparsers.add_parser('best', help='Найти лучший эксперимент')
    parser_best.add_argument('--metric', choices=['score', 'separation', 'mod_norm'], 
                            default='score', help='Метрика для сравнения (по умолчанию: score)')
    parser_best.add_argument('--export', action='store_true', 
                            help='Автоматически экспортировать лучший эксперимент в dashboard')
    parser_best.set_defaults(func=cmd_best)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()


