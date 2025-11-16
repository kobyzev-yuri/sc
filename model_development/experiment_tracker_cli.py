#!/usr/bin/env python3
"""
CLI утилита для работы с системой отслеживания экспериментов.

Позволяет:
- Просматривать список всех экспериментов
- Находить лучшие эксперименты по метрикам
- Сравнивать эксперименты
- Экспортировать сводные отчеты
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

from model_development.experiment_tracker import (
    ExperimentTracker,
    register_experiment_from_directory,
)


def cmd_list(args):
    """Команда: список экспериментов"""
    tracker = ExperimentTracker(args.experiments_dir)
    
    df = tracker.list_experiments(
        sort_by=args.sort_by,
        limit=args.limit,
        filter_by=args.filter if args.filter else None,
    )
    
    if len(df) == 0:
        print("Эксперименты не найдены.")
        return
    
    print("\n" + "="*100)
    print("СПИСОК ЭКСПЕРИМЕНТОВ")
    print("="*100)
    print(df.to_string(index=False))
    print(f"\nВсего экспериментов: {len(df)}")


def cmd_best(args):
    """Команда: лучшие эксперименты"""
    tracker = ExperimentTracker(args.experiments_dir)
    
    best_exps = tracker.get_best_experiments()
    
    if not best_exps:
        print("Лучшие эксперименты не найдены.")
        return
    
    print("\n" + "="*100)
    print("ЛУЧШИЕ ЭКСПЕРИМЕНТЫ")
    print("="*100)
    
    if "best_score" in best_exps:
        exp = best_exps["best_score"]
        print("\n🏆 Лучший по Score (комплексная оценка):")
        print(f"   Эксперимент: {exp['name']}")
        print(f"   Метод: {exp['parameters']['method']}")
        print(f"   Score: {exp['metrics']['score']:.4f}")
        print(f"   Separation: {exp['metrics']['separation']:.4f}")
        print(f"   Mod (норм. PC1): {exp['metrics']['mean_pc1_norm_mod']:.4f}")
        print(f"   Объясненная дисперсия: {exp['metrics']['explained_variance']:.4f}")
        print(f"   Признаков: {exp['parameters']['n_features']}")
        print(f"   Train set: {exp['metadata'].get('train_set', 'unknown')}")
        print(f"   Директория: {exp['directory']}")
    
    if "best_separation" in best_exps:
        exp = best_exps["best_separation"]
        print("\n🎯 Лучший по Separation:")
        print(f"   Эксперимент: {exp['name']}")
        print(f"   Separation: {exp['metrics']['separation']:.4f}")
        print(f"   Score: {exp['metrics']['score']:.4f}")
    
    if "best_mod_position" in best_exps:
        exp = best_exps["best_mod_position"]
        print("\n📊 Лучший по Mod позиции:")
        print(f"   Эксперимент: {exp['name']}")
        print(f"   Mod (норм. PC1): {exp['metrics']['mean_pc1_norm_mod']:.4f}")
        print(f"   Score: {exp['metrics']['score']:.4f}")


def cmd_compare(args):
    """Команда: сравнение экспериментов"""
    tracker = ExperimentTracker(args.experiments_dir)
    
    # Если указаны имена, находим их ID
    exp_ids = []
    for exp_name in args.experiments:
        # Ищем эксперимент по имени
        df = tracker.list_experiments()
        matching = df[df['name'] == exp_name]
        if len(matching) > 0:
            exp_ids.append(matching.iloc[0]['id'])
        else:
            # Пробуем как ID
            exp_ids.append(exp_name)
    
    comparison_df = tracker.compare_experiments(exp_ids)
    
    if len(comparison_df) == 0:
        print("Не найдено экспериментов для сравнения.")
        return
    
    print("\n" + "="*100)
    print("СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
    print("="*100)
    print(comparison_df.to_string(index=False))


def cmd_register(args):
    """Команда: регистрация существующего эксперимента"""
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        print(f"❌ Ошибка: директория {experiment_dir} не найдена")
        sys.exit(1)
    
    tracker = ExperimentTracker(args.experiments_dir)
    
    exp_id = register_experiment_from_directory(
        experiment_dir=experiment_dir,
        tracker=tracker,
        train_set=args.train_set,
        aggregation_version=args.aggregation_version,
    )
    
    print(f"\n✅ Эксперимент зарегистрирован (ID: {exp_id})")


def cmd_summary(args):
    """Команда: сводный отчет"""
    tracker = ExperimentTracker(args.experiments_dir)
    
    output_path = Path(args.output) if args.output else None
    report_path = tracker.export_summary_report(output_path)
    
    print(f"\n✅ Сводный отчет сохранен: {report_path}")


def cmd_top(args):
    """Команда: топ-N экспериментов"""
    tracker = ExperimentTracker(args.experiments_dir)
    
    df = tracker.list_experiments(
        sort_by=args.sort_by,
        limit=args.n,
    )
    
    if len(df) == 0:
        print("Эксперименты не найдены.")
        return
    
    print(f"\n{'='*100}")
    print(f"ТОП-{args.n} ЭКСПЕРИМЕНТОВ (сортировка по {args.sort_by})")
    print("="*100)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Система отслеживания экспериментов для поиска лучших моделей"
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Директория с экспериментами (по умолчанию: experiments)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команды")
    
    # Команда: list
    parser_list = subparsers.add_parser("list", help="Список всех экспериментов")
    parser_list.add_argument("--sort-by", default="score", help="Поле для сортировки")
    parser_list.add_argument("--limit", type=int, help="Максимальное число экспериментов")
    parser_list.add_argument("--filter", type=str, help="Фильтр (формат: key=value)")
    
    # Команда: best
    parser_best = subparsers.add_parser("best", help="Лучшие эксперименты по метрикам")
    
    # Команда: compare
    parser_compare = subparsers.add_parser("compare", help="Сравнение экспериментов")
    parser_compare.add_argument("experiments", nargs="+", help="Имена или ID экспериментов")
    
    # Команда: register
    parser_register = subparsers.add_parser("register", help="Регистрация существующего эксперимента")
    parser_register.add_argument("experiment_dir", type=str, help="Путь к директории эксперимента")
    parser_register.add_argument("--train-set", type=str, help="Путь к train set")
    parser_register.add_argument("--aggregation-version", type=str, help="Версия агрегации")
    
    # Команда: summary
    parser_summary = subparsers.add_parser("summary", help="Сводный отчет")
    parser_summary.add_argument("--output", type=str, help="Путь для сохранения отчета")
    
    # Команда: top
    parser_top = subparsers.add_parser("top", help="Топ-N экспериментов")
    parser_top.add_argument("--n", type=int, default=10, help="Число экспериментов")
    parser_top.add_argument("--sort-by", default="score", help="Поле для сортировки")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Парсим фильтр
    if hasattr(args, 'filter') and args.filter:
        filter_dict = {}
        for item in args.filter.split(','):
            if '=' in item:
                key, value = item.split('=', 1)
                filter_dict[key.strip()] = value.strip()
        args.filter = filter_dict
    
    # Выполняем команду
    if args.command == "list":
        cmd_list(args)
    elif args.command == "best":
        cmd_best(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "register":
        cmd_register(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "top":
        cmd_top(args)


if __name__ == "__main__":
    main()

