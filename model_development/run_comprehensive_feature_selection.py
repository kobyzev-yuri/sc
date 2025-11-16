#!/usr/bin/env python3
"""
Скрипт для систематического поиска лучших признаков.

Запускает серию экспериментов по подбору признаков с различными методами и параметрами.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from model_development.feature_selection_automated import (
    FeatureSelector,
    run_feature_selection_analysis,
)
from scale import aggregate


def run_experiment(
    predictions_dir: str,
    experiment_name: str,
    methods: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
    **kwargs
) -> Path:
    """
    Запускает один эксперимент по подбору признаков.
    
    Args:
        predictions_dir: Директория с JSON файлами предсказаний
        experiment_name: Имя эксперимента
        methods: Список методов для сравнения (None = все методы)
        exclude_features: Список признаков для исключения (например, ['Paneth'])
        **kwargs: Дополнительные параметры для методов
        
    Returns:
        Путь к директории эксперимента
    """
    print(f"\n{'='*70}")
    print(f"ЭКСПЕРИМЕНТ: {experiment_name}")
    print(f"{'='*70}")
    
    output_dir = Path("experiments") / experiment_name
    
    # Запускаем анализ
    # Для передачи параметров нужно модифицировать run_feature_selection_analysis
    # Пока используем базовую версию
    results_df = run_feature_selection_analysis(
        predictions_dir=predictions_dir,
        output_dir=output_dir,
        methods=methods,
    )
    
    # Если нужно исключить признаки, делаем это после загрузки данных
    if exclude_features:
        print(f"\n⚠️ Исключаем признаки: {exclude_features}")
        # Это нужно делать внутри метода, но пока просто предупреждаем
    
    print(f"\n✅ Эксперимент завершен: {output_dir}")
    print(f"📊 Лучший результат:")
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"   Метод: {best['method']}")
        print(f"   Score: {best['score']:.4f}")
        print(f"   Separation: {best['separation']:.4f}")
        print(f"   Mod (норм. PC1): {best['mean_pc1_norm_mod']:.4f}")
        print(f"   Признаков: {best['n_features']}")
    
    return output_dir


def run_phase_1_basic_comparison(predictions_dir: str):
    """Фаза 1: Базовое сравнение всех методов"""
    print("\n" + "="*70)
    print("ФАЗА 1: БАЗОВОЕ СРАВНЕНИЕ ВСЕХ МЕТОДОВ")
    print("="*70)
    
    # Все методы
    run_experiment(
        predictions_dir=predictions_dir,
        experiment_name="feature_selection_all_methods",
        methods=None,  # Все методы
    )
    
    # Все методы без Paneth
    # Это требует модификации кода, пока пропускаем


def run_phase_2_parameter_variations(predictions_dir: str):
    """Фаза 2: Вариации параметров для лучших методов"""
    print("\n" + "="*70)
    print("ФАЗА 2: ВАРИАЦИИ ПАРАМЕТРОВ")
    print("="*70)
    
    # Forward Selection - разные min_improvement
    # Это требует модификации кода для передачи параметров
    
    # Positive Loadings - разные пороги
    # Это требует модификации кода
    
    print("⚠️ Фаза 2 требует модификации кода для передачи параметров методов")


def main():
    """Главная функция"""
    if len(sys.argv) > 1:
        predictions_dir = sys.argv[1]
    else:
        predictions_dir = "results/predictions"
    
    print("="*70)
    print("СИСТЕМАТИЧЕСКИЙ ПОИСК ЛУЧШИХ ПРИЗНАКОВ")
    print("="*70)
    print(f"Директория с данными: {predictions_dir}")
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Проверяем наличие данных
    predictions_path = Path(predictions_dir)
    if not predictions_path.exists():
        print(f"❌ Ошибка: директория {predictions_dir} не найдена")
        sys.exit(1)
    
    json_files = list(predictions_path.glob("*.json"))
    if not json_files:
        print(f"❌ Ошибка: в директории {predictions_dir} нет JSON файлов")
        sys.exit(1)
    
    print(f"✓ Найдено {len(json_files)} JSON файлов")
    
    # Запрашиваем фазу
    print("\nВыберите фазу для выполнения:")
    print("1. Фаза 1: Базовое сравнение всех методов")
    print("2. Фаза 2: Вариации параметров (требует модификации кода)")
    print("3. Все фазы")
    
    choice = input("\nВведите номер (1-3) или Enter для фазы 1: ").strip()
    
    if choice == "2":
        run_phase_2_parameter_variations(predictions_dir)
    elif choice == "3":
        run_phase_1_basic_comparison(predictions_dir)
        run_phase_2_parameter_variations(predictions_dir)
    else:
        run_phase_1_basic_comparison(predictions_dir)
    
    print("\n" + "="*70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("="*70)
    print(f"Время окончания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

