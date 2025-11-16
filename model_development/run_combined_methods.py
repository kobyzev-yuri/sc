#!/usr/bin/env python3
"""
Скрипт для тестирования комбинированных методов отбора признаков.

Запускает последовательно различные комбинированные подходы:
- MI → Forward Selection
- Forward → Backward Elimination
- Forward ∩ Backward (пересечение)
- Forward ∪ Backward (объединение)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from model_development.feature_selection_automated import (
    FeatureSelector,
    run_feature_selection_analysis,
)
from scale import aggregate


def run_combined_experiment(
    predictions_dir: str,
    experiment_name: str,
    combined_method: str,
    method_params: Optional[dict] = None,
) -> Path:
    """
    Запускает эксперимент с комбинированным методом.
    
    Args:
        predictions_dir: Директория с JSON файлами предсказаний
        experiment_name: Имя эксперимента
        combined_method: Название комбинированного метода
        method_params: Параметры для метода
        
    Returns:
        Путь к директории эксперимента
    """
    print(f"\n{'='*70}")
    print(f"КОМБИНИРОВАННЫЙ ЭКСПЕРИМЕНТ: {experiment_name}")
    print(f"Метод: {combined_method}")
    print(f"{'='*70}")
    
    output_dir = Path("experiments") / experiment_name
    
    # Загружаем данные
    print("\n1. Загрузка данных...")
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"   Загружено образцов: {len(df)}")
    
    # Создание относительных признаков
    print("\n2. Создание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    print(f"   Создано признаков: {len(df_features.columns) - 1}")
    
    # Получение всех доступных признаков
    df_all = aggregate.select_all_feature_columns(df_features)
    candidate_features = [c for c in df_all.columns if c != 'image']
    print(f"   Кандидатных признаков: {len(candidate_features)}")
    
    # Создание селектора
    print("\n3. Инициализация селектора признаков...")
    selector = FeatureSelector(df_all)
    
    # Запускаем комбинированный метод
    print(f"\n4. Запуск комбинированного метода: {combined_method}...")
    
    if method_params is None:
        method_params = {}
    
    if combined_method == 'mi_then_forward':
        features, metrics = selector.method_combined_mi_then_forward(
            candidate_features,
            mi_k=method_params.get('mi_k', 25),
            forward_min_improvement=method_params.get('forward_min_improvement', 0.01)
        )
    elif combined_method == 'forward_then_backward':
        features, metrics = selector.method_combined_forward_then_backward(
            candidate_features,
            forward_max_features=method_params.get('forward_max_features', 30),
            forward_min_improvement=method_params.get('forward_min_improvement', 0.01),
            backward_min_improvement=method_params.get('backward_min_improvement', 0.01)
        )
    elif combined_method == 'forward_backward_intersection':
        features, metrics = selector.method_combined_forward_backward_intersection(
            candidate_features,
            forward_min_improvement=method_params.get('forward_min_improvement', 0.01),
            backward_min_improvement=method_params.get('backward_min_improvement', 0.01)
        )
    elif combined_method == 'forward_backward_union':
        features, metrics = selector.method_combined_forward_backward_union(
            candidate_features,
            forward_min_improvement=method_params.get('forward_min_improvement', 0.01),
            backward_min_improvement=method_params.get('backward_min_improvement', 0.01)
        )
    else:
        raise ValueError(f"Неизвестный комбинированный метод: {combined_method}")
    
    # Выводим результаты
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ КОМБИНИРОВАННОГО МЕТОДА")
    print("="*70)
    print(f"Метод: {combined_method}")
    print(f"Признаков: {len(features)}")
    print(f"Score: {metrics['score']:.4f}")
    print(f"Separation: {metrics['separation']:.4f}")
    print(f"Mod (норм. PC1): {metrics['mean_pc1_norm_mod']:.4f}")
    print(f"Объясненная дисперсия: {metrics['explained_variance']:.4f}")
    
    # Сохраняем результаты
    print("\n5. Экспорт результатов...")
    from model_development import feature_selection_export
    from scale import spectral_analysis
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем SpectralAnalyzer и результаты спектрального анализа
    analyzer = spectral_analysis.SpectralAnalyzer()
    analyzer.fit_pca(df_all[features + ['image']])
    df_pca = analyzer.transform_pca(df_all[features + ['image']])
    analyzer.fit_spectrum(df_pca)
    results_df = analyzer.transform_to_spectrum(df_pca)
    
    # Экспортируем в формат experiments
    feature_selection_export.export_to_experiment_format(
        selected_features=features,
        output_dir=output_dir,
        method_name=combined_method,
        metrics=metrics,
        df_results=results_df,
        analyzer=analyzer,
        use_relative_features=True,
        metadata={
            'train_set': str(predictions_dir),
            'aggregation_version': 'current',
            'combined_method': combined_method,
            'method_params': method_params,
        }
    )
    
    print(f"\n✅ Эксперимент завершен: {output_dir}")
    print(f"📊 Результаты:")
    print(f"   Метод: {combined_method}")
    print(f"   Score: {metrics['score']:.4f}")
    print(f"   Separation: {metrics['separation']:.4f}")
    print(f"   Mod (норм. PC1): {metrics['mean_pc1_norm_mod']:.4f}")
    print(f"   Признаков: {len(features)}")
    
    return output_dir


def run_phase_2_combined_methods(predictions_dir: str):
    """Фаза 2.2: Комбинированные методы"""
    print("\n" + "="*70)
    print("ФАЗА 2.2: КОМБИНИРОВАННЫЕ МЕТОДЫ")
    print("="*70)
    
    # 1. MI → Forward Selection
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 1: MI → Forward Selection")
    print("="*70)
    run_combined_experiment(
        predictions_dir=predictions_dir,
        experiment_name="fs_mi_then_forward_k25",
        combined_method="mi_then_forward",
        method_params={'mi_k': 25, 'forward_min_improvement': 0.01}
    )
    
    # 2. Forward → Backward
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 2: Forward → Backward")
    print("="*70)
    run_combined_experiment(
        predictions_dir=predictions_dir,
        experiment_name="fs_forward_then_backward",
        combined_method="forward_then_backward",
        method_params={'forward_max_features': 30, 'forward_min_improvement': 0.01, 'backward_min_improvement': 0.01}
    )
    
    # 3. Forward ∩ Backward (пересечение)
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 3: Forward ∩ Backward (пересечение)")
    print("="*70)
    run_combined_experiment(
        predictions_dir=predictions_dir,
        experiment_name="fs_forward_backward_intersection",
        combined_method="forward_backward_intersection",
        method_params={'forward_min_improvement': 0.01, 'backward_min_improvement': 0.01}
    )
    
    # 4. Forward ∪ Backward (объединение)
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 4: Forward ∪ Backward (объединение)")
    print("="*70)
    run_combined_experiment(
        predictions_dir=predictions_dir,
        experiment_name="fs_forward_backward_union",
        combined_method="forward_backward_union",
        method_params={'forward_min_improvement': 0.01, 'backward_min_improvement': 0.01}
    )
    
    print("\n" + "="*70)
    print("ВСЕ КОМБИНИРОВАННЫЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("="*70)


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование комбинированных методов отбора признаков")
    parser.add_argument("predictions_dir", nargs="?", default="results/predictions",
                       help="Директория с JSON файлами предсказаний")
    
    args = parser.parse_args()
    predictions_dir = args.predictions_dir
    
    print("="*70)
    print("КОМБИНИРОВАННЫЕ МЕТОДЫ ОТБОРА ПРИЗНАКОВ")
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
    
    # Запускаем комбинированные методы
    run_phase_2_combined_methods(predictions_dir)
    
    print("\n" + "="*70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("="*70)
    print(f"Время окончания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

