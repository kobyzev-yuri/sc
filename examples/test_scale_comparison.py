"""
Расширенный тест для сравнения разных подходов к построению шкалы.

Тестирует:
- PCA Scoring (простая нормализация)
- Spectral Analysis с разными параметрами процентилей
- Сравнение результатов
- Визуализация сравнения
"""

from pathlib import Path
import sys

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scale import aggregate
from scale.scale_comparison import ScaleComparison


def main():
    """Основная функция для тестирования разных подходов к построению шкалы."""
    
    # Путь к предсказаниям
    predictions_dir = project_root / "results" / "predictions"
    
    if not predictions_dir.exists():
        print(f"Директория {predictions_dir} не найдена")
        return
    
    print("="*80)
    print("ТЕСТИРОВАНИЕ РАЗНЫХ ПОДХОДОВ К ПОСТРОЕНИЮ ШКАЛЫ")
    print("="*80)
    
    # Загрузка предсказаний
    print(f"\nЗагрузка предсказаний из {predictions_dir}")
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"Загружено {len(df)} образцов")
    print(f"Колонки: {list(df.columns)[:5]}...")
    
    # Создание относительных признаков
    print("\n" + "="*80)
    print("ПОДГОТОВКА ДАННЫХ")
    print("="*80)
    print("\nСоздание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    df_features = aggregate.select_feature_columns(df_features)
    print(f"Признаков после отбора: {len(df_features.columns)}")
    print(f"Образцов: {len(df_features)}")
    
    # Инициализация сравнения
    comparison = ScaleComparison()
    
    # Тест 1: Простой PCA Scoring
    print("\n" + "="*80)
    print("ТЕСТ 1: PCA SCORING (простая нормализация)")
    print("="*80)
    comparison.test_pca_scoring(df_features, name="pca_simple")
    
    # Тест 2: Spectral Analysis с процентилями по умолчанию (1, 99)
    print("\n" + "="*80)
    print("ТЕСТ 2: SPECTRAL ANALYSIS (процентили 1, 99)")
    print("="*80)
    comparison.test_spectral_analysis(
        df_features,
        name="spectral_p1_p99",
        percentile_low=1.0,
        percentile_high=99.0,
        use_gmm=False
    )
    
    # Тест 3: Spectral Analysis с более широкими процентилями (0.5, 99.5)
    print("\n" + "="*80)
    print("ТЕСТ 3: SPECTRAL ANALYSIS (процентили 0.5, 99.5)")
    print("="*80)
    comparison.test_spectral_analysis(
        df_features,
        name="spectral_p05_p995",
        percentile_low=0.5,
        percentile_high=99.5,
        use_gmm=False
    )
    
    # Тест 4: Spectral Analysis с узкими процентилями (5, 95)
    print("\n" + "="*80)
    print("ТЕСТ 4: SPECTRAL ANALYSIS (процентили 5, 95)")
    print("="*80)
    comparison.test_spectral_analysis(
        df_features,
        name="spectral_p5_p95",
        percentile_low=5.0,
        percentile_high=95.0,
        use_gmm=False
    )
    
    # Тест 5: Spectral Analysis с GMM
    print("\n" + "="*80)
    print("ТЕСТ 5: SPECTRAL ANALYSIS + GMM")
    print("="*80)
    comparison.test_spectral_analysis(
        df_features,
        name="spectral_gmm",
        percentile_low=1.0,
        percentile_high=99.0,
        use_gmm=True
    )
    
    # Сравнение результатов
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    comparison_df = comparison.compare_results()
    print("\nСравнение шкал (первые 5 образцов):")
    print(comparison_df.head())
    
    # Статистика
    print("\n\nСтатистика по методам:")
    stats = comparison.get_statistics()
    print(stats.to_string(index=False))
    
    # Сохранение результатов
    print("\n" + "="*80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    output_dir = project_root / "experiments" / "scale_comparison"
    comparison.save_results(output_dir)
    
    # Визуализация
    print("\n" + "="*80)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ")
    print("="*80)
    viz_path = output_dir / "comparison_plot.png"
    comparison.visualize_comparison(save_path=viz_path)
    
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*80)
    print(f"\nВсе результаты сохранены в: {output_dir}")
    print(f"  - comparison.csv - сравнение всех методов")
    print(f"  - statistics.csv - статистика по методам")
    print(f"  - configs.json - конфигурации экспериментов")
    print(f"  - results/ - отдельные результаты каждого метода")
    print(f"  - models/ - сохраненные модели")
    print(f"  - comparison_plot.png - график сравнения")


if __name__ == "__main__":
    main()


