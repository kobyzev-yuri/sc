"""
Пример использования предсказаний из results/predictions для тестирования.

Демонстрирует:
- Загрузку предсказаний из results/predictions
- Агрегацию данных
- Спектральный анализ
- Визуализацию результатов
"""

from pathlib import Path
import sys

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scale import aggregate, spectral_analysis


def main():
    """Основная функция для тестирования с данными из results/predictions."""
    
    # Путь к предсказаниям
    predictions_dir = project_root / "results" / "predictions"
    
    if not predictions_dir.exists():
        print(f"Директория {predictions_dir} не найдена")
        return
    
    print(f"Загрузка предсказаний из {predictions_dir}")
    
    # Загрузка всех предсказаний
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"Загружено {len(df)} образцов")
    print(f"Колонки: {list(df.columns)[:5]}...")
    
    # Создание относительных признаков
    print("\nСоздание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    df_features = aggregate.select_feature_columns(df_features)
    print(f"Признаков после отбора: {len(df_features.columns)}")
    
    # Спектральный анализ
    print("\nОбучение спектрального анализатора...")
    analyzer = spectral_analysis.SpectralAnalyzer()
    
    # PCA
    analyzer.fit_pca(df_features)
    df_pca = analyzer.transform_pca(df_features)
    print(f"PC1 диапазон: [{df_pca['PC1'].min():.2f}, {df_pca['PC1'].max():.2f}]")
    
    # Анализ спектра
    analyzer.fit_spectrum(df_pca, percentile_low=1.0, percentile_high=99.0)
    spectrum_info = analyzer.get_spectrum_info()
    
    print(f"\nСпектральный анализ:")
    print(f"  Число мод: {spectrum_info['n_modes']}")
    print(f"  PC1 медиана: {spectrum_info['percentiles']['median']:.2f}")
    print(f"  PC1 std: {spectrum_info['percentiles']['std']:.2f}")
    
    # Преобразование в спектральную шкалу
    df_spectrum = analyzer.transform_to_spectrum(df_pca)
    print(f"\nСпектральная шкала:")
    print(f"  Диапазон PC1_spectrum: [{df_spectrum['PC1_spectrum'].min():.2f}, {df_spectrum['PC1_spectrum'].max():.2f}]")
    
    # Топ важных признаков
    feature_importance = analyzer.get_feature_importance()
    print(f"\nТоп-5 важных признаков:")
    for i, (feature, importance) in enumerate(feature_importance.head(5).items(), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Сохранение результатов
    output_dir = project_root / "experiments" / "test_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "results.csv"
    df_spectrum.to_csv(csv_path, index=False)
    print(f"\nРезультаты сохранены в {csv_path}")
    
    # Сохранение модели
    model_path = output_dir / "spectral_analyzer.pkl"
    analyzer.save(model_path)
    print(f"Модель сохранена в {model_path}")


if __name__ == "__main__":
    main()

