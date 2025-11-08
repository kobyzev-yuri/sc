"""
Запуск экспериментов по построению шкалы из конфигурационного файла.

Позволяет легко настраивать и запускать разные эксперименты
без изменения кода.
"""

from pathlib import Path
import sys
import json

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scale import aggregate
from scale.scale_comparison import ScaleComparison


def load_config(config_path: Path) -> dict:
    """Загружает конфигурацию из JSON файла."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiments_from_config(config_path: Path = None):
    """
    Запускает эксперименты из конфигурационного файла.
    
    Args:
        config_path: Путь к конфигурационному файлу. 
                    Если None, используется experiment_config.json в examples/
    """
    if config_path is None:
        config_path = Path(__file__).parent / "experiment_config.json"
    
    if not config_path.exists():
        print(f"Конфигурационный файл не найден: {config_path}")
        return
    
    print("="*80)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ ИЗ КОНФИГУРАЦИИ")
    print("="*80)
    print(f"Конфигурация: {config_path}")
    
    config = load_config(config_path)
    
    # Загрузка предсказаний
    predictions_dir = project_root / "results" / "predictions"
    
    if not predictions_dir.exists():
        print(f"Директория {predictions_dir} не найдена")
        return
    
    print(f"\nЗагрузка предсказаний из {predictions_dir}")
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"Загружено {len(df)} образцов")
    
    # Подготовка данных
    print("\nПодготовка данных...")
    preprocessing = config.get("data_preprocessing", {})
    
    if preprocessing.get("use_relative_features", True):
        df_features = aggregate.create_relative_features(df)
    else:
        df_features = df.copy()
    
    if preprocessing.get("select_feature_columns", True):
        df_features = aggregate.select_feature_columns(df_features)
    
    fillna_value = preprocessing.get("fillna_value", 0)
    df_features = df_features.fillna(fillna_value)
    
    print(f"Признаков: {len(df_features.columns)}")
    print(f"Образцов: {len(df_features)}")
    
    # Инициализация сравнения
    comparison = ScaleComparison()
    
    # Запуск экспериментов
    experiments = config.get("experiments", [])
    enabled_experiments = [exp for exp in experiments if exp.get("enabled", True)]
    
    print(f"\nЗапуск {len(enabled_experiments)} экспериментов...")
    
    for exp_config in enabled_experiments:
        method = exp_config.get("method")
        name = exp_config.get("name")
        description = exp_config.get("description", "")
        
        if not method or not name:
            print(f"Пропущен эксперимент: отсутствует method или name")
            continue
        
        print(f"\n{'='*80}")
        print(f"Эксперимент: {name}")
        print(f"Описание: {description}")
        print(f"{'='*80}")
        
        try:
            if method == "pca_scoring":
                comparison.test_pca_scoring(df_features, name=name)
            
            elif method == "spectral_analysis":
                percentile_low = exp_config.get("percentile_low", 1.0)
                percentile_high = exp_config.get("percentile_high", 99.0)
                use_gmm = exp_config.get("use_gmm", False)
                
                comparison.test_spectral_analysis(
                    df_features,
                    name=name,
                    percentile_low=percentile_low,
                    percentile_high=percentile_high,
                    use_gmm=use_gmm
                )
            else:
                print(f"Неизвестный метод: {method}")
                continue
                
        except Exception as e:
            print(f"Ошибка при выполнении эксперимента {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Сравнение результатов
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    try:
        comparison_df = comparison.compare_results()
        print("\nСравнение шкал (первые 5 образцов):")
        print(comparison_df.head())
        
        # Статистика
        print("\n\nСтатистика по методам:")
        stats = comparison.get_statistics()
        print(stats.to_string(index=False))
    except Exception as e:
        print(f"Ошибка при сравнении результатов: {e}")
        return
    
    # Сохранение результатов
    output_config = config.get("output", {})
    base_dir = output_config.get("base_dir", "experiments")
    experiment_name = output_config.get("experiment_name", "experiment")
    
    output_dir = project_root / base_dir / experiment_name
    
    print("\n" + "="*80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    try:
        comparison.save_results(output_dir)
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        return
    
    # Визуализация
    if output_config.get("create_visualizations", True):
        print("\n" + "="*80)
        print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ")
        print("="*80)
        
        try:
            viz_path = output_dir / "comparison_plot.png"
            comparison.visualize_comparison(save_path=viz_path)
        except Exception as e:
            print(f"Ошибка при создании визуализации: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("="*80)
    print(f"\nВсе результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск экспериментов по построению шкалы")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к конфигурационному файлу (по умолчанию: examples/experiment_config.json)"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    run_experiments_from_config(config_path)


