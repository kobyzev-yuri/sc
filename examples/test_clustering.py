"""
Пример использования модуля кластеризации.

Демонстрирует:
- HDBSCAN кластеризацию
- Визуализацию кластеров в UMAP-пространстве
- Интерпретацию кластеров
- Метрики качества кластеризации
"""

from pathlib import Path
import sys

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from scale import aggregate
from scale import clustering


def main():
    """Основная функция для тестирования кластеризации."""
    
    # Загрузка данных
    predictions_dir = Path("results/predictions")
    if not predictions_dir.exists():
        print(f"Директория {predictions_dir} не найдена!")
        return
    
    print("Загрузка предсказаний...")
    predictions = {}
    for json_file in predictions_dir.glob("*.json"):
        try:
            from scale import domain
            preds = domain.predictions_from_json(str(json_file))
            image_name = json_file.stem
            predictions[image_name] = preds
        except Exception as e:
            print(f"Ошибка при загрузке {json_file.name}: {e}")
    
    if len(predictions) == 0:
        print("Нет предсказаний для обработки!")
        return
    
    print(f"Загружено {len(predictions)} файлов")
    
    # Агрегация данных
    print("\nАгрегация данных...")
    rows = []
    for image_name, preds in predictions.items():
        stats = aggregate.aggregate_predictions_from_dict(preds, image_name)
        rows.append(stats)
    
    df = pd.DataFrame(rows)
    
    # Создание относительных признаков
    print("Создание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    df_features = aggregate.select_feature_columns(df_features)
    
    print(f"Данные: {len(df_features)} образцов, {len(df_features.columns)-1} признаков")
    
    # Кластеризация
    print("\n" + "="*60)
    print("Кластеризация (HDBSCAN)")
    print("="*60)
    
    clusterer = clustering.ClusterAnalyzer(
        method="hdbscan",
        random_state=42,
    )
    
    clusterer.fit(
        df_features,
        use_pca=True,
        pca_components=10,
        min_cluster_size=2,
    )
    
    print(f"\nНайдено кластеров: {clusterer.n_clusters_}")
    print(f"Шум (outliers): {clusterer.cluster_stats_['noise_samples']}")
    print(f"Образцов в кластерах: {clusterer.cluster_stats_['total_samples']}")
    
    # Метрики качества
    print("\nМетрики качества кластеризации:")
    metrics = clusterer.get_metrics(df_features)
    for key, value in metrics.items():
        if isinstance(value, float) and not np.isnan(value):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Интерпретация кластеров
    print("\n" + "="*60)
    print("Интерпретация кластеров:")
    print("="*60)
    
    interpretation = clusterer.get_cluster_interpretation()
    for cluster_id, info in interpretation.items():
        print(f"\nКластер {cluster_id}:")
        print(f"  Образцов: {info['n_samples']}")
        print(f"  Топ признаки: {info['features_str']}")
        print(f"  Интерпретация: {info['interpretation']}")
    
    # Добавление меток кластеров к данным
    df_with_clusters = clusterer.transform(df_features)
    
    print("\n" + "="*60)
    print("Распределение по кластерам:")
    print("="*60)
    print(df_with_clusters["cluster"].value_counts().sort_index())
    
    # UMAP визуализация
    print("\nОбучение UMAP для визуализации...")
    clusterer.fit_umap(df_features, n_neighbors=5, min_dist=0.1)
    
    # Сохранение визуализации
    output_dir = Path("results/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / "clusters_umap.png"
    print(f"\nСохранение визуализации в {plot_path}...")
    clusterer.visualize_clusters(df_features, save_path=plot_path)
    
    # Сохранение модели
    model_path = Path("models/clusterer.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Сохранение модели в {model_path}...")
    clusterer.save(model_path)
    
    # Сохранение результатов
    results_path = output_dir / "clustering_results.csv"
    df_with_clusters.to_csv(results_path, index=False)
    print(f"Сохранение результатов в {results_path}...")
    
    print("\n✅ Кластеризация завершена!")
    print(f"\nРезультаты:")
    print(f"  - Визуализация: {plot_path}")
    print(f"  - Модель: {model_path}")
    print(f"  - Результаты: {results_path}")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main()

