"""
Пример сравнения нескольких сохраненных кластеризаций и маппинга на шкалу.

Использование:
    python examples/compare_clusterings.py
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from scale import aggregate, cluster_comparison


def main():
    """Основная функция для сравнения кластеризаций."""
    
    # Загрузка данных
    print("Загрузка данных...")
    predictions_dir = Path("results/predictions")
    if not predictions_dir.exists():
        print(f"Директория {predictions_dir} не найдена!")
        return
    
    from scale import domain
    predictions = {}
    for json_file in predictions_dir.glob("*.json"):
        try:
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
    
    df = aggregate.create_relative_features(pd.DataFrame(rows))
    df_features = aggregate.select_feature_columns(df)
    
    print(f"Данные: {len(df_features)} образцов, {len(df_features.columns)-1} признаков")
    
    # Загрузка сохраненных кластеризаций
    print("\n" + "="*60)
    print("Загрузка сохраненных кластеризаций")
    print("="*60)
    
    # Укажите пути к вашим сохраненным кластеризаторам
    clusterer_paths = {
        "hdbscan_1": "models/clusterer_hdbscan_1.pkl",
        "hdbscan_2": "models/clusterer_hdbscan_2.pkl",
        "hdbscan_3": "models/clusterer_hdbscan_3.pkl",
    }
    
    # Или найдите все .pkl файлы в директории models
    models_dir = Path("models")
    if models_dir.exists():
        pkl_files = list(models_dir.glob("clusterer*.pkl"))
        if pkl_files:
            print(f"\nНайдено {len(pkl_files)} сохраненных кластеризаторов:")
            clusterer_paths = {}
            for i, pkl_file in enumerate(pkl_files[:3], 1):  # Берем первые 3
                name = pkl_file.stem
                clusterer_paths[name] = str(pkl_file)
                print(f"  {i}. {name}: {pkl_file}")
    
    if not clusterer_paths:
        print("\n⚠️ Не найдено сохраненных кластеризаторов!")
        print("Сначала сохраните кластеризаторы через dashboard или test_clustering.py")
        return
    
    # Создание объекта сравнения
    comparison = cluster_comparison.ClusterComparison()
    
    # Загрузка кластеризаторов
    print("\nЗагрузка кластеризаторов...")
    for name, path in clusterer_paths.items():
        try:
            comparison.load_clusterer(name, path, df_features)
            print(f"  ✅ {name}: загружен")
        except Exception as e:
            print(f"  ❌ {name}: ошибка - {e}")
    
    if len(comparison.clusterers) == 0:
        print("Не удалось загрузить ни одного кластеризатора!")
        return
    
    # Сравнение метрик
    print("\n" + "="*60)
    print("Сравнение метрик качества")
    print("="*60)
    metrics_df = comparison.compare_metrics()
    print(metrics_df.to_string(index=False))
    
    # Сравнение распределений по кластерам
    print("\n" + "="*60)
    print("Распределение по кластерам")
    print("="*60)
    dist_df = comparison.compare_cluster_distributions()
    print(dist_df.to_string(index=False))
    
    # Применение маппинга на шкалу
    print("\n" + "="*60)
    print("Применение маппинга кластеров на шкалу 0-1")
    print("="*60)
    
    scoring_method = "pathology_features"  # Можно изменить на "pc1_centroid", "distance_from_normal"
    print(f"Метод маппинга: {scoring_method}")
    
    comparison.apply_scoring(scoring_method=scoring_method)
    
    # Сравнение scores
    print("\n" + "="*60)
    print("Сравнение cluster_score")
    print("="*60)
    scores_df = comparison.compare_scores()
    print(scores_df.head(10).to_string(index=False))
    
    # Статистика по scores
    print("\nСтатистика cluster_score:")
    for name in comparison.scores.keys():
        scores = comparison.scores[name]["cluster_score"]
        print(f"\n{name}:")
        print(f"  Mean: {scores.mean():.3f}")
        print(f"  Std:  {scores.std():.3f}")
        print(f"  Min:  {scores.min():.3f}")
        print(f"  Max:  {scores.max():.3f}")
    
    # Визуализация
    print("\n" + "="*60)
    print("Создание визуализации сравнения")
    print("="*60)
    
    output_dir = Path("results/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / "cluster_comparison.png"
    comparison.visualize_comparison(save_path=plot_path)
    print(f"График сохранен: {plot_path}")
    
    # Сохранение результатов
    print("\n" + "="*60)
    print("Сохранение результатов")
    print("="*60)
    
    # Сохранение сравнения scores
    scores_csv = output_dir / "cluster_scores_comparison.csv"
    scores_df.to_csv(scores_csv, index=False)
    print(f"Сравнение scores: {scores_csv}")
    
    # Сохранение метрик
    metrics_csv = output_dir / "cluster_metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Метрики: {metrics_csv}")
    
    # Сохранение распределений
    dist_csv = output_dir / "cluster_distributions.csv"
    dist_df.to_csv(dist_csv, index=False)
    print(f"Распределения: {dist_csv}")
    
    # Сводка
    print("\n" + "="*60)
    print("Сводка")
    print("="*60)
    summary = comparison.get_summary()
    print(f"Кластеризаций: {summary['n_clusterers']}")
    print(f"Имена: {', '.join(summary['names'])}")
    if summary['has_scores']:
        print("\nСтатистика scores:")
        for name, stats in summary['score_statistics'].items():
            print(f"  {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    print("\n✅ Сравнение завершено!")


if __name__ == "__main__":
    import pandas as pd
    main()

