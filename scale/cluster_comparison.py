"""
Модуль для сравнения результатов нескольких кластеризаций.

Поддерживает:
- Загрузку нескольких сохраненных кластеризаторов
- Сравнение метрик качества
- Сравнение распределений по кластерам
- Маппинг всех кластеризаций на шкалу 0-1
- Визуализацию сравнения
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from . import clustering
from . import cluster_scoring


class ClusterComparison:
    """
    Класс для сравнения результатов нескольких кластеризаций.
    """
    
    def __init__(self):
        self.clusterers: Dict[str, clustering.ClusterAnalyzer] = {}
        self.results: Dict[str, pd.DataFrame] = {}
        self.scores: Dict[str, pd.DataFrame] = {}
        self.metrics: Dict[str, Dict] = {}
    
    def load_clusterer(
        self,
        name: str,
        path: Union[str, Path],
        df_features: pd.DataFrame,
    ) -> "ClusterComparison":
        """
        Загружает сохраненный кластеризатор и применяет его к данным.
        
        Args:
            name: Имя кластеризатора (для идентификации)
            path: Путь к файлу .pkl с сохраненным кластеризатором
            df_features: DataFrame с признаками для применения кластеризации
        
        Returns:
            self
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        
        # Загрузка кластеризатора
        clusterer = clustering.ClusterAnalyzer.load(path)
        self.clusterers[name] = clusterer
        
        # Применение к данным
        df_with_clusters = clusterer.transform(df_features)
        self.results[name] = df_with_clusters
        
        # Вычисление метрик
        metrics = clusterer.get_metrics(df_features)
        self.metrics[name] = metrics
        
        return self
    
    def load_multiple_clusterers(
        self,
        clusterer_paths: Dict[str, Union[str, Path]],
        df_features: pd.DataFrame,
    ) -> "ClusterComparison":
        """
        Загружает несколько сохраненных кластеризаторов.
        
        Args:
            clusterer_paths: Словарь {имя: путь} к файлам .pkl
            df_features: DataFrame с признаками
        
        Returns:
            self
        """
        for name, path in clusterer_paths.items():
            self.load_clusterer(name, path, df_features)
        
        return self
    
    def apply_scoring(
        self,
        scoring_method: str = "pathology_features",
        pathology_features: Optional[List[str]] = None,
        expert_labels: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> "ClusterComparison":
        """
        Применяет маппинг кластеров на шкалу 0-1 для всех загруженных кластеризаций.
        
        Args:
            scoring_method: Метод маппинга ("pathology_features", "pc1_centroid", "expert_labels", "distance_from_normal")
            pathology_features: Список патологических признаков (для pathology_features)
            expert_labels: Словарь {имя_кластеризации: {cluster_id: score}} для expert_labels метода
        
        Returns:
            self
        """
        self.scores = {}
        
        for name, clusterer in self.clusterers.items():
            # Получаем данные с кластерами
            df_with_clusters = self.results[name]
            
            # Создаем scorer
            scorer = cluster_scoring.ClusterScorer(method=scoring_method)
            
            # Применяем маппинг
            if scoring_method == "expert_labels" and expert_labels and name in expert_labels:
                df_with_scores = scorer.fit_transform(
                    df_with_clusters,
                    clusterer=clusterer,
                    expert_labels=expert_labels[name]
                )
            else:
                df_with_scores = scorer.fit_transform(
                    df_with_clusters,
                    clusterer=clusterer,
                    pathology_features=pathology_features
                )
            
            self.scores[name] = df_with_scores
        
        return self
    
    def compare_metrics(self) -> pd.DataFrame:
        """
        Сравнивает метрики качества всех кластеризаций.
        
        Returns:
            DataFrame с метриками для каждой кластеризации
        """
        if not self.metrics:
            raise ValueError("Нет загруженных кластеризаций. Используйте load_clusterer() сначала.")
        
        metrics_list = []
        for name, metrics in self.metrics.items():
            row = {"Имя": name}
            row.update(metrics)
            metrics_list.append(row)
        
        return pd.DataFrame(metrics_list)
    
    def compare_cluster_distributions(self) -> pd.DataFrame:
        """
        Сравнивает распределение образцов по кластерам для всех кластеризаций.
        
        Returns:
            DataFrame с распределениями
        """
        if not self.results:
            raise ValueError("Нет загруженных кластеризаций. Используйте load_clusterer() сначала.")
        
        distributions = []
        for name, df_result in self.results.items():
            cluster_counts = df_result["cluster"].value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                distributions.append({
                    "Кластеризация": name,
                    "Кластер": cluster_id,
                    "Число образцов": count
                })
        
        return pd.DataFrame(distributions)
    
    def compare_scores(self) -> pd.DataFrame:
        """
        Сравнивает cluster_score для всех кластеризаций.
        
        Returns:
            DataFrame с колонками: image, {name}_cluster, {name}_cluster_score, ...
        """
        if not self.scores:
            raise ValueError("Маппинг на шкалу не применен. Используйте apply_scoring() сначала.")
        
        # Объединяем все результаты
        comparison = pd.DataFrame()
        
        # Базовый DataFrame с image
        first_name = list(self.scores.keys())[0]
        comparison["image"] = self.scores[first_name]["image"]
        
        # Добавляем кластеры и scores для каждой кластеризации
        for name, df_scores in self.scores.items():
            comparison[f"{name}_cluster"] = df_scores["cluster"].values
            comparison[f"{name}_cluster_score"] = df_scores["cluster_score"].values
        
        return comparison
    
    def visualize_comparison(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (16, 10),
    ) -> None:
        """
        Визуализирует сравнение кластеризаций.
        
        Args:
            save_path: Путь для сохранения графика
            figsize: Размер фигуры
        """
        if not self.scores:
            raise ValueError("Маппинг на шкалу не применен. Используйте apply_scoring() сначала.")
        
        n_clusterers = len(self.scores)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # График 1: Распределение cluster_score по кластеризациям
        ax1 = axes[0, 0]
        for name, df_scores in self.scores.items():
            scores = df_scores["cluster_score"].dropna()
            ax1.hist(
                scores,
                bins=20,
                alpha=0.6,
                label=name,
                density=True
            )
        ax1.set_xlabel("Cluster Score (0-1)", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Распределение cluster_score по кластеризациям", fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Boxplot сравнение scores
        ax2 = axes[0, 1]
        score_data = []
        score_labels = []
        for name, df_scores in self.scores.items():
            scores = df_scores["cluster_score"].dropna().values
            score_data.append(scores)
            score_labels.append(name)
        ax2.boxplot(score_data, labels=score_labels)
        ax2.set_ylabel("Cluster Score (0-1)", fontsize=12)
        ax2.set_title("Boxplot сравнение cluster_score", fontsize=13)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # График 3: Scatter plot сравнения (если есть 2+ кластеризации)
        if n_clusterers >= 2:
            ax3 = axes[1, 0]
            names = list(self.scores.keys())
            df1 = self.scores[names[0]]
            df2 = self.scores[names[1]]
            
            # Объединяем по image
            merged = df1[["image", "cluster_score"]].merge(
                df2[["image", "cluster_score"]],
                on="image",
                suffixes=(f"_{names[0]}", f"_{names[1]}")
            )
            
            ax3.scatter(
                merged[f"cluster_score_{names[0]}"],
                merged[f"cluster_score_{names[1]}"],
                alpha=0.6,
                s=100
            )
            ax3.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
            ax3.set_xlabel(f"{names[0]} cluster_score", fontsize=12)
            ax3.set_ylabel(f"{names[1]} cluster_score", fontsize=12)
            ax3.set_title("Сравнение cluster_score между кластеризациями", fontsize=13)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # График 4: Метрики качества
        ax4 = axes[1, 1]
        if self.metrics:
            metrics_df = self.compare_metrics()
            
            # Выбираем метрики для визуализации
            metric_cols = ["n_clusters", "silhouette_score", "calinski_harabasz_score"]
            available_cols = [col for col in metric_cols if col in metrics_df.columns]
            
            if available_cols:
                x = np.arange(len(metrics_df))
                width = 0.25
                
                for idx, col in enumerate(available_cols):
                    values = metrics_df[col].fillna(0).values
                    # Нормализуем для визуализации (если нужно)
                    if col != "n_clusters":
                        if values.max() > 0:
                            values = values / values.max()
                    ax4.bar(x + idx * width, values, width, label=col, alpha=0.7)
                
                ax4.set_xlabel("Кластеризация", fontsize=12)
                ax4.set_ylabel("Нормализованные значения", fontsize=12)
                ax4.set_title("Метрики качества кластеризаций", fontsize=13)
                ax4.set_xticks(x + width)
                ax4.set_xticklabels(metrics_df["Имя"], rotation=45, ha="right")
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_summary(self) -> Dict:
        """
        Возвращает сводку по всем кластеризациям.
        
        Returns:
            Словарь с сводной информацией
        """
        summary = {
            "n_clusterers": len(self.clusterers),
            "names": list(self.clusterers.keys()),
            "metrics": self.compare_metrics().to_dict('records') if self.metrics else None,
            "has_scores": len(self.scores) > 0,
        }
        
        if self.scores:
            summary["score_statistics"] = {}
            for name, df_scores in self.scores.items():
                scores = df_scores["cluster_score"].dropna()
                summary["score_statistics"][name] = {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                    "median": float(scores.median()),
                }
        
        return summary

