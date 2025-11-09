"""
Модуль для создания шкалы оценки патологии на основе кластеризации.

Поддерживает несколько методов маппинга кластеров на score (0.0 - 1.0):
1. На основе средних значений патологических признаков
2. На основе PC1 центроидов кластеров
3. На основе экспертной разметки (если доступна)
4. На основе расстояния от "нормального" кластера
"""

from pathlib import Path
from typing import Optional, Union, Dict, List
import pandas as pd
import numpy as np
import pickle
import warnings

from . import clustering


class ClusterScorer:
    """
    Класс для создания шкалы оценки патологии на основе кластеризации.
    
    Маппит кластеры на score от 0 (норма) до 1 (максимальная патология).
    """
    
    def __init__(
        self,
        method: str = "pathology_features",
        clusterer: Optional[clustering.ClusterAnalyzer] = None,
    ):
        """
        Инициализация кластерного скорера.
        
        Args:
            method: Метод маппинга ("pathology_features", "pc1_centroid", "expert_labels", "distance_from_normal")
            clusterer: Обученный ClusterAnalyzer (можно передать позже)
        """
        self.method = method
        self.clusterer = clusterer
        
        # Маппинг кластеров на score
        self.cluster_to_score: Dict[int, float] = {}
        
        # Метаданные
        self.pathology_features: Optional[List[str]] = None
        self.normal_cluster: Optional[int] = None
    
    def fit(
        self,
        df: pd.DataFrame,
        clusterer: Optional[clustering.ClusterAnalyzer] = None,
        pathology_features: Optional[List[str]] = None,
        expert_labels: Optional[Dict[int, float]] = None,
        normal_cluster: Optional[int] = None,
    ) -> "ClusterScorer":
        """
        Обучение маппинга кластеров на score.
        
        Args:
            df: DataFrame с признаками
            clusterer: Обученный ClusterAnalyzer (если не передан в __init__)
            pathology_features: Список патологических признаков для метода "pathology_features"
            expert_labels: Словарь {cluster_id: score} для метода "expert_labels"
            normal_cluster: ID кластера с нормальными образцами для метода "distance_from_normal"
        
        Returns:
            self
        """
        if clusterer is not None:
            self.clusterer = clusterer
        
        if self.clusterer is None:
            raise ValueError("ClusterAnalyzer не предоставлен. Передайте clusterer в fit() или __init__()")
        
        if self.clusterer.cluster_stats_ is None:
            raise ValueError("Кластеризатор не обучен. Вызовите clusterer.fit() сначала.")
        
        # Применяем кластеризацию к данным
        df_with_clusters = self.clusterer.transform(df)
        
        # Выбираем метод маппинга
        if self.method == "pathology_features":
            self._fit_pathology_features(df_with_clusters, pathology_features)
        elif self.method == "pc1_centroid":
            self._fit_pc1_centroid(df_with_clusters)
        elif self.method == "expert_labels":
            self._fit_expert_labels(expert_labels)
        elif self.method == "distance_from_normal":
            self._fit_distance_from_normal(df_with_clusters, normal_cluster)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        return self
    
    def _fit_pathology_features(
        self,
        df_with_clusters: pd.DataFrame,
        pathology_features: Optional[List[str]] = None,
    ) -> None:
        """
        Маппинг на основе средних значений патологических признаков.
        
        Кластеры с высокими значениями патологических признаков получают высокий score.
        """
        if pathology_features is None:
            # Автоматически определяем патологические признаки
            numeric_cols = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
            pathology_features = [
                f for f in numeric_cols 
                if any(x in f.lower() for x in ['dysplasia', 'mild', 'moderate', 'eoe', 'granulomas'])
                and 'relative' in f.lower()
            ]
        
        self.pathology_features = pathology_features
        
        if not pathology_features:
            warnings.warn("Не найдено патологических признаков. Используем все числовые признаки.")
            numeric_cols = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
            pathology_features = [f for f in numeric_cols if f not in ['cluster', 'image']]
            self.pathology_features = pathology_features
        
        # Вычисляем средние значения патологических признаков по кластерам
        cluster_means = self.clusterer.cluster_stats_["means"]
        
        # Суммируем патологические признаки для каждого кластера
        available_features = [f for f in pathology_features if f in cluster_means.columns]
        if not available_features:
            raise ValueError(f"Патологические признаки {pathology_features} не найдены в данных")
        
        cluster_scores = cluster_means[available_features].sum(axis=1)
        
        # Нормализуем в диапазон [0, 1]
        if cluster_scores.max() > cluster_scores.min():
            cluster_scores_norm = (cluster_scores - cluster_scores.min()) / (cluster_scores.max() - cluster_scores.min())
        else:
            # Если все кластеры одинаковые, присваиваем средний score
            cluster_scores_norm = pd.Series(0.5, index=cluster_scores.index)
        
        # Маппинг кластеров на score
        self.cluster_to_score = cluster_scores_norm.to_dict()
        
        # Шум (-1) получает минимальный score
        if -1 in df_with_clusters["cluster"].values:
            min_score = min(self.cluster_to_score.values()) if self.cluster_to_score else 0.0
            self.cluster_to_score[-1] = min_score
    
    def _fit_pc1_centroid(self, df_with_clusters: pd.DataFrame) -> None:
        """
        Маппинг на основе PC1 центроидов кластеров.
        
        Использует среднее значение PC1 для каждого кластера.
        """
        if "PC1" not in df_with_clusters.columns:
            raise ValueError("Колонка PC1 не найдена. Выполните PCA анализ сначала.")
        
        # Вычисляем средние PC1 значения по кластерам
        cluster_pc1_means = df_with_clusters.groupby("cluster")["PC1"].mean()
        
        # Нормализуем в диапазон [0, 1]
        if cluster_pc1_means.max() > cluster_pc1_means.min():
            cluster_scores = (cluster_pc1_means - cluster_pc1_means.min()) / (
                cluster_pc1_means.max() - cluster_pc1_means.min()
            )
        else:
            cluster_scores = pd.Series(0.5, index=cluster_pc1_means.index)
        
        self.cluster_to_score = cluster_scores.to_dict()
    
    def _fit_expert_labels(self, expert_labels: Optional[Dict[int, float]]) -> None:
        """
        Маппинг на основе экспертной разметки.
        
        Args:
            expert_labels: Словарь {cluster_id: score} с экспертными оценками
        """
        if expert_labels is None:
            raise ValueError("expert_labels должны быть предоставлены для метода 'expert_labels'")
        
        # Нормализуем экспертные оценки в [0, 1]
        scores = np.array(list(expert_labels.values()))
        if scores.max() > scores.min():
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_norm = np.full_like(scores, 0.5)
        
        self.cluster_to_score = {
            cluster_id: float(score)
            for (cluster_id, score), score_norm in zip(expert_labels.items(), scores_norm)
        }
    
    def _fit_distance_from_normal(
        self,
        df_with_clusters: pd.DataFrame,
        normal_cluster: Optional[int] = None,
    ) -> None:
        """
        Маппинг на основе расстояния от "нормального" кластера.
        
        Args:
            df_with_clusters: DataFrame с кластерами
            normal_cluster: ID кластера с нормальными образцами. Если None, выбирается автоматически.
        """
        if normal_cluster is None:
            # Автоматически определяем нормальный кластер (с минимальными патологическими признаками)
            cluster_means = self.clusterer.cluster_stats_["means"]
            
            # Ищем признаки патологии
            pathology_cols = [
                col for col in cluster_means.columns
                if any(x in col.lower() for x in ['dysplasia', 'mild', 'moderate', 'eoe'])
            ]
            
            if pathology_cols:
                # Кластер с минимальной суммой патологических признаков = норма
                cluster_pathology_sums = cluster_means[pathology_cols].sum(axis=1)
                normal_cluster = cluster_pathology_sums.idxmin()
            else:
                # Если нет патологических признаков, берем кластер с минимальными значениями всех признаков
                cluster_sums = cluster_means.sum(axis=1)
                normal_cluster = cluster_sums.idxmin()
        
        self.normal_cluster = normal_cluster
        
        # Вычисляем расстояния от нормального кластера
        cluster_means = self.clusterer.cluster_stats_["means"]
        normal_centroid = cluster_means.loc[normal_cluster].values
        
        distances = {}
        for cluster_id in cluster_means.index:
            cluster_centroid = cluster_means.loc[cluster_id].values
            distance = np.linalg.norm(cluster_centroid - normal_centroid)
            distances[cluster_id] = distance
        
        # Нормализуем расстояния в [0, 1]
        dist_values = np.array(list(distances.values()))
        if dist_values.max() > dist_values.min():
            dist_norm = (dist_values - dist_values.min()) / (dist_values.max() - dist_values.min())
        else:
            dist_norm = np.full_like(dist_values, 0.5)
        
        self.cluster_to_score = {
            cluster_id: float(score)
            for (cluster_id, _), score in zip(distances.items(), dist_norm)
        }
        
        # Нормальный кластер получает score 0
        if normal_cluster in self.cluster_to_score:
            self.cluster_to_score[normal_cluster] = 0.0
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет маппинг кластеров на score к данным.
        
        Args:
            df: DataFrame с колонкой "cluster"
        
        Returns:
            DataFrame с добавленной колонкой "cluster_score"
        """
        if not self.cluster_to_score:
            raise ValueError("Маппинг не обучен. Вызовите fit() сначала.")
        
        if "cluster" not in df.columns:
            raise ValueError("Колонка 'cluster' не найдена. Выполните кластеризацию сначала.")
        
        df_result = df.copy()
        
        # Маппинг кластеров на score
        df_result["cluster_score"] = df_result["cluster"].map(self.cluster_to_score)
        
        # Заполняем NaN значениями (для неизвестных кластеров)
        if df_result["cluster_score"].isna().any():
            default_score = np.mean(list(self.cluster_to_score.values()))
            df_result["cluster_score"] = df_result["cluster_score"].fillna(default_score)
        
        return df_result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        clusterer: Optional[clustering.ClusterAnalyzer] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Обучает маппинг и применяет его к данным.
        
        Args:
            df: DataFrame с признаками
            clusterer: Обученный ClusterAnalyzer
            **kwargs: Дополнительные параметры для fit()
        
        Returns:
            DataFrame с добавленной колонкой "cluster_score"
        """
        self.fit(df, clusterer=clusterer, **kwargs)
        df_with_clusters = self.clusterer.transform(df)
        return self.transform(df_with_clusters)
    
    def get_cluster_scores(self) -> pd.Series:
        """
        Возвращает маппинг кластеров на score.
        
        Returns:
            Series с score для каждого кластера
        """
        return pd.Series(self.cluster_to_score).sort_values()
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Сохранение обученной модели.
        
        Args:
            path: Путь для сохранения
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                "method": self.method,
                "cluster_to_score": self.cluster_to_score,
                "pathology_features": self.pathology_features,
                "normal_cluster": self.normal_cluster,
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClusterScorer":
        """
        Загрузка обученной модели.
        
        Args:
            path: Путь к файлу модели
        
        Returns:
            Загруженный ClusterScorer
        """
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        scorer = cls(method=data["method"])
        scorer.cluster_to_score = data["cluster_to_score"]
        scorer.pathology_features = data.get("pathology_features")
        scorer.normal_cluster = data.get("normal_cluster")
        
        return scorer

