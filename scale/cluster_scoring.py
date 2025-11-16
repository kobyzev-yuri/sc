"""
Модуль для создания шкалы оценки патологии на основе кластеризации.

Поддерживает несколько методов маппинга кластеров на score (0.0 - 1.0):
1. На основе средних значений патологических признаков
2. На основе PC1 центроидов кластеров
3. На основе экспертной разметки (если доступна)
4. На основе расстояния от "нормального" кластера
5. На основе спектрального анализа (spectrum_projection) - интегрированный подход

Интеграция со спектральным анализом:
- Использует единую шкалу нормализации через процентили (P1/P99)
- Проецирует кластеры на спектральную шкалу через PC1
- Учитывает распределения внутри кластеров (медиана, процентили)
- Классифицирует кластеры по модам из спектрального анализа
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
    
    Интегрирован со спектральным анализом для единой шкалы нормализации
    и учета мод (стабильных состояний) в распределении патологий.
    """
    
    def __init__(
        self,
        method: str = "pathology_features",
        clusterer: Optional[clustering.ClusterAnalyzer] = None,
        use_percentiles: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ):
        """
        Инициализация кластерного скорера.
        
        Args:
            method: Метод маппинга ("pathology_features", "pc1_centroid", "expert_labels", 
                   "distance_from_normal", "spectrum_projection")
            clusterer: Обученный ClusterAnalyzer (можно передать позже)
            use_percentiles: Использовать процентили для нормализации (устойчивость к выбросам)
            percentile_low: Нижний процентиль для нормализации (по умолчанию 1.0)
            percentile_high: Верхний процентиль для нормализации (по умолчанию 99.0)
        """
        self.method = method
        self.clusterer = clusterer
        self.use_percentiles = use_percentiles
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        # Маппинг кластеров на score
        self.cluster_to_score: Dict[int, float] = {}
        
        # Метаданные
        self.pathology_features: Optional[List[str]] = None
        self.normal_cluster: Optional[int] = None
        
        # Для интеграции со спектральным анализом
        self.spectral_analyzer: Optional[object] = None  # SpectralAnalyzer
        self.cluster_to_mode: Optional[Dict[int, str]] = None  # Классификация кластеров по модам
        self.cluster_distributions: Optional[Dict[int, Dict]] = None  # Распределения внутри кластеров
    
    def fit(
        self,
        df: pd.DataFrame,
        clusterer: Optional[clustering.ClusterAnalyzer] = None,
        pathology_features: Optional[List[str]] = None,
        expert_labels: Optional[Dict[int, float]] = None,
        normal_cluster: Optional[int] = None,
        spectral_analyzer: Optional[object] = None,
        use_cluster_distribution: bool = True,
    ) -> "ClusterScorer":
        """
        Обучение маппинга кластеров на score.
        
        Args:
            df: DataFrame с признаками
            clusterer: Обученный ClusterAnalyzer (если не передан в __init__)
            pathology_features: Список патологических признаков для метода "pathology_features"
            expert_labels: Словарь {cluster_id: score} для метода "expert_labels"
            normal_cluster: ID кластера с нормальными образцами для метода "distance_from_normal"
            spectral_analyzer: Обученный SpectralAnalyzer для метода "spectrum_projection"
            use_cluster_distribution: Учитывать распределения внутри кластеров (медиана вместо среднего)
        
        Returns:
            self
        """
        if clusterer is not None:
            self.clusterer = clusterer
        
        if self.clusterer is None:
            raise ValueError("ClusterAnalyzer не предоставлен. Передайте clusterer в fit() или __init__()")
        
        # Проверяем, что кластеризатор обучен (labels_ не None)
        # cluster_stats_ может быть None, если все образцы - шум, но это не значит, что кластеризатор не обучен
        if self.clusterer.labels_ is None:
            raise ValueError("Кластеризатор не обучен. Вызовите clusterer.fit() сначала.")
        
        # Предупреждаем, если все образцы - шум
        if self.clusterer.cluster_stats_ is None:
            n_noise = (self.clusterer.labels_ == -1).sum()
            n_total = len(self.clusterer.labels_)
            if n_noise == n_total:
                raise ValueError(
                    f"Кластеризация не нашла кластеры: все {n_total} образцов помечены как шум. "
                    "Попробуйте изменить параметры кластеризации (уменьшите min_cluster_size, увеличьте число PCA компонент)."
                )
        
        # Применяем кластеризацию к данным
        df_with_clusters = self.clusterer.transform(df)
        
        # Выбираем метод маппинга
        if self.method == "pathology_features":
            self._fit_pathology_features(df_with_clusters, pathology_features, use_cluster_distribution)
        elif self.method == "pc1_centroid":
            self._fit_pc1_centroid(df_with_clusters, use_cluster_distribution)
        elif self.method == "expert_labels":
            self._fit_expert_labels(expert_labels)
        elif self.method == "distance_from_normal":
            self._fit_distance_from_normal(df_with_clusters, normal_cluster)
        elif self.method == "spectrum_projection":
            if spectral_analyzer is None:
                raise ValueError("spectral_analyzer должен быть предоставлен для метода 'spectrum_projection'")
            self.spectral_analyzer = spectral_analyzer
            self._fit_spectrum_projection(df_with_clusters, spectral_analyzer, use_cluster_distribution)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        return self
    
    def _normalize_with_percentiles(self, values: pd.Series) -> pd.Series:
        """
        Нормализация значений через процентили (устойчивость к выбросам).
        
        Args:
            values: Series с значениями для нормализации
        
        Returns:
            Series с нормализованными значениями [0, 1]
        """
        if self.use_percentiles:
            p_low = np.percentile(values, self.percentile_low)
            p_high = np.percentile(values, self.percentile_high)
            
            if p_high > p_low:
                normalized = (values - p_low) / (p_high - p_low)
                normalized = np.clip(normalized, 0.0, 1.0)
            else:
                # Если процентили совпадают, присваиваем средний score
                normalized = pd.Series(0.5, index=values.index)
        else:
            # Fallback на min-max нормализацию
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = pd.Series(0.5, index=values.index)
        
        return normalized
    
    def _fit_pathology_features(
        self,
        df_with_clusters: pd.DataFrame,
        pathology_features: Optional[List[str]] = None,
        use_cluster_distribution: bool = True,
    ) -> None:
        """
        Маппинг на основе средних значений патологических признаков.
        
        Кластеры с высокими значениями патологических признаков получают высокий score.
        
        Args:
            df_with_clusters: DataFrame с кластерами
            pathology_features: Список патологических признаков
            use_cluster_distribution: Если True, использует медиану вместо среднего (устойчивее к выбросам)
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
        
        # Вычисляем статистику по кластерам
        available_features = [f for f in pathology_features if f in df_with_clusters.columns]
        if not available_features:
            raise ValueError(f"Патологические признаки {pathology_features} не найдены в данных")
        
        # Используем медиану или среднее в зависимости от параметра
        if use_cluster_distribution:
            cluster_stats = df_with_clusters.groupby("cluster")[available_features].median()
        else:
            cluster_stats = self.clusterer.cluster_stats_["means"][available_features]
        
        # Суммируем патологические признаки для каждого кластера
        cluster_scores = cluster_stats.sum(axis=1)
        
        # Нормализуем через процентили
        cluster_scores_norm = self._normalize_with_percentiles(cluster_scores)
        
        # Маппинг кластеров на score
        self.cluster_to_score = cluster_scores_norm.to_dict()
        
        # Шум (-1) не получает фиксированный score - будет обработан в transform() на основе PC1
        # (если PC1 доступен) или получит минимальный score как fallback
    
    def _fit_pc1_centroid(
        self, 
        df_with_clusters: pd.DataFrame,
        use_cluster_distribution: bool = True,
    ) -> None:
        """
        Маппинг на основе PC1 центроидов кластеров.
        
        Использует медиану или среднее значение PC1 для каждого кластера.
        
        Args:
            df_with_clusters: DataFrame с кластерами и PC1
            use_cluster_distribution: Если True, использует медиану (устойчивее к выбросам)
        """
        if "PC1" not in df_with_clusters.columns:
            raise ValueError("Колонка PC1 не найдена. Выполните PCA анализ сначала.")
        
        # Вычисляем статистику PC1 по кластерам
        if use_cluster_distribution:
            cluster_pc1_stats = df_with_clusters.groupby("cluster")["PC1"].median()
        else:
            cluster_pc1_stats = df_with_clusters.groupby("cluster")["PC1"].mean()
        
        # Нормализуем через процентили
        cluster_scores = self._normalize_with_percentiles(cluster_pc1_stats)
        
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
        
        # Нормализуем расстояния через процентили
        dist_series = pd.Series(distances)
        dist_norm = self._normalize_with_percentiles(dist_series)
        
        self.cluster_to_score = dist_norm.to_dict()
        
        # Нормальный кластер получает score 0
        if normal_cluster in self.cluster_to_score:
            self.cluster_to_score[normal_cluster] = 0.0
    
    def _fit_spectrum_projection(
        self,
        df_with_clusters: pd.DataFrame,
        spectral_analyzer: object,
        use_cluster_distribution: bool = True,
    ) -> None:
        """
        Интегрированный подход: проецирует кластеры на спектральную шкалу.
        
        Использует единую шкалу нормализации из спектрального анализа (процентили P1/P99)
        и учитывает моды (стабильные состояния) для классификации кластеров.
        
        Args:
            df_with_clusters: DataFrame с кластерами
            spectral_analyzer: Обученный SpectralAnalyzer
            use_cluster_distribution: Если True, использует медиану PC1 кластера (устойчивее к выбросам)
        """
        # Проверяем, что спектральный анализатор обучен
        if spectral_analyzer.pc1_p1 is None or spectral_analyzer.pc1_p99 is None:
            raise ValueError(
                "Спектральный анализатор не обучен. Вызовите fit_spectrum() сначала."
            )
        
        # Вычисляем PC1 для данных (если еще не вычислено)
        if "PC1" not in df_with_clusters.columns:
            df_with_clusters = spectral_analyzer.transform_pca(df_with_clusters)
        
        # Инициализируем структуры для хранения информации о кластерах
        self.cluster_distributions = {}
        self.cluster_to_mode = {}
        
        # Для каждого кластера вычисляем статистику PC1
        cluster_pc1_stats = {}
        for cluster_id in df_with_clusters["cluster"].unique():
            if cluster_id == -1:  # Пропускаем шум
                continue
            
            cluster_data = df_with_clusters[df_with_clusters["cluster"] == cluster_id]["PC1"]
            
            if len(cluster_data) == 0:
                continue
            
            # Вычисляем статистику распределения внутри кластера
            if use_cluster_distribution:
                cluster_center = cluster_data.median()  # Медиана (устойчивее к выбросам)
            else:
                cluster_center = cluster_data.mean()
            
            cluster_pc1_stats[cluster_id] = cluster_center
            
            # Сохраняем распределение кластера
            self.cluster_distributions[cluster_id] = {
                "median": float(cluster_data.median()),
                "mean": float(cluster_data.mean()),
                "p25": float(cluster_data.quantile(0.25)),
                "p75": float(cluster_data.quantile(0.75)),
                "std": float(cluster_data.std()),
                "count": int(len(cluster_data)),
            }
        
        # Проецируем центры кластеров на спектральную шкалу через процентили
        pc1_p1 = spectral_analyzer.pc1_p1
        pc1_p99 = spectral_analyzer.pc1_p99
        
        cluster_spectrum_scores = {}
        for cluster_id, pc1_center in cluster_pc1_stats.items():
            # Используем ту же формулу нормализации, что и в спектральном анализе
            spectrum_score = (pc1_center - pc1_p1) / (pc1_p99 - pc1_p1)
            spectrum_score = np.clip(spectrum_score, 0.0, 1.0)
            cluster_spectrum_scores[cluster_id] = float(spectrum_score)
        
        self.cluster_to_score = cluster_spectrum_scores
        
        # Классифицируем кластеры по модам из спектрального анализа
        if spectral_analyzer.modes:
            mode_positions = np.array([m["position"] for m in spectral_analyzer.modes])
            mode_labels = [m["label"] for m in spectral_analyzer.modes]
            
            for cluster_id, pc1_center in cluster_pc1_stats.items():
                # Находим ближайшую моду
                distances_to_modes = np.abs(pc1_center - mode_positions)
                nearest_mode_idx = np.argmin(distances_to_modes)
                
                # Классифицируем кластер по ближайшей моде
                self.cluster_to_mode[cluster_id] = mode_labels[nearest_mode_idx]
        else:
            # Если мод нет, классифицируем по спектральной шкале (как в спектральном анализе)
            for cluster_id, spectrum_score in cluster_spectrum_scores.items():
                if spectrum_score < 0.2:
                    self.cluster_to_mode[cluster_id] = "normal"
                elif spectrum_score < 0.5:
                    self.cluster_to_mode[cluster_id] = "mild"
                elif spectrum_score < 0.8:
                    self.cluster_to_mode[cluster_id] = "moderate"
                else:
                    self.cluster_to_mode[cluster_id] = "severe"
        
        # Шум (-1) обрабатывается отдельно: каждый шумовой образец получает score на основе его PC1
        # Не присваиваем фиксированный score для кластера -1, так как шумовые образцы могут иметь разные PC1
        # В transform() шумовые образцы будут получать score индивидуально на основе их PC1
        if -1 in df_with_clusters["cluster"].values:
            # Сохраняем информацию для обработки шума в transform()
            self.spectral_analyzer = spectral_analyzer
            self.pc1_p1 = pc1_p1
            self.pc1_p99 = pc1_p99
    
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
        
        # Обработка шума (-1): вычисляем score на основе реального PC1 значения (если доступно)
        noise_mask = df_result["cluster"] == -1
        if noise_mask.any():
            # Пытаемся вычислить PC1 для шума, если его нет
            if "PC1" not in df_result.columns:
                # Если PC1 нет, пытаемся вычислить через spectral_analyzer
                if hasattr(self, 'spectral_analyzer') and self.spectral_analyzer is not None:
                    df_result = self.spectral_analyzer.transform_pca(df_result)
            
            # Если PC1 доступен, вычисляем score для шума на основе их PC1
            if "PC1" in df_result.columns:
                # Вычисляем score для шума на основе их PC1 через спектральную шкалу
                if hasattr(self, 'pc1_p1') and hasattr(self, 'pc1_p99'):
                    pc1_p1 = self.pc1_p1
                    pc1_p99 = self.pc1_p99
                else:
                    # Fallback: используем процентили из всех данных (включая шум)
                    pc1_all = df_result["PC1"]
                    pc1_p1 = pc1_all.quantile(0.01)
                    pc1_p99 = pc1_all.quantile(0.99)
                
                # Вычисляем score для каждого шумового образца индивидуально
                noise_pc1 = df_result.loc[noise_mask, "PC1"]
                if len(noise_pc1) > 0 and pc1_p99 > pc1_p1:
                    noise_scores = (noise_pc1 - pc1_p1) / (pc1_p99 - pc1_p1)
                    noise_scores = np.clip(noise_scores, 0.0, 1.0)
                    df_result.loc[noise_mask, "cluster_score"] = noise_scores.values
                else:
                    # Fallback: минимальный score
                    min_score = min(self.cluster_to_score.values()) if self.cluster_to_score else 0.0
                    df_result.loc[noise_mask, "cluster_score"] = min_score
                
                # Классифицируем шум по модам (если доступны)
                if self.cluster_to_mode is not None and hasattr(self, 'spectral_analyzer') and self.spectral_analyzer is not None:
                    if self.spectral_analyzer.modes:
                        mode_positions = np.array([m["position"] for m in self.spectral_analyzer.modes])
                        mode_labels = [m["label"] for m in self.spectral_analyzer.modes]
                        
                        for idx in df_result[noise_mask].index:
                            pc1_val = df_result.loc[idx, "PC1"]
                            distances_to_modes = np.abs(pc1_val - mode_positions)
                            nearest_mode_idx = np.argmin(distances_to_modes)
                            df_result.loc[idx, "cluster_mode"] = mode_labels[nearest_mode_idx]
                    else:
                        # Классификация по спектральной шкале
                        for idx in df_result[noise_mask].index:
                            score = df_result.loc[idx, "cluster_score"]
                            if score < 0.2:
                                df_result.loc[idx, "cluster_mode"] = "normal"
                            elif score < 0.5:
                                df_result.loc[idx, "cluster_mode"] = "mild"
                            elif score < 0.8:
                                df_result.loc[idx, "cluster_mode"] = "moderate"
                            else:
                                df_result.loc[idx, "cluster_mode"] = "severe"
                else:
                    # Классификация по спектральной шкале (fallback)
                    for idx in df_result[noise_mask].index:
                        score = df_result.loc[idx, "cluster_score"]
                        if score < 0.2:
                            df_result.loc[idx, "cluster_mode"] = "normal"
                        elif score < 0.5:
                            df_result.loc[idx, "cluster_mode"] = "mild"
                        elif score < 0.8:
                            df_result.loc[idx, "cluster_mode"] = "moderate"
                        else:
                            df_result.loc[idx, "cluster_mode"] = "severe"
            else:
                # Если PC1 недоступен, используем минимальный score как fallback
                min_score = min(self.cluster_to_score.values()) if self.cluster_to_score else 0.0
                df_result.loc[noise_mask, "cluster_score"] = min_score
                df_result.loc[noise_mask, "cluster_mode"] = "noise"
        
        # Добавляем классификацию по модам (если доступна) для обычных кластеров
        if self.cluster_to_mode is not None:
            # Заполняем только те, где еще нет значения (не шум)
            non_noise_mask = df_result["cluster"] != -1
            df_result.loc[non_noise_mask, "cluster_mode"] = df_result.loc[non_noise_mask, "cluster"].map(self.cluster_to_mode)
        
        # Заполняем NaN значениями (для неизвестных кластеров, не шум)
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
    
    def get_cluster_modes(self) -> Optional[Dict[int, str]]:
        """
        Возвращает классификацию кластеров по модам (если используется spectrum_projection).
        
        Returns:
            Словарь {cluster_id: mode_label} или None
        """
        return self.cluster_to_mode
    
    def get_cluster_distributions(self) -> Optional[Dict[int, Dict]]:
        """
        Возвращает распределения внутри кластеров (если используется spectrum_projection).
        
        Returns:
            Словарь {cluster_id: {median, mean, p25, p75, std, count}} или None
        """
        return self.cluster_distributions
    
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
                "use_percentiles": self.use_percentiles,
                "percentile_low": self.percentile_low,
                "percentile_high": self.percentile_high,
                "cluster_to_mode": self.cluster_to_mode,
                "cluster_distributions": self.cluster_distributions,
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
        
        scorer = cls(
            method=data["method"],
            use_percentiles=data.get("use_percentiles", True),
            percentile_low=data.get("percentile_low", 1.0),
            percentile_high=data.get("percentile_high", 99.0),
        )
        scorer.cluster_to_score = data["cluster_to_score"]
        scorer.pathology_features = data.get("pathology_features")
        scorer.normal_cluster = data.get("normal_cluster")
        scorer.cluster_to_mode = data.get("cluster_to_mode")
        scorer.cluster_distributions = data.get("cluster_distributions")
        
        return scorer

