"""
Модуль для кластеризации патологических данных.

Поддерживает:
- HDBSCAN кластеризацию (не требует фиксированного числа кластеров)
- Альтернативные методы: Agglomerative Clustering, KMeans
- Визуализацию кластеров в UMAP-пространстве
- Анализ средних значений признаков в каждом кластере
- Интерпретацию кластеров
"""

from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import pandas as pd
import numpy as np
import pickle
import warnings

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan не установлен. Установите: pip install hdbscan")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn не установлен. Установите: pip install umap-learn")

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class ClusterAnalyzer:
    """
    Класс для кластеризации патологических данных.
    
    Поддерживает несколько методов кластеризации:
    - HDBSCAN (рекомендуется, не требует фиксированного числа кластеров)
    - Agglomerative Clustering
    - KMeans
    """
    
    def __init__(
        self,
        method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Инициализация кластеризатора.
        
        Args:
            method: Метод кластеризации ("hdbscan", "agglomerative", "kmeans")
            n_clusters: Число кластеров (для agglomerative и kmeans). 
                       Для hdbscan не используется.
            random_state: Seed для воспроизводимости
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.clusterer = None
        self.scaler = StandardScaler()
        self.labels_ = None
        self.n_clusters_ = None
        self.cluster_centers_ = None
        
        # Для UMAP визуализации
        self.umap_reducer = None
        self.umap_embedding_ = None
        
        # Метаданные
        self.feature_columns = None
        self.cluster_stats_ = None
    
    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        use_pca: bool = True,
        pca_components: Optional[int] = None,
        min_cluster_size: int = 2,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        external_pca: Optional[object] = None,
        external_scaler: Optional[object] = None,
    ) -> "ClusterAnalyzer":
        """
        Обучение кластеризатора на данных.
        
        Args:
            df: DataFrame с признаками
            feature_columns: Список признаков для кластеризации. 
                           Если None, используются все числовые колонки кроме "image"
            use_pca: Использовать ли PCA для снижения размерности перед кластеризацией
            pca_components: Число компонент PCA (если use_pca=True и external_pca=None).
                           Если external_pca передан, это число компонент для использования из него.
            min_cluster_size: Минимальный размер кластера для HDBSCAN
            min_samples: Минимальное число образцов для HDBSCAN (если None, = min_cluster_size)
            cluster_selection_epsilon: Параметр для HDBSCAN (0.0 = автоматический выбор)
            external_pca: Внешний обученный PCA объект (например, из SpectralAnalyzer).
                         Если передан, используется вместо создания нового PCA.
            external_scaler: Внешний обученный StandardScaler (например, из SpectralAnalyzer).
                           Если передан, используется вместо создания нового scaler.
        
        Returns:
            self
        """
        from sklearn.decomposition import PCA
        
        # Выбор признаков
        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "image" in numeric_cols:
                numeric_cols.remove("image")
            # Исключаем структурные элементы, используемые только для разбора по слоям
            feature_columns = [
                col for col in numeric_cols 
                if not any(x in col.lower() for x in ['surface epithelium', 'muscularis mucosae'])
            ]
        
        self.feature_columns = feature_columns
        
        # Извлечение данных
        X = df[feature_columns].fillna(0).values
        
        # Нормализация
        if external_scaler is not None:
            X_scaled = external_scaler.transform(X)
            self.scaler = external_scaler
        else:
            X_scaled = self.scaler.fit_transform(X)
        
        # PCA для снижения размерности (опционально)
        if use_pca:
            if external_pca is not None:
                # Используем внешний PCA (из спектрального анализа)
                if pca_components is not None:
                    # Используем только первые pca_components компонент из внешнего PCA
                    X_scaled = external_pca.transform(X_scaled)[:, :pca_components]
                    self.pca_components_used = pca_components
                else:
                    # Используем все компоненты из внешнего PCA
                    X_scaled = external_pca.transform(X_scaled)
                    self.pca_components_used = X_scaled.shape[1]
                self.pca = external_pca
                self.use_pca = True
                self.use_external_pca = True
            elif X_scaled.shape[1] > (pca_components or 10):
                # Создаем новый PCA
                pca = PCA(n_components=pca_components or 10, random_state=self.random_state)
                X_scaled = pca.fit_transform(X_scaled)
                self.pca = pca
                self.use_pca = True
                self.use_external_pca = False
            else:
                self.pca = None
                self.use_pca = False
                self.use_external_pca = False
        else:
            self.pca = None
            self.use_pca = False
            self.use_external_pca = False
        
        # Кластеризация
        if self.method == "hdbscan":
            # Динамическая проверка импорта (на случай, если модуль был установлен после загрузки)
            try:
                import hdbscan as hdbscan_module
            except ImportError:
                if not HDBSCAN_AVAILABLE:
                    raise ImportError("hdbscan не установлен. Установите: pip install hdbscan")
                else:
                    # Используем глобальный импорт, если он был успешным при загрузке модуля
                    hdbscan_module = hdbscan
            
            if min_samples is None:
                min_samples = min_cluster_size
            
            self.clusterer = hdbscan_module.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric='euclidean',
                cluster_selection_method='eom',
            )
            self.labels_ = self.clusterer.fit_predict(X_scaled)
            
        elif self.method == "agglomerative":
            if self.n_clusters is None:
                raise ValueError("n_clusters должен быть указан для agglomerative clustering")
            
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward',
            )
            self.labels_ = self.clusterer.fit_predict(X_scaled)
            
        elif self.method == "kmeans":
            if self.n_clusters is None:
                raise ValueError("n_clusters должен быть указан для kmeans")
            
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            self.labels_ = self.clusterer.fit_predict(X_scaled)
            self.cluster_centers_ = self.clusterer.cluster_centers_
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        # Статистика по кластерам
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        # Вычисление статистики по кластерам
        self._compute_cluster_stats(df, feature_columns)
        
        return self
    
    def _compute_cluster_stats(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str]
    ) -> None:
        """Вычисление статистики по кластерам."""
        df_with_labels = df.copy()
        df_with_labels["cluster"] = self.labels_
        
        # Исключаем шум (-1) для статистики
        df_clustered = df_with_labels[df_with_labels["cluster"] != -1]
        
        if len(df_clustered) == 0:
            self.cluster_stats_ = None
            return
        
        # Средние значения признаков по кластерам
        cluster_means = df_clustered.groupby("cluster")[feature_columns].mean()
        cluster_stds = df_clustered.groupby("cluster")[feature_columns].std()
        cluster_counts = df_clustered.groupby("cluster").size()
        
        self.cluster_stats_ = {
            "means": cluster_means,
            "stds": cluster_stds,
            "counts": cluster_counts,
            "total_samples": len(df_clustered),
            "noise_samples": len(df_with_labels[df_with_labels["cluster"] == -1]),
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение обученной кластеризации к новым данным.
        
        Args:
            df: DataFrame с признаками
        
        Returns:
            DataFrame с добавленной колонкой "cluster"
        """
        if self.clusterer is None:
            raise ValueError("Кластеризатор не обучен. Вызовите fit() сначала.")
        
        if self.feature_columns is None:
            raise ValueError("feature_columns не определены")
        
        # Извлечение данных
        X = df[self.feature_columns].fillna(0).values
        
        # Нормализация
        X_scaled = self.scaler.transform(X)
        
        # PCA (если использовался)
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
            # Если использовался external_pca с ограничением компонент, нужно ограничить
            if hasattr(self, 'use_external_pca') and self.use_external_pca and hasattr(self, 'pca_components_used'):
                X_scaled = X_scaled[:, :self.pca_components_used]
        
        # Предсказание кластеров
        if self.method == "hdbscan":
            # Динамическая проверка импорта
            try:
                import hdbscan as hdbscan_module
            except ImportError:
                if not HDBSCAN_AVAILABLE:
                    raise ImportError("hdbscan не установлен. Установите: pip install hdbscan")
                else:
                    hdbscan_module = hdbscan
            labels = self.clusterer.fit_predict(X_scaled)
        elif self.method == "agglomerative":
            labels = self.clusterer.fit_predict(X_scaled)
        elif self.method == "kmeans":
            labels = self.clusterer.predict(X_scaled)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        # Добавление меток к DataFrame
        df_result = df.copy()
        df_result["cluster"] = labels
        
        return df_result
    
    def fit_umap(
        self,
        df: pd.DataFrame,
        n_neighbors: int = 15,
        n_components: int = 2,
        min_dist: float = 0.1,
        random_state: Optional[int] = None,
    ) -> "ClusterAnalyzer":
        """
        Обучение UMAP редуктора для визуализации.
        
        Args:
            df: DataFrame с признаками
            n_neighbors: Число соседей для UMAP
            n_components: Число измерений для UMAP (обычно 2 для визуализации)
            min_dist: Минимальное расстояние между точками в UMAP
            random_state: Seed для воспроизводимости
        
        Returns:
            self
        """
        # Динамическая проверка импорта
        try:
            import umap as umap_module
        except ImportError:
            if not UMAP_AVAILABLE:
                raise ImportError("umap-learn не установлен. Установите: pip install umap-learn")
            else:
                umap_module = umap
        
        if self.feature_columns is None:
            raise ValueError("Кластеризатор не обучен. Вызовите fit() сначала.")
        
        if random_state is None:
            random_state = self.random_state
        
        # Извлечение данных
        X = df[self.feature_columns].fillna(0).values
        
        # Нормализация
        X_scaled = self.scaler.transform(X)
        
        # PCA (если использовался)
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # UMAP
        self.umap_reducer = umap_module.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            random_state=random_state,
        )
        self.umap_embedding_ = self.umap_reducer.fit_transform(X_scaled)
        
        return self
    
    def visualize_clusters(
        self,
        df: pd.DataFrame,
        label_column: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Визуализация кластеров в UMAP-пространстве.
        
        Args:
            df: DataFrame с признаками и метками кластеров
            label_column: Колонка с метками (например, "diagnosis") для цветовой маркировки
            save_path: Путь для сохранения графика
            figsize: Размер фигуры
        """
        if self.umap_embedding_ is None:
            # Обучаем UMAP если еще не обучен
            self.fit_umap(df)
        
        # Создание фигуры
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        
        # График 1: Кластеры
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            self.umap_embedding_[:, 0],
            self.umap_embedding_[:, 1],
            c=self.labels_,
            cmap='tab10',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
        )
        ax1.set_xlabel("UMAP 1", fontsize=12)
        ax1.set_ylabel("UMAP 2", fontsize=12)
        ax1.set_title(f"Кластеры ({self.method}, {self.n_clusters_} кластеров)", fontsize=14)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label="Кластер")
        
        # График 2: Метки (если есть)
        ax2 = axes[1]
        if label_column and label_column in df.columns:
            labels_unique = df[label_column].unique()
            colors_map = plt.cm.Set3(np.linspace(0, 1, len(labels_unique)))
            for i, label in enumerate(labels_unique):
                mask = df[label_column] == label
                ax2.scatter(
                    self.umap_embedding_[mask, 0],
                    self.umap_embedding_[mask, 1],
                    c=[colors_map[i]],
                    label=label,
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=1,
                )
            ax2.set_xlabel("UMAP 1", fontsize=12)
            ax2.set_ylabel("UMAP 2", fontsize=12)
            ax2.set_title(f"Метки ({label_column})", fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Если нет меток, показываем те же кластеры
            scatter2 = ax2.scatter(
                self.umap_embedding_[:, 0],
                self.umap_embedding_[:, 1],
                c=self.labels_,
                cmap='tab10',
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=1,
            )
            ax2.set_xlabel("UMAP 1", fontsize=12)
            ax2.set_ylabel("UMAP 2", fontsize=12)
            ax2.set_title("Кластеры (дубликат)", fontsize=14)
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label="Кластер")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_cluster_interpretation(self) -> Dict:
        """
        Интерпретация кластеров на основе средних значений признаков.
        
        Returns:
            Словарь с интерпретацией каждого кластера
        """
        if self.cluster_stats_ is None:
            return {}
        
        interpretation = {}
        cluster_means = self.cluster_stats_["means"]
        
        for cluster_id in cluster_means.index:
            # Топ-5 признаков с наибольшими значениями
            top_features = cluster_means.loc[cluster_id].nlargest(5)
            
            # Интерпретация на основе признаков
            features_str = ", ".join([f"{feat}={val:.2f}" for feat, val in top_features.items()])
            
            # Попытка автоматической интерпретации
            interpretation_text = self._interpret_cluster(top_features)
            
            interpretation[cluster_id] = {
                "top_features": top_features.to_dict(),
                "features_str": features_str,
                "interpretation": interpretation_text,
                "n_samples": int(self.cluster_stats_["counts"].loc[cluster_id]),
            }
        
        return interpretation
    
    def _interpret_cluster(self, top_features: pd.Series) -> str:
        """
        Автоматическая интерпретация кластера на основе признаков.
        
        Args:
            top_features: Топ-5 признаков с наибольшими значениями
        
        Returns:
            Текстовая интерпретация
        """
        features_lower = {k.lower(): v for k, v in top_features.items()}
        
        # Проверка на паттерны
        if any("neutrophils" in f for f in features_lower.keys()):
            if any("crypts" in f for f in features_lower.keys()):
                return "Активное воспаление (высокие нейтрофилы + крипты)"
        
        if any("plasma" in f for f in features_lower.keys()):
            if any("eoe" in f or "eosinophils" in f for f in features_lower.keys()):
                return "Аллергия/EoE (высокие плазматические клетки + эозинофилы)"
        
        if any("dysplasia" in f for f in features_lower.keys()):
            if any("meta" in f for f in features_lower.keys()):
                return "Премалигнантное состояние (дисплазия + метаплазия)"
        
        if all(v < 0.5 for v in features_lower.values()):
            return "Нормальные биоптаты (низкие значения всех маркеров)"
        
        if any("mild" in f for f in features_lower.keys()):
            return "Легкая патология (Mild изменения)"
        
        if any("moderate" in f for f in features_lower.keys()):
            return "Умеренная патология (Moderate изменения)"
        
        return "Смешанная патология"
    
    def get_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Вычисление метрик качества кластеризации.
        
        Args:
            df: DataFrame с признаками
        
        Returns:
            Словарь с метриками
        """
        if self.labels_ is None:
            raise ValueError("Кластеризатор не обучен. Вызовите fit() сначала.")
        
        # Извлечение данных
        X = df[self.feature_columns].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        # PCA (если использовался)
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Исключаем шум (-1) для метрик
        mask = self.labels_ != -1
        if mask.sum() < 2:
            return {
                "silhouette_score": np.nan,
                "calinski_harabasz_score": np.nan,
                "davies_bouldin_score": np.nan,
                "n_clusters": self.n_clusters_,
                "n_noise": (self.labels_ == -1).sum(),
            }
        
        X_filtered = X_scaled[mask]
        labels_filtered = self.labels_[mask]
        
        metrics = {
            "n_clusters": self.n_clusters_,
            "n_noise": (self.labels_ == -1).sum(),
            "n_samples": len(self.labels_),
        }
        
        if len(set(labels_filtered)) > 1:
            metrics["silhouette_score"] = silhouette_score(X_filtered, labels_filtered)
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_filtered, labels_filtered)
            metrics["davies_bouldin_score"] = davies_bouldin_score(X_filtered, labels_filtered)
        else:
            metrics["silhouette_score"] = np.nan
            metrics["calinski_harabasz_score"] = np.nan
            metrics["davies_bouldin_score"] = np.nan
        
        return metrics
    
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
                "clusterer": self.clusterer,
                "scaler": self.scaler,
                "pca": getattr(self, "pca", None),
                "use_pca": getattr(self, "use_pca", False),
                "labels_": self.labels_,
                "n_clusters_": self.n_clusters_,
                "cluster_centers_": self.cluster_centers_,
                "feature_columns": self.feature_columns,
                "cluster_stats_": self.cluster_stats_,
                "method": self.method,
                "random_state": self.random_state,
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClusterAnalyzer":
        """
        Загрузка обученной модели.
        
        Args:
            path: Путь к файлу модели
        
        Returns:
            Загруженный ClusterAnalyzer
        """
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        analyzer = cls(
            method=data["method"],
            random_state=data["random_state"],
        )
        
        analyzer.clusterer = data["clusterer"]
        analyzer.scaler = data["scaler"]
        analyzer.pca = data.get("pca")
        analyzer.use_pca = data.get("use_pca", False)
        analyzer.labels_ = data["labels_"]
        analyzer.n_clusters_ = data["n_clusters_"]
        analyzer.cluster_centers_ = data.get("cluster_centers_")
        analyzer.feature_columns = data["feature_columns"]
        analyzer.cluster_stats_ = data.get("cluster_stats_")
        
        return analyzer

