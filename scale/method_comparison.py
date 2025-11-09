"""
Модуль для сравнения разных методов построения шкалы патологии.

Поддерживает сравнение:
- PCA Scoring (простая нормализация PC1)
- Spectral Analysis (PCA + KDE/GMM + моды)
- Cluster-based Scoring (кластеризация + маппинг кластеров на шкалу)

Позволяет:
- Загрузить результаты разных методов
- Сравнить распределения scores
- Вычислить корреляции между методами
- Визуализировать сравнение
- Рекомендовать лучший метод
"""

from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

from . import pca_scoring
from . import spectral_analysis
from . import clustering
from . import cluster_scoring


class MethodComparison:
    """
    Класс для сравнения разных методов построения шкалы патологии.
    """
    
    def __init__(self):
        self.results: Dict[str, pd.DataFrame] = {}
        self.methods: Dict[str, str] = {}  # имя -> тип метода
        self.models: Dict[str, Union[pca_scoring.PCAScorer, spectral_analysis.SpectralAnalyzer, clustering.ClusterAnalyzer]] = {}
        self.scorers: Dict[str, cluster_scoring.ClusterScorer] = {}
    
    def add_pca_result(
        self,
        name: str,
        df_scores: pd.DataFrame,
        scorer: Optional[pca_scoring.PCAScorer] = None,
    ) -> "MethodComparison":
        """
        Добавляет результат PCA scoring.
        
        Args:
            name: Имя метода (для идентификации)
            df_scores: DataFrame с колонками ["image", "PC1_norm"] или ["image", "score"]
            scorer: Обученный PCAScorer (опционально)
        
        Returns:
            self
        """
        # Проверяем наличие нужных колонок
        if "PC1_norm" in df_scores.columns:
            score_col = "PC1_norm"
        elif "score" in df_scores.columns:
            score_col = "score"
        else:
            raise ValueError("DataFrame должен содержать колонку 'PC1_norm' или 'score'")
        
        # Создаем стандартизированный DataFrame
        result_df = pd.DataFrame({
            "image": df_scores["image"].values,
            "score": df_scores[score_col].values,
            "method": "pca_scoring"
        })
        
        self.results[name] = result_df
        self.methods[name] = "pca_scoring"
        if scorer:
            self.models[name] = scorer
        
        return self
    
    def add_spectral_result(
        self,
        name: str,
        df_scores: pd.DataFrame,
        analyzer: Optional[spectral_analysis.SpectralAnalyzer] = None,
    ) -> "MethodComparison":
        """
        Добавляет результат Spectral Analysis.
        
        Args:
            name: Имя метода
            df_scores: DataFrame с колонками ["image", "PC1_spectrum"] или ["image", "score"]
            analyzer: Обученный SpectralAnalyzer (опционально)
        
        Returns:
            self
        """
        # Проверяем наличие нужных колонок
        if "PC1_spectrum" in df_scores.columns:
            score_col = "PC1_spectrum"
        elif "score" in df_scores.columns:
            score_col = "score"
        else:
            raise ValueError("DataFrame должен содержать колонку 'PC1_spectrum' или 'score'")
        
        result_df = pd.DataFrame({
            "image": df_scores["image"].values,
            "score": df_scores[score_col].values,
            "method": "spectral_analysis"
        })
        
        self.results[name] = result_df
        self.methods[name] = "spectral_analysis"
        if analyzer:
            self.models[name] = analyzer
        
        return self
    
    def add_cluster_result(
        self,
        name: str,
        df_scores: pd.DataFrame,
        clusterer: Optional[clustering.ClusterAnalyzer] = None,
        scorer: Optional[cluster_scoring.ClusterScorer] = None,
    ) -> "MethodComparison":
        """
        Добавляет результат Cluster-based Scoring.
        
        Args:
            name: Имя метода
            df_scores: DataFrame с колонками ["image", "cluster_score"]
            clusterer: Обученный ClusterAnalyzer (опционально)
            scorer: Обученный ClusterScorer (опционально)
        
        Returns:
            self
        """
        if "cluster_score" not in df_scores.columns:
            raise ValueError("DataFrame должен содержать колонку 'cluster_score'")
        
        result_df = pd.DataFrame({
            "image": df_scores["image"].values,
            "score": df_scores["cluster_score"].values,
            "method": "cluster_scoring"
        })
        
        self.results[name] = result_df
        self.methods[name] = "cluster_scoring"
        if clusterer:
            self.models[name] = clusterer
        if scorer:
            self.scorers[name] = scorer
        
        return self
    
    def compare_scores(self) -> pd.DataFrame:
        """
        Объединяет все scores в один DataFrame для сравнения.
        
        Returns:
            DataFrame с колонками: image, {method1}_score, {method2}_score, ...
        """
        if len(self.results) == 0:
            raise ValueError("Нет результатов для сравнения. Добавьте результаты через add_*_result()")
        
        # Начинаем с первого метода
        comparison = pd.DataFrame({"image": self.results[list(self.results.keys())[0]]["image"]})
        
        # Добавляем scores для каждого метода
        for name, df_result in self.results.items():
            comparison = comparison.merge(
                df_result[["image", "score"]],
                on="image",
                how="outer",
                suffixes=("", f"_{name}")
            )
            # Переименовываем колонку score
            if f"score_{name}" not in comparison.columns:
                comparison = comparison.rename(columns={"score": f"score_{name}"})
        
        return comparison
    
    def compute_correlations(self) -> pd.DataFrame:
        """
        Вычисляет корреляции между всеми методами.
        
        Returns:
            DataFrame с корреляционной матрицей
        """
        comparison = self.compare_scores()
        
        # Извлекаем только score колонки
        score_cols = [col for col in comparison.columns if col.startswith("score_")]
        
        if len(score_cols) < 2:
            raise ValueError("Нужно минимум 2 метода для вычисления корреляций")
        
        # Создаем матрицу корреляций
        corr_matrix = comparison[score_cols].corr(method='pearson')
        
        # Также вычисляем Spearman корреляции
        spearman_corr = comparison[score_cols].corr(method='spearman')
        
        # Создаем DataFrame с обеими метриками
        methods = [col.replace("score_", "") for col in score_cols]
        corr_data = []
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i <= j:
                    corr_data.append({
                        "Метод 1": method1,
                        "Метод 2": method2,
                        "Pearson r": corr_matrix.iloc[i, j],
                        "Spearman ρ": spearman_corr.iloc[i, j],
                    })
        
        return pd.DataFrame(corr_data)
    
    def compute_statistics(self) -> pd.DataFrame:
        """
        Вычисляет статистику для каждого метода.
        
        Returns:
            DataFrame со статистикой
        """
        stats_rows = []
        
        for name, df_result in self.results.items():
            scores = df_result["score"].dropna()
            
            stats_rows.append({
                "Метод": name,
                "Тип": self.methods[name],
                "N образцов": len(scores),
                "Mean": f"{scores.mean():.4f}",
                "Std": f"{scores.std():.4f}",
                "Min": f"{scores.min():.4f}",
                "Max": f"{scores.max():.4f}",
                "Median": f"{scores.median():.4f}",
                "Q25": f"{scores.quantile(0.25):.4f}",
                "Q75": f"{scores.quantile(0.75):.4f}",
            })
        
        return pd.DataFrame(stats_rows)
    
    def compute_distribution_similarity(self) -> pd.DataFrame:
        """
        Вычисляет схожесть распределений между методами (Jensen-Shannon divergence).
        
        Returns:
            DataFrame с метриками схожести распределений
        """
        comparison = self.compare_scores()
        score_cols = [col for col in comparison.columns if col.startswith("score_")]
        
        if len(score_cols) < 2:
            raise ValueError("Нужно минимум 2 метода для сравнения распределений")
        
        # Нормализуем scores в гистограммы (для JS divergence)
        methods = [col.replace("score_", "") for col in score_cols]
        similarity_data = []
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    scores1 = comparison[score_cols[i]].dropna().values
                    scores2 = comparison[score_cols[j]].dropna().values
                    
                    # Создаем гистограммы с одинаковыми bins
                    bins = np.linspace(0, 1, 51)  # 50 bins от 0 до 1
                    hist1, _ = np.histogram(scores1, bins=bins, density=True)
                    hist2, _ = np.histogram(scores2, bins=bins, density=True)
                    
                    # Вычисляем Jensen-Shannon divergence
                    js_div = jensenshannon(hist1, hist2)
                    
                    similarity_data.append({
                        "Метод 1": method1,
                        "Метод 2": method2,
                        "JS Divergence": f"{js_div:.4f}",
                        "Пояснение": "0 = идентичные распределения, 1 = максимально разные"
                    })
        
        return pd.DataFrame(similarity_data)
    
    def visualize_comparison(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (16, 12),
    ) -> None:
        """
        Визуализирует сравнение методов.
        
        Args:
            save_path: Путь для сохранения графика
            figsize: Размер фигуры
        """
        if len(self.results) == 0:
            raise ValueError("Нет результатов для визуализации")
        
        comparison = self.compare_scores()
        score_cols = [col for col in comparison.columns if col.startswith("score_")]
        n_methods = len(score_cols)
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # График 1: Распределения scores (гистограммы)
        ax1 = axes[0, 0]
        for col in score_cols:
            scores = comparison[col].dropna()
            method_name = col.replace("score_", "")
            ax1.hist(scores, bins=20, alpha=0.6, label=method_name, density=True)
        ax1.set_xlabel("Score (0-1)", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Распределение scores по методам", fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Boxplot сравнение
        ax2 = axes[0, 1]
        score_data = [comparison[col].dropna().values for col in score_cols]
        method_labels = [col.replace("score_", "") for col in score_cols]
        ax2.boxplot(score_data, labels=method_labels)
        ax2.set_ylabel("Score (0-1)", fontsize=12)
        ax2.set_title("Boxplot сравнение методов", fontsize=13)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # График 3: Scatter plot (если есть 2+ метода)
        if n_methods >= 2:
            ax3 = axes[1, 0]
            col1, col2 = score_cols[0], score_cols[1]
            merged = comparison[[col1, col2]].dropna()
            
            ax3.scatter(merged[col1], merged[col2], alpha=0.6, s=100)
            ax3.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
            
            # Вычисляем корреляцию
            if len(merged) > 1:
                corr, _ = pearsonr(merged[col1], merged[col2])
                ax3.set_title(f"Сравнение {method_labels[0]} vs {method_labels[1]}\n(Pearson r={corr:.3f})", fontsize=13)
            
            ax3.set_xlabel(f"{method_labels[0]} score", fontsize=12)
            ax3.set_ylabel(f"{method_labels[1]} score", fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # График 4: Корреляционная матрица
        if n_methods >= 2:
            ax4 = axes[1, 1]
            corr_matrix = comparison[score_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax4, vmin=-1, vmax=1)
            ax4.set_title("Корреляционная матрица (Pearson)", fontsize=13)
        
        # График 5: Сравнение по образцам (если образцов не слишком много)
        if len(comparison) <= 50:
            ax5 = axes[2, 0]
            x = np.arange(len(comparison))
            width = 0.8 / n_methods
            
            for i, col in enumerate(score_cols):
                scores = comparison[col].fillna(0).values
                ax5.bar(x + i * width, scores, width, label=col.replace("score_", ""), alpha=0.7)
            
            ax5.set_xlabel("Образец", fontsize=12)
            ax5.set_ylabel("Score", fontsize=12)
            ax5.set_title("Сравнение scores по образцам", fontsize=13)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.set_xticks(x + width * (n_methods - 1) / 2)
            ax5.set_xticklabels(comparison["image"].values, rotation=45, ha="right", fontsize=8)
        
        # График 6: Методы по типам
        ax6 = axes[2, 1]
        method_types = {}
        for name, method_type in self.methods.items():
            if method_type not in method_types:
                method_types[method_type] = []
            method_types[method_type].append(name)
        
        type_counts = {k: len(v) for k, v in method_types.items()}
        ax6.bar(type_counts.keys(), type_counts.values(), alpha=0.7)
        ax6.set_ylabel("Число методов", fontsize=12)
        ax6.set_title("Распределение методов по типам", fontsize=13)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def recommend_best(
        self,
        criteria: str = "consistency",
        reference_method: Optional[str] = None,
    ) -> Dict:
        """
        Рекомендует лучший метод на основе различных критериев.
        
        Args:
            criteria: Критерий выбора ("consistency", "spread", "correlation")
            reference_method: Референсный метод для сравнения (если есть)
        
        Returns:
            Словарь с рекомендацией
        """
        if len(self.results) < 2:
            return {
                "best": None,
                "reason": "Нужно минимум 2 метода для сравнения"
            }
        
        comparison = self.compare_scores()
        score_cols = [col for col in comparison.columns if col.startswith("score_")]
        methods = [col.replace("score_", "") for col in score_cols]
        
        if criteria == "consistency":
            # Выбираем метод с наименьшей вариативностью (std)
            stds = {}
            for col in score_cols:
                scores = comparison[col].dropna()
                stds[col.replace("score_", "")] = scores.std()
            
            best = min(stds, key=stds.get)
            return {
                "best": best,
                "reason": f"Наименьшая вариативность (std={stds[best]:.4f})",
                "scores": stds
            }
        
        elif criteria == "spread":
            # Выбираем метод с наибольшим разбросом (range)
            ranges = {}
            for col in score_cols:
                scores = comparison[col].dropna()
                ranges[col.replace("score_", "")] = scores.max() - scores.min()
            
            best = max(ranges, key=ranges.get)
            return {
                "best": best,
                "reason": f"Наибольший разброс (range={ranges[best]:.4f})",
                "scores": ranges
            }
        
        elif criteria == "correlation":
            if reference_method is None:
                # Используем первый метод как референс
                reference_method = methods[0]
            
            # Выбираем метод с наибольшей корреляцией с референсом
            ref_col = f"score_{reference_method}"
            if ref_col not in score_cols:
                return {"best": None, "reason": f"Референсный метод '{reference_method}' не найден"}
            
            corrs = {}
            for col in score_cols:
                if col != ref_col:
                    merged = comparison[[ref_col, col]].dropna()
                    if len(merged) > 1:
                        corr, _ = pearsonr(merged[ref_col], merged[col])
                        corrs[col.replace("score_", "")] = abs(corr)
            
            if corrs:
                best = max(corrs, key=corrs.get)
                return {
                    "best": best,
                    "reason": f"Наибольшая корреляция с {reference_method} (r={corrs[best]:.4f})",
                    "scores": corrs
                }
        
        return {"best": None, "reason": "Не удалось определить лучший метод"}

