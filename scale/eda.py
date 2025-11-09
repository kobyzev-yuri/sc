"""
Модуль для разведочного анализа данных (EDA).

Поддерживает:
- Визуализацию распределений признаков
- Корреляционный анализ
- Dimensionality reduction визуализацию (UMAP, t-SNE, PCA)
- Статистический анализ
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn не установлен. Установите: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats


def visualize_distributions_by_group(
    df: pd.DataFrame,
    feature: str,
    group_column: str,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Визуализирует распределение признака по группам (boxplot и violin plot).
    
    Args:
        df: DataFrame с данными
        feature: Название признака для визуализации
        group_column: Название колонки с группами
        figsize: Размер фигуры
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Boxplot
    ax1 = axes[0]
    groups = df[group_column].unique()
    data_by_group = [df[df[group_column] == group][feature].dropna() for group in groups]
    ax1.boxplot(data_by_group, labels=groups)
    ax1.set_xlabel(group_column, fontsize=12)
    ax1.set_ylabel(feature, fontsize=12)
    ax1.set_title(f"Boxplot: {feature} по группам", fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    ax2 = axes[1]
    sns.violinplot(data=df, x=group_column, y=feature, ax=ax2)
    ax2.set_title(f"Violin plot: {feature} по группам", fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_pca_scatter(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None,
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Визуализирует данные в пространстве PCA (PC1 vs PC2).
    
    Args:
        df: DataFrame с признаками
        feature_columns: Список признаков для PCA
        label_column: Колонка с метками для цветовой маркировки
        n_components: Число компонент PCA
        figsize: Размер фигуры
        save_path: Путь для сохранения графика
    
    Returns:
        DataFrame с добавленными колонками PC1, PC2, ...
    """
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "image" in numeric_cols:
            numeric_cols.remove("image")
        feature_columns = numeric_cols
    
    # Подготовка данных
    X = df[feature_columns].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Добавляем PC компоненты к DataFrame
    df_result = df.copy()
    for i in range(n_components):
        df_result[f"PC{i+1}"] = X_pca[:, i]
    
    # Визуализация
    fig, ax = plt.subplots(figsize=figsize)
    
    if label_column and label_column in df.columns:
        # Цветовая маркировка по группам
        groups = df[label_column].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for i, group in enumerate(groups):
            mask = df[label_column] == group
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[colors[i]],
                label=group,
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=1,
            )
        ax.legend()
    else:
        # Без группировки
        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
        )
    
    # Объясненная дисперсия
    explained_var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("PCA Scatter Plot", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return df_result


def visualize_umap(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Визуализирует данные в UMAP-пространстве.
    
    Args:
        df: DataFrame с признаками
        feature_columns: Список признаков для UMAP
        label_column: Колонка с метками для цветовой маркировки
        n_neighbors: Число соседей для UMAP
        min_dist: Минимальное расстояние для UMAP
        random_state: Seed для воспроизводимости
        figsize: Размер фигуры
        save_path: Путь для сохранения графика
    
    Returns:
        DataFrame с добавленными колонками UMAP1, UMAP2
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn не установлен. Установите: pip install umap-learn")
    
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "image" in numeric_cols:
            numeric_cols.remove("image")
        feature_columns = numeric_cols
    
    # Подготовка данных
    X = df[feature_columns].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X_scaled)
    
    # Добавляем UMAP координаты к DataFrame
    df_result = df.copy()
    df_result["UMAP1"] = X_umap[:, 0]
    df_result["UMAP2"] = X_umap[:, 1]
    
    # Визуализация
    fig, ax = plt.subplots(figsize=figsize)
    
    if label_column and label_column in df.columns:
        # Цветовая маркировка по группам
        groups = df[label_column].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for i, group in enumerate(groups):
            mask = df[label_column] == group
            ax.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                c=[colors[i]],
                label=group,
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=1,
            )
        ax.legend()
    else:
        # Без группировки
        ax.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
        )
    
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("UMAP Visualization", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return df_result


def statistical_tests(
    df: pd.DataFrame,
    feature: str,
    group_column: str,
) -> Dict[str, float]:
    """
    Выполняет статистические тесты для сравнения групп.
    
    Args:
        df: DataFrame с данными
        feature: Название признака для тестирования
        group_column: Название колонки с группами
    
    Returns:
        Словарь с результатами тестов
    """
    groups = df[group_column].unique()
    
    if len(groups) != 2:
        return {"error": "Тесты поддерживают только 2 группы"}
    
    group1_data = df[df[group_column] == groups[0]][feature].dropna()
    group2_data = df[df[group_column] == groups[1]][feature].dropna()
    
    results = {}
    
    # T-test (предполагает нормальность)
    try:
        t_stat, t_pvalue = scipy_stats.ttest_ind(group1_data, group2_data)
        results["t_test_statistic"] = t_stat
        results["t_test_pvalue"] = t_pvalue
    except Exception as e:
        results["t_test_error"] = str(e)
    
    # Mann-Whitney U test (непараметрический)
    try:
        u_stat, u_pvalue = scipy_stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        results["mannwhitney_statistic"] = u_stat
        results["mannwhitney_pvalue"] = u_pvalue
    except Exception as e:
        results["mannwhitney_error"] = str(e)
    
    # Описательная статистика
    results["group1_mean"] = group1_data.mean()
    results["group2_mean"] = group2_data.mean()
    results["group1_std"] = group1_data.std()
    results["group2_std"] = group2_data.std()
    
    return results

