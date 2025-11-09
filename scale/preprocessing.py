"""
Модуль для препроцессинга данных перед кластерным анализом.

Поддерживает:
- Анализ корреляций между признаками
- Удаление избыточных признаков
- Визуализацию распределений признаков
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    feature_columns: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Находит пары признаков с высокой корреляцией.
    
    Args:
        df: DataFrame с признаками
        threshold: Порог корреляции (по умолчанию 0.95)
        feature_columns: Список признаков для анализа. Если None, используются все числовые колонки.
    
    Returns:
        Список кортежей (признак1, признак2, корреляция)
    """
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "image" in numeric_cols:
            numeric_cols.remove("image")
        feature_columns = numeric_cols
    
    # Вычисляем корреляционную матрицу
    corr_matrix = df[feature_columns].corr()
    
    # Находим пары с высокой корреляцией
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                highly_correlated.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
    
    return highly_correlated


def remove_redundant_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    feature_columns: Optional[List[str]] = None,
    method: str = "first",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Удаляет избыточные признаки с высокой корреляцией.
    
    Args:
        df: DataFrame с признаками
        threshold: Порог корреляции для удаления
        feature_columns: Список признаков для анализа
        method: Метод выбора признака для удаления ("first" - удаляет первый, "variance" - удаляет с меньшей дисперсией)
    
    Returns:
        Tuple (DataFrame без избыточных признаков, список удаленных признаков)
    """
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "image" in numeric_cols:
            numeric_cols.remove("image")
        feature_columns = numeric_cols
    
    # Находим высоко коррелированные пары
    highly_correlated = find_highly_correlated_features(df, threshold, feature_columns)
    
    if not highly_correlated:
        return df, []
    
    # Определяем, какие признаки удалить
    features_to_remove = set()
    
    for feat1, feat2, corr_value in highly_correlated:
        if method == "variance":
            # Удаляем признак с меньшей дисперсией
            var1 = df[feat1].var()
            var2 = df[feat2].var()
            if var1 < var2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        else:  # method == "first"
            # Удаляем первый признак в паре
            features_to_remove.add(feat1)
    
    features_to_remove = list(features_to_remove)
    
    # Удаляем признаки
    df_result = df.drop(columns=features_to_remove)
    
    return df_result, features_to_remove


def visualize_correlations(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Визуализирует корреляционную матрицу признаков.
    
    Args:
        df: DataFrame с признаками
        feature_columns: Список признаков для анализа
        figsize: Размер фигуры
        save_path: Путь для сохранения графика
    """
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "image" in numeric_cols:
            numeric_cols.remove("image")
        feature_columns = numeric_cols
    
    # Вычисляем корреляционную матрицу
    corr_matrix = df[feature_columns].corr()
    
    # Создаем heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Корреляционная матрица признаков", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_feature_distributions(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Визуализирует распределения признаков (гистограммы и boxplots).
    
    Args:
        df: DataFrame с признаками
        feature_columns: Список признаков для визуализации
        n_cols: Число колонок в сетке графиков
        figsize: Размер фигуры
        save_path: Путь для сохранения графика
    """
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "image" in numeric_cols:
            numeric_cols.remove("image")
        feature_columns = numeric_cols
    
    n_features = len(feature_columns)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]
        
        # Гистограмма
        ax.hist(
            df[feature].dropna(),
            bins=20,
            alpha=0.7,
            edgecolor="black",
            color="skyblue",
        )
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Распределение: {feature}", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Скрываем лишние оси
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

