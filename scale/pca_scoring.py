"""
PCA анализ и создание шкалы оценки патологии.

Модуль для снижения размерности данных через PCA и создания
нормализованной шкалы PC1_norm от 0 до 1.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle


class PCAScorer:
    """
    Класс для PCA анализа и создания шкалы оценки патологии.
    """

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.feature_columns: Optional[list[str]] = None
        self.pc1_min: Optional[float] = None
        self.pc1_max: Optional[float] = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
    ) -> "PCAScorer":
        """
        Обучает StandardScaler и PCA на данных.

        Args:
            df: DataFrame с признаками
            feature_columns: Список колонок для использования. Если None, выбираются все числовые колонки.

        Returns:
            self
        """
        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "image" in numeric_cols:
                numeric_cols.remove("image")
            # Используем все числовые колонки (включая структурные, если они есть)
            # Структурные признаки исключаются только если явно не переданы в feature_columns
            feature_columns = numeric_cols

        self.feature_columns = feature_columns
        X = df[feature_columns].fillna(0).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=None)
        X_pca = self.pca.fit_transform(X_scaled)

        self.pc1_min = X_pca[:, 0].min()
        self.pc1_max = X_pca[:, 0].max()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует данные и добавляет колонки PC1 и PC1_norm.

        Args:
            df: DataFrame с признаками

        Returns:
            DataFrame с добавленными колонками PC1 и PC1_norm
        """
        if self.scaler is None or self.pca is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        if self.feature_columns is None:
            raise ValueError("feature_columns не установлены.")

        df_result = df.copy()
        X = df_result[self.feature_columns].fillna(0).values

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        df_result["PC1"] = X_pca[:, 0]
        df_result["PC1_norm"] = (df_result["PC1"] - self.pc1_min) / (
            self.pc1_max - self.pc1_min
        )

        return df_result

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Обучает модель и преобразует данные.

        Args:
            df: DataFrame с признаками
            feature_columns: Список колонок для использования

        Returns:
            DataFrame с добавленными колонками PC1 и PC1_norm
        """
        return self.fit(df, feature_columns).transform(df)

    def get_feature_importance(self) -> pd.Series:
        """
        Возвращает важность признаков через loadings первой главной компоненты.

        Returns:
            Series с важностью признаков, отсортированная по абсолютному значению
        """
        if self.pca is None or self.feature_columns is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        loadings = pd.Series(
            self.pca.components_[0], index=self.feature_columns
        )
        return loadings.sort_values(key=abs, ascending=False)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Сохраняет модель в файл.

        Args:
            filepath: Путь для сохранения модели
        """
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "pca": self.pca,
                    "feature_columns": self.feature_columns,
                    "pc1_min": self.pc1_min,
                    "pc1_max": self.pc1_max,
                },
                f,
            )

    def load(self, filepath: Union[str, Path]) -> "PCAScorer":
        """
        Загружает модель из файла.

        Args:
            filepath: Путь к файлу модели

        Returns:
            self
        """
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.feature_columns = data["feature_columns"]
        self.pc1_min = data["pc1_min"]
        self.pc1_max = data["pc1_max"]

        return self

