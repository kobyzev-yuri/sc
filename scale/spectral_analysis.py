"""
Спектральный анализ патологий и создание универсальной шкалы оценки.

Модуль для выявления стабильных состояний (мод) в распределении патологий
и построения спектральной шкалы оценки от 0 до 1.

Методы:
- PCA: снижение размерности через главные компоненты
- KDE: оценка плотности распределения для поиска мод
- GMM: моделирование смеси распределений для выявления состояний
- HDBSCAN: кластеризация для валидации мод
"""

from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.signal import find_peaks
import pickle


class SpectralAnalyzer:
    """
    Класс для спектрального анализа патологий и создания универсальной шкалы.
    
    Выявляет стабильные состояния (моды) в распределении патологий через
    комбинацию PCA, KDE, GMM и кластеризации.
    """

    def __init__(self):
        # PCA компоненты
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.feature_columns: Optional[list[str]] = None
        
        # Спектральные параметры
        self.pc1_percentiles: Optional[dict[str, float]] = None
        self.modes: Optional[list[dict]] = None
        self.gmm: Optional[GaussianMixture] = None
        
        # Параметры нормализации
        self.pc1_p1: Optional[float] = None
        self.pc1_p99: Optional[float] = None

    def fit_pca(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
    ) -> "SpectralAnalyzer":
        """
        Обучает PCA на данных для снижения размерности.

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
            feature_columns = numeric_cols

        self.feature_columns = feature_columns
        X = df[feature_columns].fillna(0).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=None)
        self.pca.fit(X_scaled)

        return self

    def transform_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует данные через PCA и добавляет колонку PC1.

        Args:
            df: DataFrame с признаками

        Returns:
            DataFrame с добавленной колонкой PC1
        """
        if self.scaler is None or self.pca is None:
            raise ValueError("PCA не обучен. Вызовите fit_pca() сначала.")

        if self.feature_columns is None:
            raise ValueError("feature_columns не установлены.")

        df_result = df.copy()
        X = df_result[self.feature_columns].fillna(0).values

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        df_result["PC1"] = X_pca[:, 0]

        return df_result

    def fit_spectrum(
        self,
        df: pd.DataFrame,
        pc1_column: str = "PC1",
        use_percentiles: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ) -> "SpectralAnalyzer":
        """
        Анализирует спектр распределения PC1 и находит стабильные состояния.

        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями
            use_percentiles: Использовать процентили для буферных зон
            percentile_low: Нижний процентиль (по умолчанию 1.0)
            percentile_high: Верхний процентиль (по умолчанию 99.0)

        Returns:
            self
        """
        if pc1_column not in df.columns:
            raise ValueError(f"Колонка {pc1_column} не найдена в DataFrame")

        pc1_values = df[pc1_column].dropna().values

        # Вычисление процентилей для буферных зон
        if use_percentiles:
            self.pc1_p1 = np.percentile(pc1_values, percentile_low)
            self.pc1_p99 = np.percentile(pc1_values, percentile_high)
        else:
            self.pc1_p1 = pc1_values.min()
            self.pc1_p99 = pc1_values.max()

        self.pc1_percentiles = {
            "p1": self.pc1_p1,
            "p99": self.pc1_p99,
            "median": np.median(pc1_values),
            "mean": np.mean(pc1_values),
            "std": np.std(pc1_values),
        }

        # Поиск мод через KDE
        self.modes = self._find_modes_kde(pc1_values)

        return self

    def _find_modes_kde(
        self, values: np.ndarray, bandwidth: Optional[float] = None
    ) -> list[dict]:
        """
        Находит моды (локальные максимумы) в распределении через KDE.

        Args:
            values: Массив значений PC1
            bandwidth: Ширина окна для KDE. Если None, используется правило Скотта.

        Returns:
            Список словарей с информацией о модах: [{"position": float, "density": float, "label": str}, ...]
        """
        if len(values) < 2:
            return []

        # Оценка плотности через KDE
        if bandwidth is None:
            bandwidth = stats.gaussian_kde(values).scotts_factor() * np.std(values)

        kde = stats.gaussian_kde(values, bw_method=bandwidth)

        # Создание сетки для оценки плотности
        x_min, x_max = values.min(), values.max()
        x_range = x_max - x_min
        x_grid = np.linspace(
            x_min - 0.1 * x_range, x_max + 0.1 * x_range, num=1000
        )
        density = kde(x_grid)

        # Поиск локальных максимумов (пиков)
        # Уменьшаем порог для поиска большего числа мод
        peaks, properties = find_peaks(
            density, height=np.max(density) * 0.05, distance=len(x_grid) // 20
        )

        modes = []
        for peak_idx in peaks:
            position = x_grid[peak_idx]
            peak_density = density[peak_idx]

            # Определение типа моды на основе позиции относительно медианы
            # Используем более мягкую логику: если мода близка к медиане, используем спектральную шкалу
            median = self.pc1_percentiles["median"]
            
            # Если мода значительно ниже медианы - normal
            if position < median - 0.5 * self.pc1_percentiles.get("std", 1.0):
                label = "normal"
            # Если мода значительно выше медианы - pathology
            elif position > median + 0.5 * self.pc1_percentiles.get("std", 1.0):
                label = "pathology"
            # Если мода близка к медиане - используем спектральную шкалу для классификации
            else:
                # Преобразуем в спектральную шкалу для более точной классификации
                if self.pc1_p1 is not None and self.pc1_p99 is not None:
                    mode_spectrum = (position - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
                    mode_spectrum = np.clip(mode_spectrum, 0.0, 1.0)
                    
                    if mode_spectrum < 0.2:
                        label = "normal"
                    elif mode_spectrum < 0.5:
                        label = "mild"
                    elif mode_spectrum < 0.8:
                        label = "moderate"
                    else:
                        label = "severe"
                else:
                    # Fallback: используем медиану
                    if position < median:
                        label = "normal"
                    else:
                        label = "pathology"

            modes.append(
                {
                    "position": float(position),
                    "density": float(peak_density),
                    "label": label,
                }
            )

        # Сортировка по позиции
        modes.sort(key=lambda x: x["position"])

        return modes

    def fit_gmm(
        self,
        df: pd.DataFrame,
        pc1_column: str = "PC1",
        n_components: Optional[int] = None,
        max_components: int = 10,
    ) -> "SpectralAnalyzer":
        """
        Обучает Gaussian Mixture Model для моделирования смеси состояний.

        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями
            n_components: Число компонентов. Если None, выбирается автоматически через BIC.
            max_components: Максимальное число компонентов для автоматического выбора

        Returns:
            self
        """
        if pc1_column not in df.columns:
            raise ValueError(f"Колонка {pc1_column} не найдена в DataFrame")

        pc1_values = df[pc1_column].dropna().values.reshape(-1, 1)

        # Автоматический выбор числа компонентов через BIC
        if n_components is None:
            best_bic = np.inf
            best_n = 1

            for n in range(1, min(max_components + 1, len(pc1_values) // 2)):
                gmm = GaussianMixture(n_components=n, random_state=42)
                gmm.fit(pc1_values)
                bic = gmm.bic(pc1_values)

                if bic < best_bic:
                    best_bic = bic
                    best_n = n

            n_components = best_n

        # Обучение GMM
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.gmm.fit(pc1_values)
        
        # Сохраняем информацию о выборе
        self.gmm_n_components_auto = n_components if n_components is not None else best_n
        self.gmm_bic_scores = getattr(self, 'gmm_bic_scores', {})

        return self

    def evaluate_gmm_quality(
        self,
        df: pd.DataFrame,
        pc1_column: str = "PC1",
        max_components: int = 10,
    ) -> pd.DataFrame:
        """
        Оценивает качество аппроксимации GMM для разного числа компонентов.
        
        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями
            max_components: Максимальное число компонентов для оценки
        
        Returns:
            DataFrame с метриками качества для каждого числа компонентов
        """
        if pc1_column not in df.columns:
            raise ValueError(f"Колонка {pc1_column} не найдена в DataFrame")
        
        from scipy import stats
        
        pc1_values = df[pc1_column].dropna().values
        
        # Вычисляем KDE для сравнения
        if len(pc1_values) > 1:
            kde = stats.gaussian_kde(pc1_values)
            x_min, x_max = pc1_values.min(), pc1_values.max()
            x_range = x_max - x_min
            x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, num=1000)
            kde_density = kde(x_grid)
        else:
            return pd.DataFrame()
        
        pc1_values_reshaped = pc1_values.reshape(-1, 1)
        results = []
        
        for n in range(1, min(max_components + 1, len(pc1_values) // 2)):
            try:
                gmm = GaussianMixture(n_components=n, random_state=42)
                gmm.fit(pc1_values_reshaped)
                
                # Метрики качества
                log_likelihood = gmm.score(pc1_values_reshaped)
                bic = gmm.bic(pc1_values_reshaped)
                aic = gmm.aic(pc1_values_reshaped)
                
                # Плотность GMM на сетке
                gmm_density = np.exp(gmm.score_samples(x_grid.reshape(-1, 1)))
                
                # Масштабирование для сравнения с KDE
                if gmm_density.max() > 0 and kde_density.max() > 0:
                    scale_factor = kde_density.max() / gmm_density.max()
                    gmm_density_scaled = gmm_density * scale_factor
                else:
                    gmm_density_scaled = gmm_density
                
                # RMSE между KDE и GMM
                rmse = np.sqrt(np.mean((kde_density - gmm_density_scaled)**2))
                
                # Максимальная ошибка
                max_error = np.max(np.abs(kde_density - gmm_density_scaled))
                
                # R² (коэффициент детерминации)
                ss_res = np.sum((kde_density - gmm_density_scaled)**2)
                ss_tot = np.sum((kde_density - np.mean(kde_density))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results.append({
                    "Число компонентов": n,
                    "BIC": bic,
                    "AIC": aic,
                    "Log-Likelihood": log_likelihood,
                    "RMSE": rmse,
                    "Max Error": max_error,
                    "R²": r2,
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(results)

    def transform_to_spectrum(
        self, df: pd.DataFrame, pc1_column: str = "PC1", use_gmm_classification: bool = False
    ) -> pd.DataFrame:
        """
        Преобразует PC1 значения в спектральную шкалу 0-1 с учетом мод.

        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями
            use_gmm_classification: Если True, классифицирует образцы по принадлежности к GMM компонентам.
                                   Если False, использует фиксированные пороги на спектральной шкале.

        Returns:
            DataFrame с добавленными колонками:
            - PC1_spectrum: спектральная шкала 0-1 (с процентилями)
            - PC1_mode: классификация (normal/mild/moderate/severe)
            - PC1_mode_distance: расстояние до ближайшей моды (если есть моды)
        """
        if self.pc1_p1 is None or self.pc1_p99 is None:
            raise ValueError(
                "Спектр не обучен. Вызовите fit_spectrum() сначала."
            )

        if pc1_column not in df.columns:
            raise ValueError(f"Колонка {pc1_column} не найдена в DataFrame")

        df_result = df.copy()
        pc1_values = df_result[pc1_column].values

        # Нормализация через процентили (с буферными зонами)
        pc1_spectrum = (pc1_values - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
        pc1_spectrum = np.clip(pc1_spectrum, 0.0, 1.0)

        df_result["PC1_spectrum"] = pc1_spectrum

        # Классификация образцов
        if use_gmm_classification and self.gmm is not None:
            # Классификация на основе принадлежности к GMM компонентам
            pc1_values_reshaped = pc1_values.reshape(-1, 1)
            gmm_predictions = self.gmm.predict(pc1_values_reshaped)
            
            # Получаем параметры GMM компонентов
            gmm_means = self.gmm.means_.flatten()
            gmm_weights = self.gmm.weights_
            
            # Преобразуем центры компонентов в спектральную шкалу
            gmm_spectrum_positions = []
            for mean in gmm_means:
                pos = (mean - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
                pos = np.clip(pos, 0.0, 1.0)
                gmm_spectrum_positions.append(pos)
            
            # Классифицируем каждый компонент на основе его позиции
            component_states = []
            for spectrum_pos in gmm_spectrum_positions:
                if spectrum_pos < 0.2:
                    component_states.append("normal")
                elif spectrum_pos < 0.5:
                    component_states.append("mild")
                elif spectrum_pos < 0.8:
                    component_states.append("moderate")
                else:
                    component_states.append("severe")
            
            # Присваиваем образцам состояние их компонента
            df_result["PC1_mode"] = [component_states[pred] for pred in gmm_predictions]
            df_result["PC1_gmm_component"] = gmm_predictions  # Для справки
        else:
            # Классификация на основе фиксированных порогов спектральной шкалы
            # Это искусственное разделение на 4 категории для удобства интерпретации
            def classify_by_spectrum(spectrum_value):
                """Классифицирует образец на основе позиции на спектральной шкале."""
                if spectrum_value < 0.2:
                    return "normal"
                elif spectrum_value < 0.5:
                    return "mild"
                elif spectrum_value < 0.8:
                    return "moderate"
                else:
                    return "severe"
            
            df_result["PC1_mode"] = [classify_by_spectrum(val) for val in pc1_spectrum]
        
        # Дополнительно: находим ближайшую моду для информации
        if self.modes:
            mode_positions = np.array([m["position"] for m in self.modes])
            mode_labels = [m["label"] for m in self.modes]

            # Расстояние до каждой моды
            distances = np.abs(pc1_values[:, np.newaxis] - mode_positions)
            nearest_mode_idx = np.argmin(distances, axis=1)

            df_result["PC1_mode_distance"] = distances[
                np.arange(len(pc1_values)), nearest_mode_idx
            ]
            # Дополнительная колонка с меткой ближайшей моды (для справки)
            df_result["PC1_nearest_mode"] = [mode_labels[i] for i in nearest_mode_idx]
        else:
            df_result["PC1_mode_distance"] = np.nan
            df_result["PC1_nearest_mode"] = None

        return df_result

    def get_feature_importance(self) -> pd.Series:
        """
        Возвращает важность признаков через loadings первой главной компоненты.

        Returns:
            Series с важностью признаков, отсортированная по абсолютному значению
        """
        if self.pca is None or self.feature_columns is None:
            raise ValueError("PCA не обучен. Вызовите fit_pca() сначала.")

        loadings = pd.Series(
            self.pca.components_[0], index=self.feature_columns
        )
        return loadings.sort_values(key=abs, ascending=False)

    def get_spectrum_info(self) -> dict:
        """
        Возвращает информацию о спектре патологий.

        Returns:
            Словарь с информацией о спектре:
            - percentiles: процентили PC1
            - modes: список мод
            - gmm_components: число компонентов GMM (если обучен)
        """
        info = {
            "percentiles": self.pc1_percentiles,
            "modes": self.modes,
            "n_modes": len(self.modes) if self.modes else 0,
        }

        if self.gmm is not None:
            info["gmm_components"] = self.gmm.n_components
            info["gmm_means"] = self.gmm.means_.flatten().tolist()
            info["gmm_weights"] = self.gmm.weights_.tolist()

        return info

    def get_gmm_components_table(self) -> pd.DataFrame:
        """
        Возвращает таблицу с параметрами GMM компонентов для характеристики медицинских состояний.

        Returns:
            DataFrame с колонками:
            - Компонент: номер компонента
            - Медицинское состояние: normal/mild/moderate/severe
            - Центр (μ) на PC1: позиция центра компонента
            - Центр на шкале 0-1: позиция на спектральной шкале
            - Ширина (σ): стандартное отклонение
            - Вес (w): доля образцов
            - Доля образцов (%): процент
        """
        if self.gmm is None:
            raise ValueError("GMM не обучен. Вызовите fit_gmm() сначала.")
        
        if self.pc1_p1 is None or self.pc1_p99 is None:
            raise ValueError("Спектр не обучен. Вызовите fit_spectrum() сначала.")

        gmm_means = self.gmm.means_.flatten()
        gmm_covariances = self.gmm.covariances_.flatten()
        gmm_weights = self.gmm.weights_
        gmm_stds = np.sqrt(gmm_covariances)
        
        # Преобразуем в спектральную шкалу для интерпретации
        gmm_spectrum_positions = []
        for mean in gmm_means:
            pos = (mean - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
            pos = np.clip(pos, 0.0, 1.0)
            gmm_spectrum_positions.append(pos)
        
        # Сортируем компоненты по позиции на спектральной шкале
        sorted_indices = np.argsort(gmm_spectrum_positions)
        
        # Создаем таблицу параметров
        gmm_params_data = []
        for idx in sorted_indices:
            mean = gmm_means[idx]
            std = gmm_stds[idx]
            weight = gmm_weights[idx]
            spectrum_pos = gmm_spectrum_positions[idx]
            
            # Классификация компонента на основе спектральной позиции
            if spectrum_pos < 0.2:
                state = "normal"
            elif spectrum_pos < 0.5:
                state = "mild"
            elif spectrum_pos < 0.8:
                state = "moderate"
            else:
                state = "severe"
            
            gmm_params_data.append({
                "Компонент": f"GMM {idx+1}",
                "Медицинское состояние": state,
                "Центр (μ) на PC1": mean,
                "Центр на шкале 0-1": spectrum_pos,
                "Ширина (σ)": std,
                "Вес (w)": weight,
                "Доля образцов (%)": weight * 100
            })
        
        return pd.DataFrame(gmm_params_data)

    def get_gmm_components_table_normalized(self) -> pd.DataFrame:
        """
        Возвращает таблицу с параметрами GMM компонентов на нормализованной шкале 0-1.

        Returns:
            DataFrame с колонками:
            - Компонент: номер компонента
            - Медицинское состояние: normal/mild/moderate/severe
            - Центр (μ) на шкале 0-1: позиция центра компонента на нормализованной шкале
            - Ширина (σ) на шкале 0-1: стандартное отклонение на нормализованной шкале
            - Вес (w): доля образцов
            - Доля образцов (%): процент
        """
        if self.gmm is None:
            raise ValueError("GMM не обучен. Вызовите fit_gmm() сначала.")
        
        if self.pc1_p1 is None or self.pc1_p99 is None:
            raise ValueError("Спектр не обучен. Вызовите fit_spectrum() сначала.")

        gmm_means = self.gmm.means_.flatten()
        gmm_covariances = self.gmm.covariances_.flatten()
        gmm_weights = self.gmm.weights_
        gmm_stds = np.sqrt(gmm_covariances)
        
        scale_factor = self.pc1_p99 - self.pc1_p1
        
        # Преобразуем параметры на нормализованную шкалу
        gmm_means_norm = []
        gmm_stds_norm = []
        gmm_spectrum_positions = []
        
        for mean, std in zip(gmm_means, gmm_stds):
            mean_norm = (mean - self.pc1_p1) / scale_factor
            std_norm = std / scale_factor
            mean_norm = np.clip(mean_norm, 0.0, 1.0)
            gmm_means_norm.append(mean_norm)
            gmm_stds_norm.append(std_norm)
            gmm_spectrum_positions.append(mean_norm)
        
        # Сортируем компоненты по позиции на спектральной шкале
        sorted_indices = np.argsort(gmm_spectrum_positions)
        
        # Создаем таблицу параметров
        gmm_params_data = []
        for idx in sorted_indices:
            mean_norm = gmm_means_norm[idx]
            std_norm = gmm_stds_norm[idx]
            weight = gmm_weights[idx]
            
            # Классификация компонента на основе спектральной позиции
            if mean_norm < 0.2:
                state = "normal"
            elif mean_norm < 0.5:
                state = "mild"
            elif mean_norm < 0.8:
                state = "moderate"
            else:
                state = "severe"
            
            gmm_params_data.append({
                "Компонент": f"GMM {idx+1}",
                "Медицинское состояние": state,
                "Центр (μ) на шкале 0-1": mean_norm,
                "Ширина (σ) на шкале 0-1": std_norm,
                "Вес (w)": weight,
                "Доля образцов (%)": weight * 100
            })
        
        return pd.DataFrame(gmm_params_data)

    def visualize_gmm_components(
        self,
        df: pd.DataFrame,
        pc1_column: str = "PC1",
        save_path: Optional[Union[str, Path]] = None,
        return_figure: bool = False,
    ):
        """
        Визуализирует отдельные GMM компоненты как чистые медицинские состояния.

        ВАЖНО: Гауссианы строятся на НЕ нормализованной шкале PC1 (сырые значения).
        GMM обучается на сырых значениях PC1, и все параметры (μ, σ) также в сырой шкале.
        Верхняя ось X показывает нормализованную шкалу 0-1 только для интерпретации.

        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями
            save_path: Путь для сохранения графика. Если None, график не сохраняется.
            return_figure: Если True, возвращает matplotlib figure вместо показа/сохранения.
        
        Returns:
            Если return_figure=True, возвращает matplotlib.figure.Figure, иначе None.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib требуется для визуализации. Установите: pip install matplotlib"
            ) from e

        if self.gmm is None:
            raise ValueError("GMM не обучен. Вызовите fit_gmm() сначала.")
        
        if pc1_column not in df.columns:
            raise ValueError(f"Колонка {pc1_column} не найдена в DataFrame")

        pc1_values = df[pc1_column].dropna().values
        
        # Получаем параметры GMM
        gmm_means = self.gmm.means_.flatten()
        gmm_covariances = self.gmm.covariances_.flatten()
        gmm_weights = self.gmm.weights_
        gmm_stds = np.sqrt(gmm_covariances)
        
        # Преобразуем в спектральную шкалу для интерпретации
        gmm_spectrum_positions = []
        for mean in gmm_means:
            pos = (mean - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
            pos = np.clip(pos, 0.0, 1.0)
            gmm_spectrum_positions.append(pos)
        
        # Сортируем компоненты по позиции на спектральной шкале
        sorted_indices = np.argsort(gmm_spectrum_positions)
        
        # Создаем сетку для построения
        x_min, x_max = pc1_values.min(), pc1_values.max()
        x_range = x_max - x_min
        x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, num=1000)
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Цвета для компонентов
        component_colors = plt.cm.Set3(np.linspace(0, 1, len(gmm_means)))
        
        # Рисуем каждый компонент отдельно
        for idx in sorted_indices:
            mean = gmm_means[idx]
            std = gmm_stds[idx]
            weight = gmm_weights[idx]
            spectrum_pos = gmm_spectrum_positions[idx]
            color = component_colors[idx]
            
            # Вычисляем плотность отдельного гауссиана
            gaussian_density = weight * stats.norm.pdf(x_grid, mean, std)
            
            # Определяем состояние
            if spectrum_pos < 0.2:
                state = "normal"
            elif spectrum_pos < 0.5:
                state = "mild"
            elif spectrum_pos < 0.8:
                state = "moderate"
            else:
                state = "severe"
            
            # Рисуем кривую компонента
            ax.plot(x_grid, gaussian_density, 
                   linewidth=2.5, alpha=0.8, color=color,
                   label=f"Компонент {idx+1} ({state}): μ={mean:.2f}, σ={std:.2f}, w={weight:.3f}")
            
            # Отмечаем центр (пик)
            peak_height = gaussian_density.max()
            ax.scatter([mean], [peak_height], 
                      color=color, s=200, marker='o', 
                      edgecolors='black', linewidths=2, zorder=10)
            
            # Вертикальная линия на центре
            ax.axvline(mean, color=color, linestyle='--', 
                      linewidth=2, alpha=0.5, zorder=5)
            
            # Подпись центра
            ax.text(mean, peak_height * 1.1, 
                  f"μ={mean:.2f}\n{state}\nw={weight:.3f}",
                  ha='center', va='bottom', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=color, alpha=0.3))
        
        # Общая смесь GMM для сравнения
        gmm_density_total = np.exp(self.gmm.score_samples(x_grid.reshape(-1, 1)))
        ax.plot(x_grid, gmm_density_total, 
               'k-', linewidth=3, alpha=0.5, 
               label='Общая смесь GMM', zorder=1)
        
        ax.set_xlabel("PC1 (сырые значения, не нормализованные)", fontsize=12)
        ax.set_ylabel("Плотность", fontsize=12)
        ax.set_title("GMM компоненты - Чистые медицинские состояния", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Добавляем вторую ось X для спектральной шкалы (для интерпретации)
        # ВАЖНО: Гауссианы строятся на НЕ нормализованной шкале PC1 (нижняя ось)
        # Верхняя ось показывает нормализованную шкалу 0-1 только для интерпретации
        ax2_spectrum = ax.twiny()
        spectrum_ticks = [0.0, 0.2, 0.5, 0.8, 1.0]
        spectrum_pc1_values = [self.pc1_p1 + t * (self.pc1_p99 - self.pc1_p1) 
                              for t in spectrum_ticks]
        ax2_spectrum.set_xlim(ax.get_xlim())
        ax2_spectrum.set_xticks(spectrum_pc1_values)
        ax2_spectrum.set_xticklabels([f"{t:.1f}" for t in spectrum_ticks])
        ax2_spectrum.set_xlabel("Спектральная шкала (0-1, для интерпретации)", fontsize=11, color='gray')
        ax2_spectrum.tick_params(colors='gray')
        
        plt.tight_layout()

        if return_figure:
            return fig
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_spectrum_comparison(
        self,
        df: pd.DataFrame,
        pc1_column: str = "PC1",
        label_column: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        return_figure: bool = False,
    ):
        """
        Создает 4 графика: сравнение спектра и GMM компонентов на нормализованной и не нормализованной шкалах.
        
        Верхний ряд (не нормализованные, сырые PC1):
        1. Спектр (KDE + GMM)
        2. GMM компоненты
        
        Нижний ряд (нормализованные, спектральная шкала 0-1):
        3. Спектр (KDE + GMM)
        4. GMM компоненты
        
        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями
            label_column: Опциональная колонка с метками для группировки
            save_path: Путь для сохранения графика
            return_figure: Если True, возвращает matplotlib figure
        
        Returns:
            Если return_figure=True, возвращает matplotlib.figure.Figure, иначе None.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib требуется для визуализации. Установите: pip install matplotlib"
            ) from e

        if pc1_column not in df.columns:
            raise ValueError(f"Колонка {pc1_column} не найдена в DataFrame")

        if self.pc1_p1 is None or self.pc1_p99 is None:
            raise ValueError("Спектр не обучен. Вызовите fit_spectrum() сначала.")

        pc1_values = df[pc1_column].dropna().values
        
        # Преобразуем в спектральную шкалу
        pc1_spectrum = (pc1_values - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
        pc1_spectrum = np.clip(pc1_spectrum, 0.0, 1.0)
        
        # Параметры нормализации
        scale_factor = self.pc1_p99 - self.pc1_p1  # Для масштабирования плотности
        
        # Создаем фигуру с 4 подграфиками
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, top=0.95, bottom=0.05)
        
        # ========== ВЕРХНИЙ РЯД: НЕ НОРМАЛИЗОВАННЫЕ (СЫРЫЕ PC1) ==========
        
        # График 1: Спектр на сырой шкале
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_spectrum_raw(ax1, pc1_values, label_column, df)
        
        # График 2: GMM компоненты на сырой шкале
        ax2 = fig.add_subplot(gs[0, 1])
        if self.gmm is not None:
            self._plot_gmm_components_raw(ax2, pc1_values)
        else:
            ax2.text(0.5, 0.5, "GMM не обучен", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("GMM компоненты (сырые PC1)", fontsize=12, fontweight='bold')
        
        # ========== НИЖНИЙ РЯД: НОРМАЛИЗОВАННЫЕ (СПЕКТРАЛЬНАЯ ШКАЛА 0-1) ==========
        
        # График 3: Спектр на нормализованной шкале
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_spectrum_normalized(ax3, pc1_spectrum, label_column, df, scale_factor)
        
        # График 4: GMM компоненты на нормализованной шкале
        ax4 = fig.add_subplot(gs[1, 1])
        if self.gmm is not None:
            self._plot_gmm_components_normalized(ax4, pc1_spectrum, scale_factor)
        else:
            ax4.text(0.5, 0.5, "GMM не обучен", ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("GMM компоненты (спектральная шкала 0-1)", fontsize=12, fontweight='bold')
        
        if return_figure:
            return fig
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _plot_spectrum_raw(self, ax, pc1_values, label_column, df):
        """Строит спектр на сырой шкале PC1."""
        from scipy import stats
        
        # KDE
        if len(pc1_values) > 1:
            kde = stats.gaussian_kde(pc1_values)
            x_min, x_max = pc1_values.min(), pc1_values.max()
            x_range = x_max - x_min
            x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, num=1000)
            density = kde(x_grid)
            
            ax.plot(x_grid, density, 'b-', linewidth=2, label='KDE', alpha=0.8)
            
            # GMM
            if self.gmm is not None:
                gmm_density = np.exp(self.gmm.score_samples(x_grid.reshape(-1, 1)))
                if gmm_density.max() > 0 and density.max() > 0:
                    scale_factor = density.max() / gmm_density.max()
                    gmm_density_normalized = gmm_density * scale_factor
                else:
                    gmm_density_normalized = gmm_density
                ax.plot(x_grid, gmm_density_normalized, 'm-', linewidth=2, alpha=0.8, 
                       label=f'GMM смесь ({self.gmm.n_components} компонентов)')
        
        ax.hist(pc1_values, bins=30, alpha=0.5, density=True, label="Histogram", color='gray')
        ax.set_xlabel("PC1 (сырые значения)", fontsize=11)
        ax.set_ylabel("Плотность", fontsize=11)
        ax.set_title("Спектр распределения (сырые PC1)", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_gmm_components_raw(self, ax, pc1_values):
        """Строит GMM компоненты на сырой шкале PC1."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        gmm_means = self.gmm.means_.flatten()
        gmm_covariances = self.gmm.covariances_.flatten()
        gmm_weights = self.gmm.weights_
        gmm_stds = np.sqrt(gmm_covariances)
        
        # Преобразуем в спектральную шкалу для классификации
        gmm_spectrum_positions = []
        for mean in gmm_means:
            pos = (mean - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
            pos = np.clip(pos, 0.0, 1.0)
            gmm_spectrum_positions.append(pos)
        
        sorted_indices = np.argsort(gmm_spectrum_positions)
        
        x_min, x_max = pc1_values.min(), pc1_values.max()
        x_range = x_max - x_min
        x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, num=1000)
        
        component_colors = plt.cm.Set3(np.linspace(0, 1, len(gmm_means)))
        
        for idx in sorted_indices:
            mean = gmm_means[idx]
            std = gmm_stds[idx]
            weight = gmm_weights[idx]
            spectrum_pos = gmm_spectrum_positions[idx]
            color = component_colors[idx]
            
            gaussian_density = weight * stats.norm.pdf(x_grid, mean, std)
            
            if spectrum_pos < 0.2:
                state = "normal"
            elif spectrum_pos < 0.5:
                state = "mild"
            elif spectrum_pos < 0.8:
                state = "moderate"
            else:
                state = "severe"
            
            ax.plot(x_grid, gaussian_density, linewidth=2, alpha=0.7, color=color,
                   label=f"{state}: μ={mean:.2f}, σ={std:.2f}")
            ax.axvline(mean, color=color, linestyle='--', linewidth=1.5, alpha=0.5)
            peak_height = gaussian_density.max()
            ax.scatter([mean], [peak_height], color=color, s=100, marker='o', 
                      edgecolors='black', linewidths=1, zorder=10)
        
        gmm_density_total = np.exp(self.gmm.score_samples(x_grid.reshape(-1, 1)))
        ax.plot(x_grid, gmm_density_total, 'k-', linewidth=2, alpha=0.4, 
               label='Общая смесь', zorder=1)
        
        ax.set_xlabel("PC1 (сырые значения)", fontsize=11)
        ax.set_ylabel("Плотность", fontsize=11)
        ax.set_title("GMM компоненты (сырые PC1)", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_spectrum_normalized(self, ax, pc1_spectrum, label_column, df, scale_factor):
        """Строит спектр на нормализованной шкале 0-1."""
        
        if len(pc1_spectrum) > 1:
            kde = stats.gaussian_kde(pc1_spectrum)
            x_grid = np.linspace(0, 1, num=1000)
            density = kde(x_grid)
            # Масштабируем плотность обратно пропорционально масштабу оси X
            density_scaled = density * scale_factor
            
            ax.plot(x_grid, density_scaled, 'b-', linewidth=2, label='KDE', alpha=0.8)
            
            # GMM на нормализованной шкале
            if self.gmm is not None:
                # Преобразуем сетку обратно в сырую шкалу для GMM
                x_grid_raw = self.pc1_p1 + x_grid * (self.pc1_p99 - self.pc1_p1)
                gmm_density_raw = np.exp(self.gmm.score_samples(x_grid_raw.reshape(-1, 1)))
                # Масштабируем плотность
                gmm_density_norm = gmm_density_raw * scale_factor
                
                if gmm_density_norm.max() > 0 and density_scaled.max() > 0:
                    scale = density_scaled.max() / gmm_density_norm.max()
                    gmm_density_norm_scaled = gmm_density_norm * scale
                else:
                    gmm_density_norm_scaled = gmm_density_norm
                    
                ax.plot(x_grid, gmm_density_norm_scaled, 'm-', linewidth=2, alpha=0.8,
                       label=f'GMM смесь ({self.gmm.n_components} компонентов)')
        
        ax.hist(pc1_spectrum, bins=30, alpha=0.5, density=True, label="Histogram", color='gray')
        # Масштабируем гистограмму
        counts, bins, patches = ax.hist(pc1_spectrum, bins=30, alpha=0.3, density=False, color='lightgray')
        # Преобразуем counts в density с учетом масштабирования
        bin_width = bins[1] - bins[0]
        density_hist = counts / (len(pc1_spectrum) * bin_width) * scale_factor
        ax.bar(bins[:-1], density_hist, width=bin_width, alpha=0.5, color='gray', label="Histogram")
        
        ax.set_xlabel("Спектральная шкала (0-1)", fontsize=11)
        ax.set_ylabel("Плотность (масштабированная)", fontsize=11)
        ax.set_title("Спектр распределения (нормализованная шкала 0-1)", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    def _plot_gmm_components_normalized(self, ax, pc1_spectrum, scale_factor):
        """Строит GMM компоненты на нормализованной шкале 0-1."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        gmm_means = self.gmm.means_.flatten()
        gmm_covariances = self.gmm.covariances_.flatten()
        gmm_weights = self.gmm.weights_
        gmm_stds = np.sqrt(gmm_covariances)
        
        # Преобразуем параметры на нормализованную шкалу
        gmm_means_norm = []
        gmm_stds_norm = []
        gmm_spectrum_positions = []
        
        for mean, std in zip(gmm_means, gmm_stds):
            mean_norm = (mean - self.pc1_p1) / (self.pc1_p99 - self.pc1_p1)
            std_norm = std / (self.pc1_p99 - self.pc1_p1)
            mean_norm = np.clip(mean_norm, 0.0, 1.0)
            gmm_means_norm.append(mean_norm)
            gmm_stds_norm.append(std_norm)
            gmm_spectrum_positions.append(mean_norm)
        
        sorted_indices = np.argsort(gmm_spectrum_positions)
        
        x_grid = np.linspace(0, 1, num=1000)
        component_colors = plt.cm.Set3(np.linspace(0, 1, len(gmm_means)))
        
        for idx in sorted_indices:
            mean_norm = gmm_means_norm[idx]
            std_norm = gmm_stds_norm[idx]
            weight = gmm_weights[idx]
            color = component_colors[idx]
            
            # Вычисляем плотность на нормализованной шкале
            gaussian_density_norm = weight * stats.norm.pdf(x_grid, mean_norm, std_norm)
            # Масштабируем плотность обратно пропорционально масштабу оси X
            gaussian_density_scaled = gaussian_density_norm * scale_factor
            
            if mean_norm < 0.2:
                state = "normal"
            elif mean_norm < 0.5:
                state = "mild"
            elif mean_norm < 0.8:
                state = "moderate"
            else:
                state = "severe"
            
            ax.plot(x_grid, gaussian_density_scaled, linewidth=2, alpha=0.7, color=color,
                   label=f"{state}: μ={mean_norm:.3f}, σ={std_norm:.3f}")
            ax.axvline(mean_norm, color=color, linestyle='--', linewidth=1.5, alpha=0.5)
            peak_height = gaussian_density_scaled.max()
            ax.scatter([mean_norm], [peak_height], color=color, s=100, marker='o',
                      edgecolors='black', linewidths=1, zorder=10)
        
        # Общая смесь на нормализованной шкале
        x_grid_raw = self.pc1_p1 + x_grid * (self.pc1_p99 - self.pc1_p1)
        gmm_density_total_raw = np.exp(self.gmm.score_samples(x_grid_raw.reshape(-1, 1)))
        gmm_density_total_norm = gmm_density_total_raw * scale_factor
        ax.plot(x_grid, gmm_density_total_norm, 'k-', linewidth=2, alpha=0.4,
               label='Общая смесь', zorder=1)
        
        ax.set_xlabel("Спектральная шкала (0-1)", fontsize=11)
        ax.set_ylabel("Плотность (масштабированная)", fontsize=11)
        ax.set_title("GMM компоненты (нормализованная шкала 0-1)", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    def visualize_spectrum(
        self,
        df: pd.DataFrame,
        pc1_column: str = "PC1",
        label_column: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Визуализирует спектр патологий на шкале 0-1.

        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями
            label_column: Опциональная колонка с метками (для цветовой маркировки)
            save_path: Путь для сохранения графика. Если None, график не сохраняется.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib требуется для визуализации. Установите: pip install matplotlib"
            )

        if pc1_column not in df.columns:
            raise ValueError(f"Колонка {pc1_column} не найдена в DataFrame")

        # Преобразование в спектральную шкалу
        df_spectrum = self.transform_to_spectrum(df, pc1_column)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # График 1: Распределение PC1 с модами
        ax1 = axes[0]
        pc1_values = df[pc1_column].dropna().values

        # KDE для оценки плотности
        if len(pc1_values) > 1:
            kde = stats.gaussian_kde(pc1_values)
            x_min, x_max = pc1_values.min(), pc1_values.max()
            x_range = x_max - x_min
            x_grid = np.linspace(
                x_min - 0.1 * x_range, x_max + 0.1 * x_range, num=1000
            )
            density = kde(x_grid)

            ax1.plot(x_grid, density, "b-", linewidth=2, label="KDE")
            ax1.fill_between(x_grid, density, alpha=0.3)
            
            # Визуализация GMM компонентов (если обучен)
            if self.gmm is not None:
                # Вычисляем общую плотность GMM для каждой точки
                gmm_density = np.exp(self.gmm.score_samples(x_grid.reshape(-1, 1)))
                
                # Улучшенное масштабирование: используем максимум вместо интеграла
                # Это более стабильно для визуализации
                if gmm_density.max() > 0 and density.max() > 0:
                    # Масштабируем так, чтобы максимумы совпадали
                    scale_factor = density.max() / gmm_density.max()
                    gmm_density_normalized = gmm_density * scale_factor
                else:
                    gmm_density_normalized = gmm_density
                
                # Вычисляем качество аппроксимации
                pc1_values_reshaped = pc1_values.reshape(-1, 1)
                log_likelihood = self.gmm.score(pc1_values_reshaped)
                bic = self.gmm.bic(pc1_values_reshaped)
                aic = self.gmm.aic(pc1_values_reshaped)
                
                # Вычисляем ошибку аппроксимации (RMSE между KDE и GMM)
                kde_on_grid = kde(x_grid)
                rmse = np.sqrt(np.mean((kde_on_grid - gmm_density_normalized)**2))
                max_error = np.max(np.abs(kde_on_grid - gmm_density_normalized))
                
                # Общая смесь GMM (только общая кривая, без отдельных компонентов)
                ax1.plot(x_grid, gmm_density_normalized, "m-", linewidth=2, alpha=0.8, 
                        label=f"GMM смесь ({self.gmm.n_components} компонентов, RMSE={rmse:.4f})", zorder=4)
                
                # Добавляем информацию о качестве в заголовок или текст
                quality_text = f"GMM: LL={log_likelihood:.1f}, BIC={bic:.1f}, RMSE={rmse:.4f}"
                ax1.text(0.02, 0.98, quality_text, transform=ax1.transAxes, 
                        fontsize=9, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Отметка мод
        if self.modes:
            for mode in self.modes:
                ax1.axvline(
                    mode["position"],
                    color="r",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Mode ({mode['label']})" if mode == self.modes[0] else "",
                )

        # Отметка процентилей
        if self.pc1_p1 is not None and self.pc1_p99 is not None:
            ax1.axvline(
                self.pc1_p1, color="g", linestyle=":", linewidth=1, label="P1"
            )
            ax1.axvline(
                self.pc1_p99, color="g", linestyle=":", linewidth=1, label="P99"
            )

        ax1.hist(pc1_values, bins=30, alpha=0.5, density=True, label="Histogram")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("Density")
        ax1.set_title("Спектр патологий (PC1)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Спектральная шкала 0-1
        ax2 = axes[1]
        spectrum_values = df_spectrum["PC1_spectrum"].dropna().values

        if label_column and label_column in df.columns:
            labels = df[label_column].dropna()
            unique_labels = labels.unique()

            for label in unique_labels:
                mask = df[label_column] == label
                ax2.hist(
                    df_spectrum.loc[mask, "PC1_spectrum"],
                    bins=30,
                    alpha=0.6,
                    label=str(label),
                )
        else:
            ax2.hist(spectrum_values, bins=30, alpha=0.6)

        # Отметка позиций мод на спектральной шкале
        if self.modes:
            for mode in self.modes:
                mode_spectrum = (mode["position"] - self.pc1_p1) / (
                    self.pc1_p99 - self.pc1_p1
                )
                mode_spectrum = np.clip(mode_spectrum, 0.0, 1.0)
                ax2.axvline(
                    mode_spectrum,
                    color="r",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                )

        ax2.set_xlabel("Спектральная шкала (0-1)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Распределение на спектральной шкале")
        ax2.set_xlim(0, 1)
        if label_column and label_column in df.columns:
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Добавляем краткие комментарии ниже графиков
        gmm_method_text = ""
        if self.gmm is not None:
            gmm_method_text = (
                f"GMM обучен с {self.gmm.n_components} компонентами через EM-алгоритм. "
                f"Число компонентов выбрано автоматически по BIC критерию. "
            )
        
        comment_text = (
            "📊 KDE (синяя линия): непараметрическая оценка плотности распределения PC1. "
            "Показывает реальную форму распределения без предположений о модели. "
            "Пики = области с высокой концентрацией образцов.\n\n"
            "🔴 Mode (красные пунктирные линии): стабильные состояния (локальные максимумы плотности). "
            "Каждая мода = группа образцов с похожими характеристиками. "
            "Моды помогают выявить основные патологические состояния.\n\n"
            "🟣 GMM (фиолетовая линия): параметрическая модель смеси гауссовых распределений. "
            f"{gmm_method_text}"
            "Аппроксимирует распределение через несколько компонентов (состояний). "
            "Центры компонентов отмечены вертикальными линиями. "
            "RMSE показывает качество аппроксимации KDE. "
            "Подробнее о методах см. раздел 'Спектральный анализ'."
        )
        
        fig.text(0.5, 0.01, comment_text,
                ha='center', va='bottom', fontsize=8.5, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3),
                family='monospace')

        plt.tight_layout(rect=[0, 0.12, 1, 1])  # Оставляем место внизу для текста

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

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
                    "pc1_percentiles": self.pc1_percentiles,
                    "pc1_p1": self.pc1_p1,
                    "pc1_p99": self.pc1_p99,
                    "modes": self.modes,
                    "gmm": self.gmm,
                },
                f,
            )

    def load(self, filepath: Union[str, Path]) -> "SpectralAnalyzer":
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
        self.pc1_percentiles = data["pc1_percentiles"]
        self.pc1_p1 = data["pc1_p1"]
        self.pc1_p99 = data["pc1_p99"]
        self.modes = data["modes"]
        self.gmm = data.get("gmm")

        return self

