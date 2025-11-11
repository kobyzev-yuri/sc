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

        return self

    def transform_to_spectrum(
        self, df: pd.DataFrame, pc1_column: str = "PC1"
    ) -> pd.DataFrame:
        """
        Преобразует PC1 значения в спектральную шкалу 0-1 с учетом мод.

        Args:
            df: DataFrame с колонкой PC1
            pc1_column: Имя колонки с PC1 значениями

        Returns:
            DataFrame с добавленными колонками:
            - PC1_spectrum: спектральная шкала 0-1 (с процентилями)
            - PC1_mode: ближайшая мода
            - PC1_mode_distance: расстояние до ближайшей моды
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

        # Классификация образцов на основе спектральной шкалы
        # Это более точный способ, чем просто ближайшая мода
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
                
                # Нормализуем для визуализации (масштабируем к KDE)
                # Используем интеграл для правильного масштабирования
                kde_integral = np.trapz(density, x_grid)
                gmm_integral = np.trapz(gmm_density, x_grid)
                if gmm_integral > 0:
                    scale_factor = kde_integral / gmm_integral
                    gmm_density_normalized = gmm_density * scale_factor
                else:
                    gmm_density_normalized = gmm_density * (density.max() / gmm_density.max()) if gmm_density.max() > 0 else gmm_density
                
                # Вычисляем качество аппроксимации
                pc1_values_reshaped = pc1_values.reshape(-1, 1)
                log_likelihood = self.gmm.score(pc1_values_reshaped)
                bic = self.gmm.bic(pc1_values_reshaped)
                aic = self.gmm.aic(pc1_values_reshaped)
                
                # Вычисляем ошибку аппроксимации (RMSE между KDE и GMM)
                kde_on_grid = kde(x_grid)
                rmse = np.sqrt(np.mean((kde_on_grid - gmm_density_normalized)**2))
                max_error = np.max(np.abs(kde_on_grid - gmm_density_normalized))
                
                # Общая смесь GMM
                ax1.plot(x_grid, gmm_density_normalized, "m-", linewidth=3, alpha=0.9, 
                        label=f"GMM смесь ({self.gmm.n_components} компонентов, RMSE={rmse:.4f})", zorder=4)
                
                # Визуализация отдельных компонентов GMM
                gmm_means = self.gmm.means_.flatten()
                gmm_covariances = self.gmm.covariances_.flatten()
                gmm_weights = self.gmm.weights_
                
                # Сортируем компоненты по весу (от большего к меньшему) для лучшей визуализации
                sorted_indices = np.argsort(gmm_weights)[::-1]
                
                # Цвета для разных компонентов
                component_colors = plt.cm.Set3(np.linspace(0, 1, len(gmm_means)))
                
                for idx, i in enumerate(sorted_indices):
                    mean = gmm_means[i]
                    cov = gmm_covariances[i]
                    weight = gmm_weights[i]
                    color = component_colors[i]
                    
                    # Вычисляем плотность отдельного гауссиана
                    std = np.sqrt(cov)
                    gaussian_density = weight * stats.norm.pdf(x_grid, mean, std)
                    # Масштабируем так же, как общую смесь
                    if gmm_integral > 0:
                        gaussian_density_scaled = gaussian_density * scale_factor
                    else:
                        gaussian_density_scaled = gaussian_density * (density.max() / gmm_density.max()) if gmm_density.max() > 0 else gaussian_density
                    
                    # Рисуем отдельный компонент (только если вес значимый)
                    if weight > 0.01:  # Показываем только компоненты с весом > 1%
                        ax1.plot(x_grid, gaussian_density_scaled, "--", linewidth=1.5, alpha=0.5, 
                                color=color, label=f"Компонент {i+1} (μ={mean:.2f}, σ={std:.2f}, w={weight:.2f})")
                    
                    # Отмечаем центр компонента вертикальной линией на оси X (не на уровне плотности!)
                    # Высота = вес компонента, масштабированный к максимуму плотности
                    center_height = weight * density.max() * 0.1  # 10% от максимума для видимости
                    ax1.axvline(mean, color=color, linestyle=':', linewidth=2, alpha=0.7, 
                               label=f"μ{i+1}={mean:.2f}" if idx == 0 else "")
                    
                    # Небольшой маркер на оси X для центра
                    ax1.scatter([mean], [0], color=color, s=80, marker='|', linewidths=3, 
                              zorder=7, label="")
                
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

        plt.tight_layout()

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

