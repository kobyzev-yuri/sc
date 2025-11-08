"""
Модуль для сравнения разных подходов к построению шкалы оценки патологии.

Поддерживает:
- PCA Scoring (простая нормализация)
- Spectral Analysis (с модами и процентилями)
- Сравнение результатов разных методов
"""

from pathlib import Path
from typing import Optional, Union, Dict, List
import pandas as pd
import numpy as np
import json

from . import aggregate
from . import pca_scoring
from . import spectral_analysis


class ScaleComparison:
    """
    Класс для сравнения разных методов построения шкалы.
    """
    
    def __init__(self):
        self.results: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Union[pca_scoring.PCAScorer, spectral_analysis.SpectralAnalyzer]] = {}
        self.configs: Dict[str, dict] = {}
    
    def test_pca_scoring(
        self,
        df_features: pd.DataFrame,
        name: str = "pca_simple",
        **kwargs
    ) -> pd.DataFrame:
        """
        Тестирует простой PCA scoring метод.
        
        Args:
            df_features: DataFrame с признаками
            name: Имя эксперимента
            **kwargs: Дополнительные параметры (игнорируются для PCA)
        
        Returns:
            DataFrame с колонками: image, PC1, PC1_norm
        """
        print(f"\n{'='*60}")
        print(f"Тестирование: {name} (PCA Scoring)")
        print(f"{'='*60}")
        
        scorer = pca_scoring.PCAScorer()
        df_result = scorer.fit_transform(df_features)
        
        self.models[name] = scorer
        self.results[name] = df_result[["image", "PC1", "PC1_norm"]].copy()
        self.configs[name] = {
            "method": "pca_scoring",
            "description": "Простая PCA нормализация (min-max на PC1)"
        }
        
        print(f"PC1 диапазон: [{df_result['PC1'].min():.2f}, {df_result['PC1'].max():.2f}]")
        print(f"PC1_norm диапазон: [{df_result['PC1_norm'].min():.2f}, {df_result['PC1_norm'].max():.2f}]")
        
        return df_result
    
    def test_spectral_analysis(
        self,
        df_features: pd.DataFrame,
        name: str = "spectral_default",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        use_gmm: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Тестирует спектральный анализ.
        
        Args:
            df_features: DataFrame с признаками
            name: Имя эксперимента
            percentile_low: Нижний процентиль для нормализации
            percentile_high: Верхний процентиль для нормализации
            use_gmm: Использовать ли GMM для моделирования состояний
            **kwargs: Дополнительные параметры
        
        Returns:
            DataFrame с колонками: image, PC1, PC1_spectrum, PC1_mode, ...
        """
        print(f"\n{'='*60}")
        print(f"Тестирование: {name} (Spectral Analysis)")
        print(f"  Процентили: [{percentile_low}, {percentile_high}]")
        print(f"  GMM: {use_gmm}")
        print(f"{'='*60}")
        
        analyzer = spectral_analysis.SpectralAnalyzer()
        
        # PCA
        analyzer.fit_pca(df_features)
        df_pca = analyzer.transform_pca(df_features)
        print(f"PC1 диапазон: [{df_pca['PC1'].min():.2f}, {df_pca['PC1'].max():.2f}]")
        
        # Спектральный анализ
        analyzer.fit_spectrum(
            df_pca,
            percentile_low=percentile_low,
            percentile_high=percentile_high
        )
        
        # Опционально: GMM
        if use_gmm:
            analyzer.fit_gmm(df_pca)
        
        # Преобразование в спектральную шкалу
        df_result = analyzer.transform_to_spectrum(df_pca)
        
        self.models[name] = analyzer
        self.results[name] = df_result[["image", "PC1", "PC1_spectrum"]].copy()
        if "PC1_mode" in df_result.columns:
            self.results[name]["PC1_mode"] = df_result["PC1_mode"]
        
        self.configs[name] = {
            "method": "spectral_analysis",
            "percentile_low": percentile_low,
            "percentile_high": percentile_high,
            "use_gmm": use_gmm,
            "n_modes": analyzer.get_spectrum_info()["n_modes"],
            "description": f"Спектральный анализ с процентилями [{percentile_low}, {percentile_high}]"
        }
        
        spectrum_info = analyzer.get_spectrum_info()
        print(f"Число мод: {spectrum_info['n_modes']}")
        print(f"PC1 медиана: {spectrum_info['percentiles']['median']:.2f}")
        print(f"PC1 std: {spectrum_info['percentiles']['std']:.2f}")
        print(f"PC1_spectrum диапазон: [{df_result['PC1_spectrum'].min():.2f}, {df_result['PC1_spectrum'].max():.2f}]")
        
        return df_result
    
    def compare_results(self) -> pd.DataFrame:
        """
        Сравнивает результаты всех методов.
        
        Returns:
            DataFrame с колонками: image, {method}_score, ...
        """
        if not self.results:
            raise ValueError("Нет результатов для сравнения. Запустите тесты сначала.")
        
        # Объединение всех результатов
        comparison = pd.DataFrame()
        comparison["image"] = list(self.results.values())[0]["image"]
        
        for name, df_result in self.results.items():
            # Выбираем колонку со шкалой (PC1_norm или PC1_spectrum)
            if "PC1_norm" in df_result.columns:
                comparison[f"{name}_score"] = df_result["PC1_norm"].values
            elif "PC1_spectrum" in df_result.columns:
                comparison[f"{name}_score"] = df_result["PC1_spectrum"].values
            else:
                raise ValueError(f"Не найдена колонка со шкалой в {name}")
            
            # Добавляем PC1 для сравнения
            if "PC1" in df_result.columns:
                comparison[f"{name}_PC1"] = df_result["PC1"].values
        
        return comparison
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Возвращает статистику по всем методам.
        
        Returns:
            DataFrame со статистикой
        """
        if not self.results:
            raise ValueError("Нет результатов для сравнения.")
        
        stats_rows = []
        
        for name, df_result in self.results.items():
            # Выбираем колонку со шкалой
            if "PC1_norm" in df_result.columns:
                score_col = "PC1_norm"
            elif "PC1_spectrum" in df_result.columns:
                score_col = "PC1_spectrum"
            else:
                continue
            
            scores = df_result[score_col].values
            pc1_values = df_result["PC1"].values
            
            stats_rows.append({
                "method": name,
                "description": self.configs[name].get("description", ""),
                "score_mean": np.mean(scores),
                "score_std": np.std(scores),
                "score_min": np.min(scores),
                "score_max": np.max(scores),
                "score_median": np.median(scores),
                "PC1_mean": np.mean(pc1_values),
                "PC1_std": np.std(pc1_values),
                "PC1_min": np.min(pc1_values),
                "PC1_max": np.max(pc1_values),
            })
            
            # Дополнительная информация для spectral analysis
            if name in self.models and isinstance(self.models[name], spectral_analysis.SpectralAnalyzer):
                spectrum_info = self.models[name].get_spectrum_info()
                stats_rows[-1]["n_modes"] = spectrum_info.get("n_modes", 0)
                stats_rows[-1]["percentile_low"] = self.configs[name].get("percentile_low", None)
                stats_rows[-1]["percentile_high"] = self.configs[name].get("percentile_high", None)
        
        return pd.DataFrame(stats_rows)
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Сохраняет все результаты и модели.
        
        Args:
            output_dir: Директория для сохранения
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранение сравнения
        comparison = self.compare_results()
        comparison_path = output_dir / "comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"\nСравнение сохранено: {comparison_path}")
        
        # Сохранение статистики
        stats = self.get_statistics()
        stats_path = output_dir / "statistics.csv"
        stats.to_csv(stats_path, index=False)
        print(f"Статистика сохранена: {stats_path}")
        
        # Сохранение конфигураций
        config_path = output_dir / "configs.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.configs, f, ensure_ascii=False, indent=2)
        print(f"Конфигурации сохранены: {config_path}")
        
        # Сохранение отдельных результатов
        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        for name, df_result in self.results.items():
            result_path = results_dir / f"{name}.csv"
            df_result.to_csv(result_path, index=False)
        
        # Сохранение моделей
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_dir / f"{name}.pkl"
            model.save(model_path)
        
        print(f"Все результаты сохранены в: {output_dir}")
    
    def visualize_comparison(
        self,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Визуализирует сравнение методов.
        
        Args:
            save_path: Путь для сохранения графика
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib и seaborn требуются для визуализации")
        
        if not self.results:
            raise ValueError("Нет результатов для визуализации.")
        
        comparison = self.compare_results()
        
        # Подготовка данных для визуализации
        score_cols = [col for col in comparison.columns if col.endswith("_score")]
        n_methods = len(score_cols)
        
        if n_methods == 0:
            raise ValueError("Не найдены колонки со шкалами для сравнения.")
        
        # Создание фигуры
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # График 1: Распределение шкал
        ax1 = axes[0, 0]
        # Используем цветовую палитру для согласованности
        # Если методов <= 10, используем tab10, иначе используем более широкую палитру
        if n_methods <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_methods))
        
        for idx, col in enumerate(score_cols):
            method_name = col.replace("_score", "")
            ax1.hist(
                comparison[col].dropna(),
                bins=20,
                alpha=0.6,
                label=method_name,
                density=True,
                color=colors[idx]
            )
        ax1.set_xlabel("Score (0-1)")
        ax1.set_ylabel("Density")
        ax1.set_title("Распределение шкал по методам")
        # Убеждаемся, что легенда показывает все методы
        ax1.legend(loc='best', ncol=1 if n_methods <= 5 else 2)
        ax1.grid(True, alpha=0.3)
        
        # График 2: Boxplot сравнение
        ax2 = axes[0, 1]
        data_for_box = []
        labels_for_box = []
        for col in score_cols:
            data_for_box.append(comparison[col].dropna().values)
            labels_for_box.append(col.replace("_score", ""))
        ax2.boxplot(data_for_box, labels=labels_for_box)
        ax2.set_ylabel("Score (0-1)")
        ax2.set_title("Boxplot сравнение методов")
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # График 3: Scatter plot сравнения (если есть 2 метода)
        if n_methods >= 2:
            ax3 = axes[1, 0]
            col1, col2 = score_cols[0], score_cols[1]
            ax3.scatter(comparison[col1], comparison[col2], alpha=0.6)
            ax3.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
            ax3.set_xlabel(col1.replace("_score", ""))
            ax3.set_ylabel(col2.replace("_score", ""))
            ax3.set_title("Сравнение двух методов")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # График 4: Корреляционная матрица
        ax4 = axes[1, 1]
        if n_methods >= 2:
            corr_matrix = comparison[score_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".3f",
                cmap="coolwarm",
                center=0,
                ax=ax4,
                square=True
            )
            ax4.set_title("Корреляция между методами")
        else:
            ax4.text(0.5, 0.5, "Требуется минимум 2 метода\nдля корреляционного анализа", 
                    ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("Корреляция между методами")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"График сохранен: {save_path}")
            plt.close()
        else:
            plt.show()


