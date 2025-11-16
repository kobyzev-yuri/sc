"""
Автоматизированный подбор признаков для построения медицинской шкалы на базе PCA.

Цель: найти оптимальный набор признаков, при котором образцы с заболеваниями (mod)
получают высокие значения PC1 (ближе к 1 на нормализованной шкале).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Union
from itertools import combinations
import json
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    RFECV,
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import sys
from pathlib import Path

# Добавляем путь к scale для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

from scale import aggregate, pca_scoring
from model_development import feature_selection_export


def identify_sample_type(image_name: str) -> str:
    """
    Определяет тип образца по имени файла.
    
    Args:
        image_name: Имя файла/образца
        
    Returns:
        'mod' для патологических образцов, 'normal' для нормальных
    """
    image_name_lower = image_name.lower()
    if 'mod' in image_name_lower or 'ibd' in image_name_lower:
        return 'mod'
    elif 'wnl' in image_name_lower:
        return 'normal'
    else:
        return 'unknown'


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_columns: List[str],
    mod_samples: List[str],
    normal_samples: List[str],
) -> Dict[str, float]:
    """
    Оценивает качество набора признаков для разделения mod и normal образцов.
    
    Args:
        df: DataFrame с признаками
        feature_columns: Список признаков для оценки
        mod_samples: Список имен mod образцов
        normal_samples: Список имен normal образцов
        
    Returns:
        Словарь с метриками качества
    """
    # Фильтруем данные
    mod_mask = df['image'].isin(mod_samples)
    normal_mask = df['image'].isin(normal_samples)
    
    if mod_mask.sum() == 0 or normal_mask.sum() == 0:
        return {
            'score': -np.inf,
            'mean_pc1_mod': -np.inf,
            'mean_pc1_normal': np.inf,
            'separation': -np.inf,
            'explained_variance': 0.0,
        }
    
    # Обучаем PCA
    X = df[feature_columns].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)
    
    # Вычисляем метрики
    pc1_mod = X_pca[mod_mask, 0]
    pc1_normal = X_pca[normal_mask, 0]
    
    mean_pc1_mod = np.mean(pc1_mod)
    mean_pc1_normal = np.mean(pc1_normal)
    separation = mean_pc1_mod - mean_pc1_normal
    
    # Нормализуем PC1 для оценки позиции mod образцов
    pc1_min = X_pca.min()
    pc1_max = X_pca.max()
    if pc1_max > pc1_min:
        pc1_norm_mod = (pc1_mod - pc1_min) / (pc1_max - pc1_min)
        mean_pc1_norm_mod = np.mean(pc1_norm_mod)
    else:
        mean_pc1_norm_mod = 0.5
    
    explained_variance = pca.explained_variance_ratio_[0]
    
    # Комплексная оценка: максимизируем разделение и позицию mod образцов
    score = (
        0.4 * separation +  # Разделение между группами
        0.3 * mean_pc1_norm_mod +  # Позиция mod образцов (ближе к 1)
        0.3 * explained_variance  # Объясненная дисперсия
    )
    
    return {
        'score': score,
        'mean_pc1_mod': mean_pc1_mod,
        'mean_pc1_normal': mean_pc1_normal,
        'mean_pc1_norm_mod': mean_pc1_norm_mod,
        'separation': separation,
        'explained_variance': explained_variance,
    }


class FeatureSelector:
    """
    Класс для автоматизированного подбора признаков.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame с признаками и колонкой 'image'
        """
        self.df = df.copy()
        self.df['sample_type'] = self.df['image'].apply(identify_sample_type)
        
        mod_mask = self.df['sample_type'] == 'mod'
        normal_mask = self.df['sample_type'] == 'normal'
        
        self.mod_samples = self.df[mod_mask]['image'].tolist()
        self.normal_samples = self.df[normal_mask]['image'].tolist()
        
        print(f"Найдено образцов: mod={len(self.mod_samples)}, normal={len(self.normal_samples)}")
    
    def method_1_forward_selection(
        self,
        candidate_features: List[str],
        max_features: Optional[int] = None,
        min_improvement: float = 0.01,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Метод 1: Forward Selection (жадный алгоритм).
        
        Начинает с пустого набора и последовательно добавляет признаки,
        которые максимизируют целевую метрику.
        
        Args:
            candidate_features: Список кандидатных признаков
            max_features: Максимальное число признаков (None = без ограничений)
            min_improvement: Минимальное улучшение для добавления признака
            
        Returns:
            Кортеж (отобранные признаки, метрики)
        """
        selected = []
        best_score = -np.inf
        best_metrics = {}
        
        remaining = candidate_features.copy()
        
        if max_features is None:
            max_features = len(candidate_features)
        
        print(f"\n=== Forward Selection ===")
        print(f"Кандидатных признаков: {len(candidate_features)}")
        
        while len(selected) < max_features and remaining:
            best_feature = None
            best_new_score = best_score
            
            for feature in remaining:
                test_features = selected + [feature]
                metrics = evaluate_feature_set(
                    self.df, test_features, self.mod_samples, self.normal_samples
                )
                
                if metrics['score'] > best_new_score:
                    best_new_score = metrics['score']
                    best_feature = feature
                    best_metrics = metrics
            
            if best_feature is None or (best_new_score - best_score) < min_improvement:
                break
            
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_score = best_new_score
            
            print(f"Шаг {len(selected)}: добавлен '{best_feature}', score={best_score:.4f}, "
                  f"separation={best_metrics['separation']:.4f}, "
                  f"mod_norm={best_metrics['mean_pc1_norm_mod']:.4f}")
        
        return selected, best_metrics
    
    def method_2_backward_elimination(
        self,
        candidate_features: List[str],
        min_features: int = 1,
        min_improvement: float = 0.01,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Метод 2: Backward Elimination.
        
        Начинает со всех признаков и последовательно удаляет наименее важные.
        
        Args:
            candidate_features: Список кандидатных признаков
            min_features: Минимальное число признаков
            min_improvement: Минимальное улучшение для удаления признака
            
        Returns:
            Кортеж (отобранные признаки, метрики)
        """
        selected = candidate_features.copy()
        
        # Базовая оценка
        metrics = evaluate_feature_set(
            self.df, selected, self.mod_samples, self.normal_samples
        )
        best_score = metrics['score']
        best_metrics = metrics
        
        print(f"\n=== Backward Elimination ===")
        print(f"Начальное число признаков: {len(selected)}")
        print(f"Начальный score: {best_score:.4f}")
        
        while len(selected) > min_features:
            worst_feature = None
            best_new_score = best_score
            
            for feature in selected:
                test_features = [f for f in selected if f != feature]
                test_metrics = evaluate_feature_set(
                    self.df, test_features, self.mod_samples, self.normal_samples
                )
                
                if test_metrics['score'] > best_new_score:
                    best_new_score = test_metrics['score']
                    worst_feature = feature
            
            if worst_feature is None or (best_new_score - best_score) < min_improvement:
                break
            
            selected.remove(worst_feature)
            best_score = best_new_score
            best_metrics = evaluate_feature_set(
                self.df, selected, self.mod_samples, self.normal_samples
            )
            
            print(f"Шаг: удален '{worst_feature}', осталось {len(selected)}, "
                  f"score={best_score:.4f}, "
                  f"separation={best_metrics['separation']:.4f}, "
                  f"mod_norm={best_metrics['mean_pc1_norm_mod']:.4f}")
        
        return selected, best_metrics
    
    def method_3_positive_loadings_filter(
        self,
        candidate_features: List[str],
        min_loading: float = 0.05,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Метод 3: Фильтрация по положительным loadings PC1.
        
        Использует только признаки с положительными loadings в PC1,
        что обеспечивает положительную корреляцию с патологией.
        
        Args:
            candidate_features: Список кандидатных признаков
            min_loading: Минимальный loading для включения
            
        Returns:
            Кортеж (отобранные признаки, метрики)
        """
        print(f"\n=== Positive Loadings Filter ===")
        
        # Обучаем PCA на всех признаках
        X = self.df[candidate_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=1)
        pca.fit(X_scaled)
        
        # Получаем loadings
        loadings = pd.Series(pca.components_[0], index=candidate_features)
        
        # Фильтруем положительные loadings
        positive_features = [
            feat for feat, loading in loadings.items()
            if loading > min_loading
        ]
        
        print(f"Положительных loadings (> {min_loading}): {len(positive_features)} из {len(candidate_features)}")
        
        if not positive_features:
            print("Предупреждение: не найдено признаков с положительными loadings!")
            positive_features = candidate_features
        
        metrics = evaluate_feature_set(
            self.df, positive_features, self.mod_samples, self.normal_samples
        )
        
        return positive_features, metrics
    
    def method_4_mutual_information(
        self,
        candidate_features: List[str],
        k: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Метод 4: Отбор по Mutual Information.
        
        Использует взаимную информацию между признаками и метками классов.
        
        Args:
            candidate_features: Список кандидатных признаков
            k: Число признаков для отбора (None = автоматически)
            
        Returns:
            Кортеж (отобранные признаки, метрики)
        """
        print(f"\n=== Mutual Information Selection ===")
        
        # Создаем метки классов
        y = (self.df['sample_type'] == 'mod').astype(int).values
        
        # Вычисляем mutual information
        X = self.df[candidate_features].fillna(0).values
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': candidate_features,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print(f"Топ-10 признаков по MI:")
        print(mi_df.head(10).to_string(index=False))
        
        # Выбираем топ-k признаков
        if k is None:
            # Автоматически выбираем признаки с MI > медианы
            threshold = mi_df['mi_score'].median()
            selected_features = mi_df[mi_df['mi_score'] > threshold]['feature'].tolist()
        else:
            selected_features = mi_df.head(k)['feature'].tolist()
        
        print(f"Отобрано признаков: {len(selected_features)}")
        
        metrics = evaluate_feature_set(
            self.df, selected_features, self.mod_samples, self.normal_samples
        )
        
        return selected_features, metrics
    
    def method_5_lasso_selection(
        self,
        candidate_features: List[str],
        cv: int = 5,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Метод 5: L1-regularization (LASSO) для отбора признаков.
        
        Использует LASSO для автоматического отбора признаков через регуляризацию.
        
        Args:
            candidate_features: Список кандидатных признаков
            cv: Число фолдов для кросс-валидации
            
        Returns:
            Кортеж (отобранные признаки, метрики)
        """
        print(f"\n=== LASSO Selection ===")
        
        # Создаем метки классов
        y = (self.df['sample_type'] == 'mod').astype(int).values
        
        # Подготовка данных
        X = self.df[candidate_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # LASSO с кросс-валидацией
        lasso = LassoCV(cv=cv, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        # Отбираем признаки с ненулевыми коэффициентами
        selected_features = [
            candidate_features[i] for i in range(len(candidate_features))
            if abs(lasso.coef_[i]) > 1e-6
        ]
        
        print(f"LASSO выбрал {len(selected_features)} признаков из {len(candidate_features)}")
        print(f"Alpha (регуляризация): {lasso.alpha_:.6f}")
        
        if not selected_features:
            print("Предупреждение: LASSO не выбрал ни одного признака!")
            selected_features = candidate_features
        
        metrics = evaluate_feature_set(
            self.df, selected_features, self.mod_samples, self.normal_samples
        )
        
        return selected_features, metrics
    
    def method_6_rfe_selection(
        self,
        candidate_features: List[str],
        n_features: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Метод 6: Recursive Feature Elimination (RFE).
        
        Использует RFE для отбора признаков на основе важности модели.
        
        Args:
            candidate_features: Список кандидатных признаков
            n_features: Число признаков для отбора (None = автоматически через RFECV)
            
        Returns:
            Кортеж (отобранные признаки, метрики)
        """
        print(f"\n=== RFE Selection ===")
        
        # Создаем метки классов
        y = (self.df['sample_type'] == 'mod').astype(int).values
        
        # Подготовка данных
        X = self.df[candidate_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Базовый классификатор
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        if n_features is None:
            # Автоматический выбор через RFECV
            rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
            rfecv.fit(X_scaled, y)
            selected_features = [
                candidate_features[i] for i in range(len(candidate_features))
                if rfecv.support_[i]
            ]
            print(f"RFECV выбрал {len(selected_features)} признаков")
        else:
            # Фиксированное число признаков
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            rfe.fit(X_scaled, y)
            selected_features = [
                candidate_features[i] for i in range(len(candidate_features))
                if rfe.support_[i]
            ]
            print(f"RFE выбрал {len(selected_features)} признаков")
        
        metrics = evaluate_feature_set(
            self.df, selected_features, self.mod_samples, self.normal_samples
        )
        
        return selected_features, metrics
    
    def method_7_brute_force_combinations(
        self,
        candidate_features: List[str],
        max_features: int = 5,
        max_combinations: int = 1000,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Метод 7: Перебор комбинаций признаков (для малого числа признаков).
        
        Перебирает все возможные комбинации признаков и выбирает лучшую.
        
        Args:
            candidate_features: Список кандидатных признаков
            max_features: Максимальное число признаков в комбинации
            max_combinations: Максимальное число комбинаций для проверки
            
        Returns:
            Кортеж (отобранные признаки, метрики)
        """
        print(f"\n=== Brute Force Combinations ===")
        print(f"Предупреждение: может быть медленным для большого числа признаков")
        
        best_features = None
        best_score = -np.inf
        best_metrics = {}
        
        total_combinations = sum(
            len(list(combinations(candidate_features, k)))
            for k in range(1, min(max_features + 1, len(candidate_features) + 1))
        )
        
        if total_combinations > max_combinations:
            print(f"Слишком много комбинаций ({total_combinations}), ограничиваем до {max_combinations}")
            # Используем случайные комбинации
            import random
            combinations_to_test = []
            for k in range(1, min(max_features + 1, len(candidate_features) + 1)):
                for _ in range(max_combinations // max_features):
                    combo = random.sample(candidate_features, min(k, len(candidate_features)))
                    combinations_to_test.append(combo)
        else:
            # Перебираем все комбинации
            combinations_to_test = []
            for k in range(1, min(max_features + 1, len(candidate_features) + 1)):
                combinations_to_test.extend(combinations(candidate_features, k))
        
        print(f"Проверяем {len(combinations_to_test)} комбинаций...")
        
        for i, combo in enumerate(combinations_to_test):
            if i % 100 == 0:
                print(f"Проверено {i}/{len(combinations_to_test)} комбинаций...")
            
            metrics = evaluate_feature_set(
                self.df, list(combo), self.mod_samples, self.normal_samples
            )
            
            if metrics['score'] > best_score:
                best_score = metrics['score']
                best_features = list(combo)
                best_metrics = metrics
        
        print(f"Лучшая комбинация: {len(best_features)} признаков, score={best_score:.4f}")
        
        return best_features, best_metrics
    
    def compare_all_methods(
        self,
        candidate_features: List[str],
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Сравнивает все методы отбора признаков.
        
        Args:
            candidate_features: Список кандидатных признаков
            methods: Список методов для сравнения (None = все методы)
            
        Returns:
            DataFrame с результатами сравнения
        """
        if methods is None:
            methods = [
                'forward',
                'backward',
                'positive_loadings',
                'mutual_information',
                'lasso',
                'rfe',
            ]
        
        results = []
        
        for method_name in methods:
            print(f"\n{'='*60}")
            print(f"Метод: {method_name}")
            print(f"{'='*60}")
            
            try:
                if method_name == 'forward':
                    features, metrics = self.method_1_forward_selection(candidate_features)
                elif method_name == 'backward':
                    features, metrics = self.method_2_backward_elimination(candidate_features)
                elif method_name == 'positive_loadings':
                    features, metrics = self.method_3_positive_loadings_filter(candidate_features)
                elif method_name == 'mutual_information':
                    features, metrics = self.method_4_mutual_information(candidate_features)
                elif method_name == 'lasso':
                    features, metrics = self.method_5_lasso_selection(candidate_features)
                elif method_name == 'rfe':
                    features, metrics = self.method_6_rfe_selection(candidate_features)
                else:
                    continue
                
                results.append({
                    'method': method_name,
                    'n_features': len(features),
                    'features': features,
                    **metrics
                })
            except Exception as e:
                print(f"Ошибка в методе {method_name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        return results_df


def run_feature_selection_analysis(
    predictions_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Запускает полный анализ подбора признаков.
    
    Args:
        predictions_dir: Директория с JSON файлами предсказаний
        output_dir: Директория для сохранения результатов (None = не сохранять)
        methods: Список методов для сравнения
        
    Returns:
        DataFrame с результатами сравнения методов
    """
    print("="*60)
    print("АВТОМАТИЗИРОВАННЫЙ ПОДБОР ПРИЗНАКОВ ДЛЯ МЕДИЦИНСКОЙ ШКАЛЫ")
    print("="*60)
    
    # Загрузка данных
    print("\n1. Загрузка данных...")
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"   Загружено образцов: {len(df)}")
    
    # Создание относительных признаков
    print("\n2. Создание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    print(f"   Создано признаков: {len(df_features.columns) - 1}")
    
    # Получение всех доступных признаков
    df_all = aggregate.select_all_feature_columns(df_features)
    candidate_features = [c for c in df_all.columns if c != 'image']
    print(f"   Кандидатных признаков: {len(candidate_features)}")
    
    # Создание селектора
    print("\n3. Инициализация селектора признаков...")
    selector = FeatureSelector(df_all)
    
    # Сравнение методов
    print("\n4. Сравнение методов отбора признаков...")
    results_df = selector.compare_all_methods(candidate_features, methods=methods)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ")
    print("="*60)
    print(results_df[['method', 'n_features', 'score', 'separation', 
                      'mean_pc1_norm_mod', 'explained_variance']].to_string(index=False))
    
    # Сохранение результатов
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n5. Экспорт результатов...")
        saved_files = feature_selection_export.export_complete_results(
            results_df=results_df,
            output_dir=output_dir,
            use_relative_features=True,
            auto_export_to_dashboard=False,  # НЕ экспортируем автоматически для безопасности
            df_aggregated=df,  # Агрегированные данные (абсолютные признаки)
            df_features=df_features,  # Относительные признаки
            df_all_features=df_all,  # Все доступные признаки
        )
        
        print("\n" + "="*60)
        print("ЭКСПОРТ ЗАВЕРШЕН")
        print("="*60)
        print(f"✓ Медицинский отчет: {saved_files.get('medical_report', 'N/A')}")
        print(f"✓ CSV результаты: {saved_files.get('csv', 'N/A')}")
        print(f"✓ JSON конфигурация: {saved_files.get('json', 'N/A')}")
        if saved_files.get('aggregated_data'):
            print(f"✓ Агрегированные данные: {saved_files.get('aggregated_data', 'N/A')}")
        if saved_files.get('relative_features'):
            print(f"✓ Относительные признаки: {saved_files.get('relative_features', 'N/A')}")
        if saved_files.get('all_features'):
            print(f"✓ Все доступные признаки: {saved_files.get('all_features', 'N/A')}")
        print(f"\n💡 Конфигурация dashboard НЕ была обновлена (для безопасности)")
        print(f"   Чтобы экспортировать этот эксперимент в dashboard, используйте:")
        print(f"   python3 -m scale.feature_selection_versioning_cli export {output_dir.name}")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    predictions_dir = sys.argv[1] if len(sys.argv) > 1 else "results/predictions"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/feature_selection"
    
    results = run_feature_selection_analysis(predictions_dir, output_dir)

