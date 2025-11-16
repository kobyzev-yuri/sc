#!/usr/bin/env python3
"""
Анализ оптимальности весов в формуле score.

Тестирует различные комбинации весов для компонентов:
- separation (разделение классов)
- mean_pc1_norm_mod (позиция mod образцов)
- explained_variance (объясненная дисперсия)

Цель: найти эмпирическое обоснование для выбора весов.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from itertools import product

from model_development.feature_selection_automated import (
    FeatureSelector,
    evaluate_feature_set,
)
from scale import aggregate


def evaluate_feature_set_with_weights(
    df: pd.DataFrame,
    feature_columns: List[str],
    mod_samples: List[str],
    normal_samples: List[str],
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> Dict[str, float]:
    """
    Оценивает качество набора признаков с заданными весами.
    
    Args:
        df: DataFrame с признаками
        feature_columns: Список признаков для оценки
        mod_samples: Список имен mod образцов
        normal_samples: Список имен normal образцов
        weights: Кортеж (w_separation, w_mod, w_variance)
        
    Returns:
        Словарь с метриками качества
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    w_sep, w_mod, w_var = weights
    
    # Фильтруем данные
    mod_mask = df['image'].isin(mod_samples)
    normal_mask = df['image'].isin(normal_samples)
    
    if mod_mask.sum() == 0 or normal_mask.sum() == 0:
        return {
            'score': -np.inf,
            'mean_pc1_mod': -np.inf,
            'mean_pc1_normal': np.inf,
            'mean_pc1_norm_mod': -np.inf,
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
    
    # Комплексная оценка с заданными весами
    score = (
        w_sep * separation +
        w_mod * mean_pc1_norm_mod +
        w_var * explained_variance
    )
    
    return {
        'score': score,
        'mean_pc1_mod': mean_pc1_mod,
        'mean_pc1_normal': mean_pc1_normal,
        'mean_pc1_norm_mod': mean_pc1_norm_mod,
        'separation': separation,
        'explained_variance': explained_variance,
    }


class WeightedFeatureSelector(FeatureSelector):
    """Расширенный FeatureSelector с поддержкой кастомных весов."""
    
    def method_1_forward_selection_weighted(
        self,
        candidate_features: List[str],
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        max_features: int = None,
        min_improvement: float = 0.01,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Forward Selection с кастомными весами.
        
        Args:
            candidate_features: Список кандидатных признаков
            weights: Кортеж (w_separation, w_mod, w_variance)
            max_features: Максимальное число признаков
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
        
        while remaining and len(selected) < max_features:
            best_candidate = None
            best_candidate_score = -np.inf
            best_candidate_metrics = {}
            
            for feature in remaining:
                test_features = selected + [feature]
                metrics = evaluate_feature_set_with_weights(
                    self.df, test_features, self.mod_samples, self.normal_samples, weights
                )
                
                if metrics['score'] > best_candidate_score:
                    best_candidate_score = metrics['score']
                    best_candidate = feature
                    best_candidate_metrics = metrics
            
            if best_candidate is None:
                break
            
            # Проверяем улучшение
            improvement = best_candidate_score - best_score
            if improvement < min_improvement:
                break
            
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            best_metrics = best_candidate_metrics
        
        return selected, best_metrics


def test_weight_combinations(
    predictions_dir: str,
    weight_combinations: List[Tuple[float, float, float]],
) -> pd.DataFrame:
    """
    Тестирует различные комбинации весов.
    
    Args:
        predictions_dir: Директория с JSON файлами предсказаний
        weight_combinations: Список комбинаций весов (w_sep, w_mod, w_var)
        
    Returns:
        DataFrame с результатами для каждой комбинации весов
    """
    print("="*80)
    print("АНАЛИЗ ОПТИМАЛЬНОСТИ ВЕСОВ В ФОРМУЛЕ SCORE")
    print("="*80)
    
    # Загружаем данные
    print("\n1. Загрузка данных...")
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"   Загружено образцов: {len(df)}")
    
    # Создание относительных признаков
    print("\n2. Создание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    df_all = aggregate.select_all_feature_columns(df_features)
    candidate_features = [c for c in df_all.columns if c != 'image']
    print(f"   Кандидатных признаков: {len(candidate_features)}")
    
    # Создание селектора
    print("\n3. Инициализация селектора...")
    selector = WeightedFeatureSelector(df_all)
    
    # Тестируем каждую комбинацию весов
    print("\n4. Тестирование комбинаций весов...")
    print("="*80)
    
    results = []
    
    for i, weights in enumerate(weight_combinations, 1):
        w_sep, w_mod, w_var = weights
        print(f"\n[{i}/{len(weight_combinations)}] Веса: separation={w_sep:.2f}, mod={w_mod:.2f}, variance={w_var:.2f}")
        print("-"*80)
        
        try:
            # Запускаем Forward Selection с данными весами
            features, metrics = selector.method_1_forward_selection_weighted(
                candidate_features,
                weights=weights,
                min_improvement=0.01
            )
            
            results.append({
                'weights_separation': w_sep,
                'weights_mod': w_mod,
                'weights_variance': w_var,
                'n_features': len(features),
                'score': metrics['score'],
                'separation': metrics['separation'],
                'mean_pc1_norm_mod': metrics['mean_pc1_norm_mod'],
                'explained_variance': metrics['explained_variance'],
                'features': features,
            })
            
            print(f"✓ Признаков: {len(features)}")
            print(f"  Score: {metrics['score']:.4f}")
            print(f"  Separation: {metrics['separation']:.4f}")
            print(f"  Mod (норм.): {metrics['mean_pc1_norm_mod']:.4f}")
            print(f"  Variance: {metrics['explained_variance']:.4f}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    
    return results_df


def analyze_weight_sensitivity(results_df: pd.DataFrame) -> Dict:
    """Анализирует чувствительность к изменению весов."""
    
    print("\n" + "="*80)
    print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К ВЕСАМ")
    print("="*80)
    
    # Проверяем стабильность выбора признаков
    print("\n1. Стабильность выбора признаков:")
    print("-"*80)
    
    all_features_sets = [set(f) for f in results_df['features']]
    
    # Находим общие признаки для всех комбинаций
    if all_features_sets:
        common_features = set.intersection(*all_features_sets)
        print(f"   Общие признаки для всех комбинаций: {len(common_features)}")
        if len(common_features) > 0:
            print(f"   Первые 10: {list(common_features)[:10]}")
        
        # Находим уникальные признаки
        all_unique_features = set.union(*all_features_sets)
        print(f"   Всего уникальных признаков: {len(all_unique_features)}")
        
        # Проверяем, сколько признаков стабильно выбираются
        feature_counts = {}
        for features in results_df['features']:
            for f in features:
                feature_counts[f] = feature_counts.get(f, 0) + 1
        
        stable_features = [f for f, count in feature_counts.items() 
                          if count >= len(results_df) * 0.8]  # В 80%+ случаев
        print(f"   Стабильные признаки (в 80%+ случаев): {len(stable_features)}")
    
    # Анализ влияния весов на метрики
    print("\n2. Влияние весов на метрики:")
    print("-"*80)
    
    # Корреляция между весами и метриками
    correlations = {
        'weights_separation vs score': results_df['weights_separation'].corr(results_df['score']),
        'weights_mod vs score': results_df['weights_mod'].corr(results_df['score']),
        'weights_variance vs score': results_df['weights_variance'].corr(results_df['score']),
        'weights_separation vs separation': results_df['weights_separation'].corr(results_df['separation']),
        'weights_mod vs mod_norm': results_df['weights_mod'].corr(results_df['mean_pc1_norm_mod']),
        'weights_variance vs variance': results_df['weights_variance'].corr(results_df['explained_variance']),
    }
    
    for metric, corr in correlations.items():
        print(f"   {metric}: {corr:.4f}")
    
    # Анализ диапазона изменений
    print("\n3. Диапазон изменений метрик:")
    print("-"*80)
    print(f"   Score: [{results_df['score'].min():.4f}, {results_df['score'].max():.4f}] "
          f"(разброс: {results_df['score'].max() - results_df['score'].min():.4f})")
    print(f"   Separation: [{results_df['separation'].min():.4f}, {results_df['separation'].max():.4f}] "
          f"(разброс: {results_df['separation'].max() - results_df['separation'].min():.4f})")
    print(f"   Mod (норм.): [{results_df['mean_pc1_norm_mod'].min():.4f}, {results_df['mean_pc1_norm_mod'].max():.4f}] "
          f"(разброс: {results_df['mean_pc1_norm_mod'].max() - results_df['mean_pc1_norm_mod'].min():.4f})")
    print(f"   Variance: [{results_df['explained_variance'].min():.4f}, {results_df['explained_variance'].max():.4f}] "
          f"(разброс: {results_df['explained_variance'].max() - results_df['explained_variance'].min():.4f})")
    
    return {
        'correlations': correlations,
        'score_range': (results_df['score'].min(), results_df['score'].max()),
        'common_features': common_features if all_features_sets else set(),
        'stable_features': stable_features if all_features_sets else [],
    }


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ оптимальности весов в формуле score")
    parser.add_argument("predictions_dir", nargs="?", default="results/predictions",
                       help="Директория с JSON файлами предсказаний")
    
    args = parser.parse_args()
    predictions_dir = args.predictions_dir
    
    # Определяем комбинации весов для тестирования
    # Используем сетку поиска с шагом 0.1, сумма = 1.0
    weight_combinations = []
    
    # Базовые комбинации
    base_combinations = [
        (0.4, 0.3, 0.3),  # Текущие
        (0.5, 0.25, 0.25),  # Больше separation
        (0.33, 0.33, 0.33),  # Равные веса
        (0.3, 0.4, 0.3),  # Больше mod позиции
        (0.3, 0.3, 0.4),  # Больше variance
        (0.6, 0.2, 0.2),  # Сильный акцент на separation
        (0.2, 0.6, 0.2),  # Сильный акцент на mod
        (0.2, 0.2, 0.6),  # Сильный акцент на variance
        (0.5, 0.4, 0.1),  # Separation + mod
        (0.5, 0.1, 0.4),  # Separation + variance
        (0.1, 0.5, 0.4),  # Mod + variance
        (0.7, 0.15, 0.15),  # Очень сильный акцент на separation
        (0.15, 0.7, 0.15),  # Очень сильный акцент на mod
        (0.15, 0.15, 0.7),  # Очень сильный акцент на variance
    ]
    
    # Добавляем более детальную сетку вокруг текущих весов
    for w_sep in [0.3, 0.35, 0.4, 0.45, 0.5]:
        for w_mod in [0.2, 0.25, 0.3, 0.35, 0.4]:
            w_var = 1.0 - w_sep - w_mod
            if w_var >= 0.1 and w_var <= 0.5:
                weight_combinations.append((w_sep, w_mod, w_var))
    
    # Убираем дубликаты
    weight_combinations = list(set(weight_combinations))
    weight_combinations.sort()
    
    print(f"Всего комбинаций весов для тестирования: {len(weight_combinations)}")
    
    # Тестируем комбинации
    results_df = test_weight_combinations(predictions_dir, weight_combinations)
    
    # Анализируем результаты
    sensitivity_analysis = analyze_weight_sensitivity(results_df)
    
    # Сохраняем результаты
    output_file = Path("experiments") / "weight_analysis_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Результаты сохранены: {output_file}")
    
    # Выводим топ-10 лучших комбинаций
    print("\n" + "="*80)
    print("ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ ВЕСОВ:")
    print("="*80)
    print(f"{'№':<4} {'Sep':<6} {'Mod':<6} {'Var':<6} {'Score':<10} {'Separation':<12} {'Mod_norm':<10} {'Variance':<10} {'Features':<6}")
    print("-"*80)
    
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"{i:<4} {row['weights_separation']:<6.2f} {row['weights_mod']:<6.2f} "
              f"{row['weights_variance']:<6.2f} {row['score']:<10.4f} "
              f"{row['separation']:<12.4f} {row['mean_pc1_norm_mod']:<10.4f} "
              f"{row['explained_variance']:<10.4f} {row['n_features']:<6}")
    
    # Сравнение с текущими весами
    current_weights = (0.4, 0.3, 0.3)
    current_row = results_df[
        (results_df['weights_separation'] == current_weights[0]) &
        (results_df['weights_mod'] == current_weights[1]) &
        (results_df['weights_variance'] == current_weights[2])
    ]
    
    if len(current_row) > 0:
        current_score = current_row.iloc[0]['score']
        best_score = results_df.iloc[0]['score']
        improvement = best_score - current_score
        
        print("\n" + "="*80)
        print("СРАВНЕНИЕ С ТЕКУЩИМИ ВЕСАМИ (0.4, 0.3, 0.3):")
        print("="*80)
        print(f"Текущий score: {current_score:.4f}")
        print(f"Лучший score: {best_score:.4f}")
        print(f"Улучшение: {improvement:+.4f} ({improvement/current_score*100:+.2f}%)")
        
        if improvement > 0.01:
            best_weights = (
                results_df.iloc[0]['weights_separation'],
                results_df.iloc[0]['weights_mod'],
                results_df.iloc[0]['weights_variance']
            )
            print(f"\n✅ Рекомендуемые веса: {best_weights}")
        else:
            print("\n✅ Текущие веса близки к оптимальным")
    
    print("\n" + "="*80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("="*80)


if __name__ == "__main__":
    main()

