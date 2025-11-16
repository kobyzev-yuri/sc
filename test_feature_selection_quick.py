#!/usr/bin/env python3
"""
Быстрый тест методов подбора признаков для интуиции и обсуждения результатов.
"""

import sys
from pathlib import Path
import pandas as pd

from scale import aggregate
from scale.feature_selection_automated import FeatureSelector
from scale import feature_selection_export

def main():
    print("="*70)
    print("БЫСТРЫЙ ТЕСТ МЕТОДОВ ПОДБОРА ПРИЗНАКОВ")
    print("="*70)
    
    # Загрузка данных
    print("\n1. Загрузка данных...")
    predictions_dir = sys.argv[1] if len(sys.argv) > 1 else "results/predictions"
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"   ✓ Загружено образцов: {len(df)}")
    
    # Создание относительных признаков
    print("\n2. Создание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    print(f"   ✓ Создано признаков: {len(df_features.columns) - 1}")
    
    # Получение всех доступных признаков
    print("\n3. Подготовка кандидатных признаков...")
    df_all = aggregate.select_all_feature_columns(df_features)
    candidate_features = [c for c in df_all.columns if c != 'image']
    print(f"   ✓ Кандидатных признаков: {len(candidate_features)}")
    print(f"   Примеры: {candidate_features[:5]}")
    
    # Создание селектора
    print("\n4. Инициализация селектора...")
    selector = FeatureSelector(df_all)
    print(f"   ✓ Mod образцов: {len(selector.mod_samples)}")
    print(f"   ✓ Normal образцов: {len(selector.normal_samples)}")
    
    # Тестирование нескольких методов
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ МЕТОДОВ")
    print("="*70)
    
    results = []
    
    # Метод 1: Positive Loadings Filter (быстрый)
    print("\n[1/4] Positive Loadings Filter...")
    try:
        features, metrics = selector.method_3_positive_loadings_filter(
            candidate_features,
            min_loading=0.05
        )
        results.append({
            'method': 'positive_loadings',
            'n_features': len(features),
            'features': features,
            **metrics
        })
        print(f"   ✓ Отобрано признаков: {len(features)}")
        print(f"   ✓ Score: {metrics['score']:.4f}")
        print(f"   ✓ Separation: {metrics['separation']:.4f}")
        print(f"   ✓ Mod (норм.): {metrics['mean_pc1_norm_mod']:.4f}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    
    # Метод 2: Forward Selection (ограниченный)
    print("\n[2/4] Forward Selection (до 15 признаков)...")
    try:
        features, metrics = selector.method_1_forward_selection(
            candidate_features,
            max_features=15,
            min_improvement=0.005
        )
        results.append({
            'method': 'forward_selection',
            'n_features': len(features),
            'features': features,
            **metrics
        })
        print(f"   ✓ Отобрано признаков: {len(features)}")
        print(f"   ✓ Score: {metrics['score']:.4f}")
        print(f"   ✓ Separation: {metrics['separation']:.4f}")
        print(f"   ✓ Mod (норм.): {metrics['mean_pc1_norm_mod']:.4f}")
        print(f"   ✓ Объясненная дисперсия: {metrics['explained_variance']:.4f}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    
    # Метод 3: Mutual Information
    print("\n[3/4] Mutual Information...")
    try:
        features, metrics = selector.method_4_mutual_information(
            candidate_features,
            k=None  # Автоматический выбор
        )
        results.append({
            'method': 'mutual_information',
            'n_features': len(features),
            'features': features,
            **metrics
        })
        print(f"   ✓ Отобрано признаков: {len(features)}")
        print(f"   ✓ Score: {metrics['score']:.4f}")
        print(f"   ✓ Separation: {metrics['separation']:.4f}")
        print(f"   ✓ Mod (норм.): {metrics['mean_pc1_norm_mod']:.4f}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    
    # Метод 4: LASSO
    print("\n[4/4] LASSO Selection...")
    try:
        features, metrics = selector.method_5_lasso_selection(
            candidate_features,
            cv=3  # Уменьшаем для скорости
        )
        results.append({
            'method': 'lasso',
            'n_features': len(features),
            'features': features,
            **metrics
        })
        print(f"   ✓ Отобрано признаков: {len(features)}")
        print(f"   ✓ Score: {metrics['score']:.4f}")
        print(f"   ✓ Separation: {metrics['separation']:.4f}")
        print(f"   ✓ Mod (норм.): {metrics['mean_pc1_norm_mod']:.4f}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    
    # Вывод сводной таблицы
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    if results:
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\nМетрики качества:")
        print("-" * 70)
        for idx, row in results_df.iterrows():
            print(f"\n{row['method']}:")
            print(f"  Количество признаков: {row['n_features']}")
            print(f"  Score:                {row['score']:.4f}")
            print(f"  Separation:           {row['separation']:.4f}")
            print(f"  Mod (норм. PC1):      {row['mean_pc1_norm_mod']:.4f}")
            print(f"  Объясненная дисперсия: {row['explained_variance']:.4f}")
            print(f"  Mean PC1 (mod):       {row['mean_pc1_mod']:.4f}")
            print(f"  Mean PC1 (normal):    {row['mean_pc1_normal']:.4f}")
        
        print("\n" + "="*70)
        print("ЛУЧШИЙ МЕТОД (по score):")
        print("="*70)
        best = results_df.iloc[0]
        print(f"\nМетод: {best['method']}")
        print(f"Количество признаков: {best['n_features']}")
        print(f"\nОтобранные признаки:")
        for i, feat in enumerate(best['features'], 1):
            print(f"  {i:2d}. {feat}")
        
        print(f"\nМетрики:")
        print(f"  Score:                {best['score']:.4f}")
        print(f"  Separation:           {best['separation']:.4f}")
        print(f"  Mod (норм. PC1):      {best['mean_pc1_norm_mod']:.4f}")
        print(f"  Объясненная дисперсия: {best['explained_variance']:.4f}")
        
        # Анализ признаков
        print("\n" + "="*70)
        print("АНАЛИЗ ОТОБРАННЫХ ПРИЗНАКОВ")
        print("="*70)
        
        # Подсчет частоты признаков
        from collections import Counter
        all_features = []
        for r in results:
            all_features.extend(r['features'])
        
        feature_counts = Counter(all_features)
        print("\nЧастота появления признаков в разных методах:")
        for feat, count in feature_counts.most_common():
            print(f"  {count:2d}x {feat}")
        
        # Экспорт результатов
        print("\n" + "="*70)
        print("ЭКСПОРТ РЕЗУЛЬТАТОВ")
        print("="*70)
        
        results_df = pd.DataFrame(results)
        output_dir = Path("experiments/feature_selection_quick")
        
        try:
            saved_files = feature_selection_export.export_complete_results(
                results_df=results_df,
                output_dir=output_dir,
                use_relative_features=True,
                auto_export_to_dashboard=False,  # НЕ экспортируем автоматически
                df_aggregated=df,  # Агрегированные данные
                df_features=df_features,  # Относительные признаки
                df_all_features=df_all,  # Все доступные признаки
            )
            
            print("\n✓ Результаты экспортированы:")
            print(f"  - Медицинский отчет: {saved_files.get('medical_report', 'N/A')}")
            print(f"  - CSV результаты: {saved_files.get('csv', 'N/A')}")
            print(f"  - JSON конфигурация: {saved_files.get('json', 'N/A')}")
            if saved_files.get('aggregated_data'):
                print(f"  - Агрегированные данные: {saved_files.get('aggregated_data', 'N/A')}")
            if saved_files.get('relative_features'):
                print(f"  - Относительные признаки: {saved_files.get('relative_features', 'N/A')}")
            if saved_files.get('all_features'):
                print(f"  - Все доступные признаки: {saved_files.get('all_features', 'N/A')}")
            print(f"\n💡 Конфигурация dashboard НЕ была обновлена (для безопасности)")
            print(f"   Чтобы экспортировать этот эксперимент в dashboard, используйте:")
            print(f"   python3 -m scale.feature_selection_versioning_cli export {output_dir.name}")
        except Exception as e:
            print(f"⚠️ Ошибка при экспорте: {e}")
        
    else:
        print("Не удалось получить результаты ни от одного метода.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

