#!/usr/bin/env python3
"""
Тест подбора признаков БЕЗ Paneth признаков для улучшения позиционирования mod образцов.
"""

import sys
from pathlib import Path
import pandas as pd

from scale import aggregate
from scale.feature_selection_automated import FeatureSelector
from scale import feature_selection_export

def main():
    print("="*70)
    print("ТЕСТ ПОДБОРА ПРИЗНАКОВ БЕЗ PANETH")
    print("="*70)
    
    # Загрузка данных
    print("\n1. Загрузка данных...")
    predictions_dir = sys.argv[1] if len(sys.argv) > 1 else "results/predictions"
    df = aggregate.load_predictions_batch(predictions_dir)
    print(f"   ✓ Загружено образцов: {len(df)}")
    
    # Создание относительных признаков
    print("\n2. Создание относительных признаков...")
    df_features = aggregate.create_relative_features(df)
    
    # Получение всех доступных признаков
    print("\n3. Подготовка кандидатных признаков (БЕЗ Paneth)...")
    df_all = aggregate.select_all_feature_columns(df_features)
    candidate_features_all = [c for c in df_all.columns if c != 'image']
    
    # Исключаем Paneth признаки
    candidate_features = [f for f in candidate_features_all if 'Paneth' not in f]
    
    print(f"   ✓ Всего кандидатных признаков: {len(candidate_features_all)}")
    print(f"   ✓ После исключения Paneth: {len(candidate_features)}")
    print(f"   ✓ Исключено Paneth признаков: {len(candidate_features_all) - len(candidate_features)}")
    
    # Сохраняем данные для экспорта
    df_aggregated = df.copy()
    df_features_saved = df_features.copy()
    df_all_saved = df_all.copy()
    
    # Создание селектора
    print("\n4. Инициализация селектора...")
    selector = FeatureSelector(df_all)
    print(f"   ✓ Mod образцов: {len(selector.mod_samples)}")
    print(f"   ✓ Normal образцов: {len(selector.normal_samples)}")
    
    # Тестирование методов БЕЗ Paneth
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ МЕТОДОВ БЕЗ PANETH")
    print("="*70)
    
    results = []
    
    # Метод 1: Forward Selection
    print("\n[1/3] Forward Selection (БЕЗ Paneth)...")
    try:
        features, metrics = selector.method_1_forward_selection(
            candidate_features,
            max_features=15,
            min_improvement=0.005
        )
        results.append({
            'method': 'forward_selection_no_paneth',
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
    
    # Метод 2: Mutual Information
    print("\n[2/3] Mutual Information (БЕЗ Paneth)...")
    try:
        features, metrics = selector.method_4_mutual_information(
            candidate_features,
            k=None
        )
        results.append({
            'method': 'mutual_information_no_paneth',
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
    
    # Метод 3: LASSO
    print("\n[3/3] LASSO (БЕЗ Paneth)...")
    try:
        features, metrics = selector.method_5_lasso_selection(
            candidate_features,
            cv=3
        )
        results.append({
            'method': 'lasso_no_paneth',
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
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (БЕЗ PANETH)")
    print("="*70)
    
    if results:
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
        
        print("\n" + "="*70)
        print("ЛУЧШИЙ МЕТОД (БЕЗ PANETH):")
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
        
        # Сравнение с предыдущими результатами
        print("\n" + "="*70)
        print("СРАВНЕНИЕ С РЕЗУЛЬТАТАМИ С PANETH:")
        print("="*70)
        print("\nС Paneth (forward_selection):")
        print("  Score:                3.0783")
        print("  Separation:           6.7904")
        print("  Mod (норм. PC1):      0.6800")
        print("  Объясненная дисперсия: 0.5271")
        print("\nБЕЗ Paneth (лучший метод):")
        print(f"  Score:                {best['score']:.4f}")
        print(f"  Separation:           {best['separation']:.4f}")
        print(f"  Mod (норм. PC1):      {best['mean_pc1_norm_mod']:.4f}")
        print(f"  Объясненная дисперсия: {best['explained_variance']:.4f}")
        
        improvement_mod = best['mean_pc1_norm_mod'] - 0.6800
        if improvement_mod > 0:
            print(f"\n✅ Улучшение позиционирования mod образцов: +{improvement_mod:.4f}")
        else:
            print(f"\n⚠️ Позиционирование mod образцов: {improvement_mod:.4f}")
        
        # Экспорт результатов
        print("\n" + "="*70)
        print("ЭКСПОРТ РЕЗУЛЬТАТОВ")
        print("="*70)
        
        output_dir = Path("experiments/feature_selection_no_paneth")
        
        try:
            # Нужно получить df, df_features, df_all из main функции
            # Пока передаем None, но можно улучшить структуру
            saved_files = feature_selection_export.export_complete_results(
                results_df=results_df,
                output_dir=output_dir,
                use_relative_features=True,
                auto_export_to_dashboard=True,
                df_aggregated=df_aggregated,
                df_features=df_features_saved,
                df_all_features=df_all_saved,
            )
            
            print("\n✓ Результаты экспортированы:")
            print(f"  - Dashboard конфигурация: {saved_files.get('dashboard_config', 'N/A')}")
            print(f"  - Медицинский отчет: {saved_files.get('medical_report', 'N/A')}")
            print(f"  - CSV результаты: {saved_files.get('csv', 'N/A')}")
            print(f"  - JSON конфигурация: {saved_files.get('json', 'N/A')}")
            print("\n💡 При следующем запуске dashboard отобранные признаки будут автоматически загружены!")
        except Exception as e:
            print(f"⚠️ Ошибка при экспорте: {e}")
        
    else:
        print("Не удалось получить результаты ни от одного метода.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

