"""
Веб-интерфейс для анализа патологий и визуализации результатов.

Модуль для создания интерактивного дашборда на Streamlit для:
- Загрузки предсказаний (JSON файлы)
- Агрегации данных и создания признаков
- Построения графиков и визуализации спектра
- Сохранения результатов экспериментов
"""

import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# Добавляем путь к проекту для импортов
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Для работы без GUI
except ImportError as e:
    raise ImportError(
        f"Требуются зависимости для дашборда. Установите: pip install streamlit matplotlib"
    ) from e

from scale import aggregate, spectral_analysis, domain, scale_comparison, pca_scoring, clustering, preprocessing, eda


def load_predictions_from_upload(uploaded_files) -> dict[str, dict]:
    """
    Загружает предсказания из загруженных файлов.

    Args:
        uploaded_files: Список загруженных файлов (Streamlit UploadedFile)

    Returns:
        Словарь {image_name: predictions_dict}
    """
    predictions = {}

    for uploaded_file in uploaded_files:
        try:
            data = json.load(uploaded_file)
            image_name = Path(uploaded_file.name).stem
            predictions[image_name] = domain.predictions_from_dict(data)
        except Exception as e:
            st.error(f"Ошибка при загрузке {uploaded_file.name}: {e}")

    return predictions


def create_experiment_dir(base_dir: Path = Path("experiments")) -> Path:
    """
    Создает директорию для нового эксперимента.

    Args:
        base_dir: Базовая директория для экспериментов

    Returns:
        Путь к директории эксперимента
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"experiment_{timestamp}"
    exp_dir.mkdir(exist_ok=True)

    return exp_dir


def save_experiment(
    exp_dir: Path,
    df: pd.DataFrame,
    analyzer: Optional[spectral_analysis.SpectralAnalyzer] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Сохраняет результаты эксперимента.

    Args:
        exp_dir: Директория эксперимента
        df: DataFrame с результатами
        analyzer: Обученный SpectralAnalyzer (опционально)
        metadata: Дополнительные метаданные (опционально)
    """
    exp_dir = Path(exp_dir)

    # Сохранение DataFrame
    csv_path = exp_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    # Сохранение модели спектрального анализа
    if analyzer is not None:
        model_path = exp_dir / "spectral_analyzer.pkl"
        analyzer.save(model_path)

    # Сохранение метаданных
    if metadata is None:
        metadata = {}

    metadata["timestamp"] = datetime.now().isoformat()
    metadata["n_samples"] = len(df)

    metadata_path = exp_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def render_dashboard():
    """
    Основная функция для рендеринга дашборда Streamlit.
    """
    st.set_page_config(
        page_title="Анализ патологий WSI",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Анализ патологий Whole Slide Images")
    st.markdown("---")

    # Боковая панель для загрузки файлов
    with st.sidebar:
        st.header("📁 Загрузка данных")

        # Опция загрузки из директории
        use_default_data = st.checkbox(
            "Использовать данные из results/predictions", value=False
        )

        if use_default_data:
            predictions_dir = Path("results/predictions")
            if predictions_dir.exists():
                json_files = list(predictions_dir.glob("*.json"))
                st.info(f"Найдено {len(json_files)} файлов в results/predictions")
            else:
                st.warning("Директория results/predictions не найдена")
                use_default_data = False

        uploaded_files = None
        if not use_default_data:
            uploaded_files = st.file_uploader(
                "Загрузите JSON файлы с предсказаниями",
                type=["json"],
                accept_multiple_files=True,
            )

        st.markdown("---")

        st.header("⚙️ Настройки")

        with st.expander("ℹ️ Относительные vs Абсолютные признаки"):
            st.markdown("""
            **Относительные признаки (нормализация по Crypts):**
            - ✅ Устраняют влияние размера биоптата
            - ✅ Позволяют сравнивать образцы разного размера
            - ✅ Фокус на плотности/интенсивности патологии
            - ✅ Хорошо для выявления паттернов независимо от размера
            - 📊 Формула: `X_count / Crypts_count`, `X_area / Crypts_area`
            
            **Абсолютные значения:**
            - ✅ Сохраняют информацию о размере биоптата
            - ✅ Важны, когда размер сам по себе значим
            - ✅ Полезны для оценки общей тяжести
            - ✅ Могут лучше работать при большом разбросе размеров
            - 📊 Формула: `X_count`, `X_area` (без нормализации)
            
            **Рекомендация:**
            - Начать с относительных признаков (по умолчанию)
            - Попробовать абсолютные, если относительные не дают хорошего разделения
            - Можно сравнить оба подхода через "Сравнение методов"
            """)

        use_relative_features = st.checkbox(
            "Использовать относительные признаки", value=True
        )

        use_spectral_analysis = st.checkbox(
            "Применить спектральный анализ", value=True
        )

        percentile_low = st.slider(
            "Нижний процентиль", 0.0, 10.0, 1.0, 0.1
        )

        percentile_high = st.slider(
            "Верхний процентиль", 90.0, 100.0, 99.0, 0.1
        )

        st.markdown("---")
        st.header("🔬 Сравнение методов")
        
        enable_comparison = st.checkbox(
            "Включить сравнение методов", value=False
        )
        
        # Инициализация переменных для сравнения
        use_pca_simple = False
        use_spectral_p1_p99 = False
        use_spectral_p05_p995 = False
        use_spectral_p5_p95 = False
        use_spectral_gmm = False
        use_custom_spectral = False
        custom_percentile_low = 2.0
        custom_percentile_high = 98.0
        
        if enable_comparison:
            st.subheader("Выберите методы для сравнения:")
            
            use_pca_simple = st.checkbox("PCA Scoring (простая нормализация)", value=True)
            
            use_spectral_p1_p99 = st.checkbox(
                "Spectral Analysis [1, 99]", value=True
            )
            
            use_spectral_p05_p995 = st.checkbox(
                "Spectral Analysis [0.5, 99.5]", value=False
            )
            
            use_spectral_p5_p95 = st.checkbox(
                "Spectral Analysis [5, 95]", value=False
            )
            
            use_spectral_gmm = st.checkbox(
                "Spectral Analysis + GMM", value=False
            )
            
            # Настройки для кастомного spectral analysis
            st.subheader("Кастомный Spectral Analysis:")
            custom_percentile_low = st.slider(
                "Нижний процентиль (кастомный)", 0.0, 10.0, 2.0, 0.1, key="custom_low"
            )
            custom_percentile_high = st.slider(
                "Верхний процентиль (кастомный)", 90.0, 100.0, 98.0, 0.1, key="custom_high"
            )
            use_custom_spectral = st.checkbox(
                f"Spectral Analysis [{custom_percentile_low}, {custom_percentile_high}]", 
                value=False
            )

        st.markdown("---")

        st.header("💾 Эксперименты")

        if st.button("Сохранить эксперимент"):
            if "df_results" in st.session_state:
                exp_dir = create_experiment_dir()
                save_experiment(
                    exp_dir,
                    st.session_state.df_results,
                    st.session_state.get("analyzer"),
                    {"settings": st.session_state.get("settings", {})},
                )
                
                # Сохранение результатов сравнения, если они есть
                if "comparison" in st.session_state:
                    try:
                        comparison = st.session_state.comparison
                        comparison.save_results(exp_dir / "comparison")
                        st.success(f"Результаты сравнения сохранены в: {exp_dir / 'comparison'}")
                    except Exception as e:
                        st.warning(f"Не удалось сохранить результаты сравнения: {e}")
                
                st.success(f"Эксперимент сохранен: {exp_dir}")
            else:
                st.warning("Нет данных для сохранения")

    # Основная область
    predictions = None

    # Загрузка данных
    if use_default_data:
        predictions_dir = Path("results/predictions")
        if predictions_dir.exists():
            json_files = list(predictions_dir.glob("*.json"))
            if json_files:
                with st.spinner("Загрузка предсказаний из results/predictions..."):
                    predictions = {}
                    for json_file in json_files:
                        try:
                            preds = domain.predictions_from_json(str(json_file))
                            image_name = json_file.stem
                            predictions[image_name] = preds
                        except Exception as e:
                            st.error(f"Ошибка при загрузке {json_file.name}: {e}")

    elif uploaded_files:
        # Загрузка предсказаний из загруженных файлов
        with st.spinner("Загрузка предсказаний..."):
            predictions = load_predictions_from_upload(uploaded_files)

    # Обработка данных
    if predictions and len(predictions) > 0:
        st.success(f"Загружено {len(predictions)} файлов")

        # Агрегация данных
        with st.spinner("Агрегация данных..."):
            rows = []

            for image_name, preds in predictions.items():
                stats = aggregate.aggregate_predictions_from_dict(
                    preds, image_name
                )
                rows.append(stats)

            df = pd.DataFrame(rows)

            if use_relative_features:
                df_features = aggregate.create_relative_features(df)
                df_features = aggregate.select_feature_columns(df_features)
            else:
                df_features = df
            
            # Исключение или выбор признаков (применяется автоматически)
            if "selection_mode" in st.session_state:
                if st.session_state.selection_mode == "Исключить признаки (blacklist)":
                    # Blacklist режим
                    if "excluded_features" in st.session_state and st.session_state.excluded_features:
                        excluded = st.session_state.excluded_features
                        available_excluded = [f for f in excluded if f in df_features.columns]
                        if available_excluded:
                            df_features = df_features.drop(columns=available_excluded)
                elif st.session_state.selection_mode == "Использовать только выбранные (whitelist)":
                    # Whitelist режим - используем только выбранные
                    if "included_features" in st.session_state and st.session_state.included_features:
                        included = st.session_state.included_features
                        available_included = [f for f in included if f in df_features.columns]
                        if available_included:
                            # Сохраняем image и добавляем выбранные признаки
                            cols_to_keep = ["image"] + available_included
                            df_features = df_features[cols_to_keep]
            elif "excluded_features" in st.session_state and st.session_state.excluded_features:
                # Обратная совместимость
                excluded = st.session_state.excluded_features
                available_excluded = [f for f in excluded if f in df_features.columns]
                if available_excluded:
                    df_features = df_features.drop(columns=available_excluded)

        st.session_state.df_results = df_features
        st.session_state.settings = {
            "use_relative_features": use_relative_features,
            "use_spectral_analysis": use_spectral_analysis,
            "percentile_low": percentile_low,
            "percentile_high": percentile_high,
        }

        # Вкладки для визуализации
        tab_names = ["📊 Данные", "🎯 Выбор признаков", "📈 Распределения", "🔬 Спектральный анализ", "🔍 Анализ образцов", "📋 Статистика", "🔗 Кластеризация"]
        if enable_comparison:
            tab_names.append("⚖️ Сравнение методов")
        
        tabs = st.tabs(tab_names)
        tab1, tab_features, tab2, tab3, tab4, tab5, tab_clustering = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5], tabs[6]
        tab_comparison = tabs[7] if enable_comparison else None

        with tab1:
            st.header("Загруженные данные")
            
            # Пояснение для relative_count
            if use_relative_features:
                with st.expander("ℹ️ Пояснение к relative_count и relative_area"):
                    st.markdown("""
                    **Relative Count (относительное количество):**
                    - Каждое значение = `X_count / Crypts_count`
                    - Это отношение количества объектов типа X к количеству крипт
                    - **Сумма по строке НЕ равна 1**, так как это независимые отношения каждого признака к Crypts
                    
                    **Пример для Count:**
                    - Если Mild_count = 10, Dysplasia_count = 5, Crypts_count = 100
                    - То Mild_relative_count = 0.1, Dysplasia_relative_count = 0.05
                    - Сумма = 0.15 (не 1!)
                    
                    **Relative Area (относительная площадь):**
                    - Аналогично: `X_area / Crypts_area`
                    - Отношение площади объектов типа X к площади крипт
                    - **Сумма по строке ТАКЖЕ НЕ равна 1**, по той же причине - это независимые отношения
                    
                    **Пример для Area:**
                    - Если Mild_area = 1000, Dysplasia_area = 500, Crypts_area = 10000
                    - То Mild_relative_area = 0.1, Dysplasia_relative_area = 0.05
                    - Сумма = 0.15 (не 1!)
                    
                    **Mean Relative Area:**
                    - Средняя относительная площадь на один объект
                    - `relative_area / count` (если count > 0)
                    - Это средний размер объекта типа X относительно размера крипты
                    """)
            
            st.dataframe(df_features, use_container_width=True)

            # Скачивание CSV
            csv = df_features.to_csv(index=False)
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        with tab_features:
            st.header("🎯 Выбор признаков для анализа")
            st.markdown("Выберите, какие признаки использовать для построения шкалы патологии.")
            
            if len(df_features) > 0:
                numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
                if "image" in numeric_cols:
                    numeric_cols.remove("image")
                
                # Режим выбора: исключить или использовать только выбранные
                selection_mode = st.radio(
                    "Режим выбора признаков",
                    ["Использовать все признаки", "Исключить признаки (blacklist)", "Использовать только выбранные (whitelist)"],
                    horizontal=True,
                    help="Все признаки: использует все доступные. Blacklist: исключает выбранные. Whitelist: использует только выбранные."
                )
                
                # Инициализация из session state
                if "selection_mode" not in st.session_state:
                    st.session_state.selection_mode = "Использовать все признаки"
                if "excluded_features" not in st.session_state:
                    st.session_state.excluded_features = []
                if "included_features" not in st.session_state:
                    st.session_state.included_features = []
                
                # Предложенные исключения из анализа образца
                suggested = []
                if "suggested_exclusions" in st.session_state:
                    suggested = st.session_state.suggested_exclusions
                
                excluded_features = None
                included_features = None
                
                if selection_mode == "Исключить признаки (blacklist)":
                    st.markdown("**Выберите признаки для исключения:**")
                    
                    # Группируем признаки по категориям для удобства
                    pathology_features = [f for f in numeric_cols if any(x in f.lower() for x in 
                        ['dysplasia', 'mild', 'moderate', 'eoe', 'granulomas'])]
                    meta_features = [f for f in numeric_cols if 'meta' in f.lower()]
                    immune_features = [f for f in numeric_cols if any(x in f.lower() for x in 
                        ['neutrophils', 'plasma', 'enterocytes'])]
                    other_features = [f for f in numeric_cols if f not in pathology_features + meta_features + immune_features]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Патологические признаки:**")
                        pathology_selected = st.multiselect(
                            "Исключить патологические",
                            pathology_features,
                            default=[],
                            key="exclude_pathology",
                            label_visibility="collapsed"
                        )
                        
                        st.markdown("**Метаплазия:**")
                        meta_selected = st.multiselect(
                            "Исключить Meta",
                            meta_features,
                            default=[f for f in suggested if f in meta_features],
                            key="exclude_meta",
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        st.markdown("**Иммунные клетки:**")
                        immune_selected = st.multiselect(
                            "Исключить иммунные",
                            immune_features,
                            default=[f for f in suggested if f in immune_features],
                            key="exclude_immune",
                            label_visibility="collapsed"
                        )
                        
                        st.markdown("**Другие признаки:**")
                        other_selected = st.multiselect(
                            "Исключить другие",
                            other_features,
                            default=[],
                            key="exclude_other",
                            label_visibility="collapsed"
                        )
                    
                    excluded_features = pathology_selected + meta_selected + immune_selected + other_selected
                    
                elif selection_mode == "Использовать только выбранные (whitelist)":
                    st.markdown("**Выберите признаки для использования:**")
                    
                    # Группируем признаки
                    pathology_features = [f for f in numeric_cols if any(x in f.lower() for x in 
                        ['dysplasia', 'mild', 'moderate', 'eoe', 'granulomas'])]
                    meta_features = [f for f in numeric_cols if 'meta' in f.lower()]
                    immune_features = [f for f in numeric_cols if any(x in f.lower() for x in 
                        ['neutrophils', 'plasma', 'enterocytes'])]
                    other_features = [f for f in numeric_cols if f not in pathology_features + meta_features + immune_features]
                    
                    # Быстрый выбор: кнопки для предустановок
                    st.markdown("**Быстрый выбор (нажмите кнопку для автоматического выбора):**")
                    preset_cols = st.columns(4)
                    
                    with preset_cols[0]:
                        if st.button("Только патология", use_container_width=True, key="preset_pathology"):
                            st.session_state.included_features = pathology_features
                            st.rerun()
                    
                    with preset_cols[1]:
                        if st.button("Патология + Иммунные", use_container_width=True, key="preset_path_immune"):
                            st.session_state.included_features = pathology_features + immune_features
                            st.rerun()
                    
                    with preset_cols[2]:
                        if st.button("Все кроме Meta", use_container_width=True, key="preset_no_meta"):
                            st.session_state.included_features = [f for f in numeric_cols if f not in meta_features]
                            st.rerun()
                    
                    with preset_cols[3]:
                        if st.button("Очистить", use_container_width=True, key="preset_clear"):
                            st.session_state.included_features = []
                            st.rerun()
                    
                    # Используем сохраненные значения или патологические по умолчанию
                    if st.session_state.included_features:
                        default_whitelist = st.session_state.included_features
                    else:
                        default_whitelist = pathology_features
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Патологические признаки:**")
                        pathology_selected = st.multiselect(
                            "Выбрать патологические",
                            pathology_features,
                            default=[f for f in default_whitelist if f in pathology_features],
                            key="include_pathology",
                            label_visibility="collapsed"
                        )
                        
                        st.markdown("**Метаплазия:**")
                        meta_selected = st.multiselect(
                            "Выбрать Meta",
                            meta_features,
                            default=[f for f in default_whitelist if f in meta_features],
                            key="include_meta",
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        st.markdown("**Иммунные клетки:**")
                        immune_selected = st.multiselect(
                            "Выбрать иммунные",
                            immune_features,
                            default=[f for f in default_whitelist if f in immune_features],
                            key="include_immune",
                            label_visibility="collapsed"
                        )
                        
                        st.markdown("**Другие признаки:**")
                        other_selected = st.multiselect(
                            "Выбрать другие",
                            other_features,
                            default=[f for f in default_whitelist if f in other_features],
                            key="include_other",
                            label_visibility="collapsed"
                        )
                    
                    included_features = pathology_selected + meta_selected + immune_selected + other_selected
                
                # Сохраняем в session state
                st.session_state.selection_mode = selection_mode
                st.session_state.excluded_features = excluded_features if excluded_features else []
                st.session_state.included_features = included_features if included_features else []
                
                # Показываем текущий статус
                st.markdown("---")
                if selection_mode == "Исключить признаки (blacklist)" and excluded_features:
                    st.success(f"✅ Исключено {len(excluded_features)} признаков: {', '.join(excluded_features[:5])}{'...' if len(excluded_features) > 5 else ''}")
                elif selection_mode == "Использовать только выбранные (whitelist)":
                    if included_features:
                        st.success(f"✅ Используется {len(included_features)} признаков: {', '.join(included_features[:5])}{'...' if len(included_features) > 5 else ''}")
                    else:
                        st.warning("⚠️ Не выбрано ни одного признака! Будут использованы все признаки.")
                else:
                    st.info(f"ℹ️ Используются все {len(numeric_cols)} признаков")
                
                # Рекомендации
                with st.expander("💡 Рекомендации по выбору признаков"):
                    st.markdown("""
                    **Когда стоит исключать признаки (blacklist):**
                    
                    1. **Аномально высокие значения** (например, Meta_relative_count > 50)
                       - Могут доминировать в PCA и "перетягивать" образец в неправильную сторону
                    
                    2. **Признаки, которые мешают классификации**
                       - Если образец явно патологический, но получает низкий score
                    
                    **Когда использовать whitelist (только выбранные признаки):**
                    
                    1. **Фокус на патологических признаках**
                       - Используйте только Dysplasia, Mild, Moderate признаки
                       - Это может помочь, если другие признаки (Meta, Neutrophils) мешают
                    
                    2. **Для образца 6mod:**
                       - Нажмите кнопку "Только патология" для быстрого выбора
                       - Или выберите вручную: Dysplasia, Mild, Moderate признаки
                    
                    **После изменения признаков** данные автоматически пересчитаются.
                    """)
            else:
                st.info("Загрузите данные, чтобы выбрать признаки")

        with tab2:
            st.header("Распределения признаков")

            if len(df_features) > 0:
                # Выбор признаков для визуализации
                numeric_cols = df_features.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                if "image" in numeric_cols:
                    numeric_cols.remove("image")

                selected_features = st.multiselect(
                    "Выберите признаки для визуализации",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                )

                if selected_features:
                    cols = st.columns(2)

                    for idx, feature in enumerate(selected_features):
                        col = cols[idx % 2]

                        with col:
                            st.subheader(feature)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.hist(
                                df_features[feature].dropna(),
                                bins=20,
                                alpha=0.7,
                                edgecolor="black",
                            )
                            ax.set_xlabel(feature)
                            ax.set_ylabel("Frequency")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)

        with tab3:
            st.header("Спектральный анализ")

            if use_spectral_analysis and len(df_features) > 0:
                # Обучение спектрального анализатора
                with st.spinner("Обучение спектрального анализатора..."):
                    analyzer = spectral_analysis.SpectralAnalyzer()

                    # PCA
                    analyzer.fit_pca(df_features)

                    # Преобразование через PCA
                    df_pca = analyzer.transform_pca(df_features)

                    # Анализ спектра
                    analyzer.fit_spectrum(
                        df_pca,
                        percentile_low=percentile_low,
                        percentile_high=percentile_high,
                    )

                    # GMM (опционально)
                    if st.checkbox("Использовать GMM для моделирования состояний"):
                        analyzer.fit_gmm(df_pca)

                    # Преобразование в спектральную шкалу
                    df_spectrum = analyzer.transform_to_spectrum(df_pca)

                st.session_state.analyzer = analyzer

                # Информация о спектре
                spectrum_info = analyzer.get_spectrum_info()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Число мод", spectrum_info["n_modes"])
                with col2:
                    st.metric(
                        "PC1 медиана",
                        f"{spectrum_info['percentiles']['median']:.2f}",
                    )
                with col3:
                    st.metric(
                        "PC1 std",
                        f"{spectrum_info['percentiles']['std']:.2f}",
                    )
                with col4:
                    if "gmm_components" in spectrum_info:
                        st.metric("GMM компонентов", spectrum_info["gmm_components"])
                    else:
                        st.metric("GMM компонентов", "Не обучен")

                # Визуализация спектра
                st.subheader("Визуализация спектра")
                
                with st.expander("ℹ️ Как интерпретировать графики спектра?"):
                    st.markdown("""
                    ## 🔗 Связь между двумя графиками
                    
                    **Это одно и то же распределение, но в разных масштабах:**
                    
                    1. **Верхний график** показывает **сырые значения PC1** (результат PCA анализа)
                    2. **Нижний график** показывает **нормализованные значения 0-1** (спектральная шкала)
                    
                    **Преобразование:**
                    ```
                    PC1_spectrum = (PC1 - P1) / (P99 - P1)
                    ```
                    - P1 (1-й процентиль) → становится 0
                    - P99 (99-й процентиль) → становится 1
                    - Все значения между P1 и P99 → масштабируются линейно в диапазон [0, 1]
                    - Значения за пределами P1-P99 → обрезаются до 0 или 1
                    
                    **Почему это нужно?**
                    - Сырые значения PC1 зависят от конкретных данных (могут быть, например, от -5 до +10)
                    - Нормализованная шкала 0-1 универсальна и интерпретируема:
                      - 0 = минимальная патология (ближе к норме)
                      - 1 = максимальная патология
                    
                    ---
                    
                    **Верхний график: Распределение PC1 (сырые значения)**
                    
                    - **Синяя линия (KDE)**: Оценка плотности распределения PC1 значений
                      - Пики = области с высокой концентрацией образцов
                      - Широкое распределение = большой разброс патологий
                      - Узкое распределение = образцы похожи друг на друга
                    
                    - **Гистограмма (серые столбцы)**: Реальное распределение ваших данных
                      - Показывает, сколько образцов попадает в каждый диапазон PC1
                      - Ось Y = **Density** (плотность, нормализованная)
                    
                    - **Красные пунктирные линии**: Моды (стабильные состояния)
                      - ⚠️ **ВАЖНО**: Моды НЕ обязательно разделяют нормальные и патологические образцы
                      - Моды = локальные максимумы плотности = центры кластеров
                      - Каждая мода = группа образцов с похожими характеристиками
                      - Мода слева → обычно ближе к норме, справа → обычно патология
                      - Но граница между норма/патология может быть между модами или в другой позиции
                    
                    - **Зеленые пунктирные линии**: Процентили (P1, P99)
                      - **P1** (слева) → будет соответствовать 0 на нижнем графике
                      - **P99** (справа) → будет соответствовать 1 на нижнем графике
                      - Образцы за пределами P1-P99 → выбросы (обрезаются до 0 или 1)
                    
                    ---
                    
                    **Нижний график: Спектральная шкала 0-1 (нормализованные значения)**
                    
                    - **Гистограмма**: Распределение образцов на шкале от 0 до 1
                      - 0 = норма (минимальная патология) = соответствует P1 на верхнем графике
                      - 1 = максимальная патология = соответствует P99 на верхнем графике
                      - ⚠️ **ВАЖНО**: Гистограмма группирует данные в 30 bins (интервалов)
                      - **Один столбик НЕ равен одному WSI** - в одном столбике может быть несколько WSI
                      - Если у вас 10 WSI, они распределены по этим 30 bins
                      - Высота столбика = **Frequency** (частота, количество образцов в bin)
                    
                    - **Красные пунктирные линии**: Позиции мод на шкале 0-1
                      - Те же моды, что на верхнем графике, но пересчитанные в шкалу 0-1
                      - Показывают, где находятся стабильные состояния на шкале
                      - Можно интерпретировать как "уровни патологии"
                    
                    ---
                    
                    **Интерпретация для ваших данных:**
                    - Если моды группируются слева (ближе к 0) → много нормальных образцов
                    - Если моды справа (ближе к 1) → много патологических образцов
                    - Равномерное распределение → плавный переход от нормы к патологии
                    - Два четких пика → бимодальное распределение (норма vs патология)
                    
                    **Чтобы увидеть каждый WSI отдельно**, посмотрите таблицу "Результаты спектрального анализа" ниже
                    
                    ---
                    
                    ## 🔬 GMM (Gaussian Mixture Model) - что добавляет?
                    
                    **GMM** - это модель смеси гауссовых распределений, которая:
                    
                    - **Автоматически определяет оптимальное число состояний** (компонентов) через BIC критерий
                      - Не нужно вручную задавать число кластеров
                      - Модель сама выбирает, сколько "состояний патологии" есть в данных
                    
                    - **Моделирует распределение PC1 как смесь нескольких гауссовых распределений**
                      - Каждый компонент = одно "состояние патологии" (например: норма, mild, moderate)
                      - Каждый компонент имеет свой центр (mean) и вес (weight)
                    
                    - **Визуализация на графике:**
                      - **Фиолетовая пунктирная линия**: Плотность распределения, предсказанная GMM
                      - **Фиолетовые крестики (X)**: Центры компонентов GMM (состояния патологии)
                      - Число рядом с крестиком = вес компонента (доля образцов в этом состоянии)
                    
                    - **Сравнение с KDE:**
                      - KDE (синяя линия) = сглаженная оценка плотности из данных
                      - GMM (фиолетовая линия) = параметрическая модель, которая пытается объяснить данные через смесь гауссовых
                      - Если GMM хорошо соответствует KDE → модель хорошо описывает данные
                      - Если GMM сильно отличается от KDE → возможно, нужно больше компонентов или данные не гауссовы
                    
                    - **Практическое применение:**
                      - Валидация мод, найденных через KDE
                      - Автоматическое определение числа патологических состояний
                      - Более точное моделирование распределения для прогнозирования
                    
                    **Когда использовать GMM:**
                    - Когда хотите автоматически определить число состояний патологии
                    - Когда нужна параметрическая модель для дальнейшего анализа
                    - Для валидации результатов KDE анализа
                    """)

                label_column = None
                if "label" in df_spectrum.columns:
                    label_column = "label"

                # Сохранение графика
                plot_path = Path("temp_spectrum_plot.png")
                analyzer.visualize_spectrum(
                    df_pca, label_column=label_column, save_path=plot_path
                )

                if plot_path.exists():
                    st.image(str(plot_path))
                    plot_path.unlink()  # Удаление временного файла

                # Таблица с результатами
                st.subheader("Результаты спектрального анализа")
                st.markdown(
                    "**Эта таблица показывает каждый WSI отдельно** - здесь вы можете увидеть точное значение "
                    "спектральной шкалы для каждого образца."
                )
                display_cols = ["image", "PC1", "PC1_spectrum"]
                if "PC1_mode" in df_spectrum.columns:
                    display_cols.append("PC1_mode")

                st.dataframe(
                    df_spectrum[display_cols].sort_values(
                        by="PC1_spectrum", ascending=False
                    ),
                    use_container_width=True,
                )
                
                # Дополнительная визуализация: гистограмма с точками для каждого WSI
                st.subheader("📊 Распределение WSI на спектральной шкале (с точками)")
                st.markdown(
                    "**Этот график показывает гистограмму (как на нижнем графике выше) с наложенными точками для каждого WSI.** "
                    "Вы можете увидеть, где именно расположен каждый из ваших образцов."
                )
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Получаем значения спектра (те же, что используются в гистограмме)
                spectrum_values = df_spectrum["PC1_spectrum"].dropna().values
                image_names = df_spectrum.loc[df_spectrum["PC1_spectrum"].notna(), "image"].values
                
                # Строим гистограмму (как в нижнем графике visualize_spectrum)
                counts, bins, patches = ax.hist(
                    spectrum_values, 
                    bins=30, 
                    alpha=0.6, 
                    color='lightblue',
                    edgecolor='black',
                    linewidth=0.5,
                    label='Гистограмма (частота)'
                )
                
                # Добавляем точки для каждого WSI поверх гистограммы
                # Размещаем точки на высоте, соответствующей частоте в этом bin + небольшой отступ
                np.random.seed(42)  # Фиксированный seed для воспроизводимости
                
                # Для каждой точки находим соответствующий bin и размещаем на его высоте
                point_heights = []
                for val in spectrum_values:
                    # Находим индекс bin для этого значения
                    bin_idx = np.digitize(val, bins) - 1
                    bin_idx = np.clip(bin_idx, 0, len(counts) - 1)
                    # Высота = частота в этом bin + небольшой случайный отступ
                    height = counts[bin_idx] + np.random.uniform(0.1, 0.3)
                    point_heights.append(height)
                
                point_heights = np.array(point_heights)
                
                # Цвета в зависимости от значения (зеленый = норма, красный = патология)
                colors = plt.cm.RdYlGn_r(spectrum_values)  # Красный-желтый-зеленый (инвертированный)
                
                # Рисуем точки поверх гистограммы
                ax.scatter(spectrum_values, point_heights, alpha=0.8, s=120, c=colors, 
                          edgecolors='black', linewidth=1.5, zorder=5, label='WSI образцы')
                
                # Подписи для каждого образца (первые 20 символов имени)
                for i, (x, y, name) in enumerate(zip(spectrum_values, point_heights, image_names)):
                    short_name = name[:20] + "..." if len(name) > 20 else name
                    ax.annotate(short_name, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
                
                # Отметка мод (как в оригинальном графике)
                if analyzer.modes:
                    for mode in analyzer.modes:
                        mode_spectrum = (mode["position"] - analyzer.pc1_p1) / (
                            analyzer.pc1_p99 - analyzer.pc1_p1
                        )
                        mode_spectrum = np.clip(mode_spectrum, 0.0, 1.0)
                        ax.axvline(
                            mode_spectrum,
                            color="r",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label="Мода" if mode == analyzer.modes[0] else ""
                        )
                
                ax.set_xlabel("Спектральная шкала (0-1)", fontsize=12)
                ax.set_ylabel("Частота (количество образцов в bin)", fontsize=12)
                ax.set_title(
                    f"Распределение WSI на спектральной шкале (всего {len(spectrum_values)} образцов)\n"
                    "Гистограмма показывает частоту, точки - расположение каждого WSI",
                    fontsize=13
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.3, axis="both")
                ax.legend(loc='upper right')
                plt.tight_layout()
                st.pyplot(fig)

                # Важность признаков
                st.subheader("📊 Важность признаков (PC1 loadings)")
                st.markdown(
                    "**Loadings PC1** показывают вклад каждого признака в первую главную компоненту. "
                    "Чем больше абсолютное значение, тем важнее признак для разделения образцов."
                )
                
                feature_importance = analyzer.get_feature_importance()

                # Таблица с важностью признаков
                top_n = st.slider("Показать топ N признаков", 5, len(feature_importance), 15)
                top_features = feature_importance.head(top_n)
                
                # Создание DataFrame для таблицы
                importance_df = pd.DataFrame({
                    "Признак": top_features.index,
                    "Loading (важность)": top_features.values,
                    "Абсолютное значение": top_features.abs().values
                }).sort_values("Абсолютное значение", ascending=False)
                
                st.dataframe(importance_df, use_container_width=True, hide_index=True)

                # График важности признаков
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Сортируем по абсолютному значению для графика
                top_features_sorted = top_features.sort_values(key=abs, ascending=True)

                colors = ['red' if x < 0 else 'blue' for x in top_features_sorted.values]
                ax.barh(
                    range(len(top_features_sorted)),
                    top_features_sorted.values,
                    align="center",
                    color=colors,
                    alpha=0.7
                )
                ax.set_yticks(range(len(top_features_sorted)))
                ax.set_yticklabels(top_features_sorted.index)
                ax.set_xlabel("Loading value")
                ax.set_title(f"Топ-{top_n} важных признаков в PC1")
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)
                
                # Интерпретация
                with st.expander("ℹ️ Как интерпретировать loadings?"):
                    st.markdown("""
                    **Положительные loadings (> 0):**
                    - Признак увеличивается вместе с PC1
                    - Высокие значения признака → высокий PC1 → высокий score патологии
                    
                    **Отрицательные loadings (< 0):**
                    - Признак уменьшается при увеличении PC1
                    - Высокие значения признака → низкий PC1 → низкий score (ближе к норме)
                    
                    **Абсолютное значение:**
                    - Показывает силу влияния признака на PC1
                    - Чем больше, тем важнее признак для разделения норма/патология
                    
                    **Примечание:**
                    - Loadings могут немного отличаться при разных наборах данных
                    - Значения из ноутбука (0.272) были на другом наборе образцов
                    - Текущие значения отражают важность признаков на ваших данных
                    """)

            else:
                st.info("Включите спектральный анализ в настройках")

        with tab4:
            st.header("🔍 Анализ конкретных образцов")
            
            if len(df_features) > 0:
                # Выбор образца для анализа
                sample_names = df_features["image"].tolist()
                selected_sample = st.selectbox(
                    "Выберите образец для анализа",
                    sample_names,
                    help="Выберите образец, который нужно проанализировать. Например, 9_ibd_mod_6mod"
                )
                
                if selected_sample:
                    sample_data = df_features[df_features["image"] == selected_sample].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"📊 Данные образца: {selected_sample}")
                        
                        # Показываем значения признаков
                        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
                        if "image" in numeric_cols:
                            numeric_cols.remove("image")
                        
                        sample_features = sample_data[numeric_cols].sort_values(ascending=False)
                        
                        st.markdown("**Топ-10 признаков с наибольшими значениями:**")
                        top_features_df = pd.DataFrame({
                            "Признак": sample_features.head(10).index,
                            "Значение": sample_features.head(10).values
                        })
                        st.dataframe(top_features_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.subheader("📈 Сравнение с другими образцами")
                        
                        # Вычисляем статистику по всем образцам
                        all_stats = df_features[numeric_cols].describe()
                        
                        # Показываем, где находится этот образец относительно других
                        comparison_data = []
                        for feat in numeric_cols:
                            sample_val = sample_data[feat]
                            mean_val = all_stats.loc['mean', feat]
                            std_val = all_stats.loc['std', feat]
                            
                            if std_val > 0:
                                z_score = (sample_val - mean_val) / std_val
                            else:
                                z_score = 0
                            
                            comparison_data.append({
                                "Признак": feat,
                                "Значение": sample_val,
                                "Среднее": mean_val,
                                "Z-score": z_score,
                                "Отклонение": "Выше нормы" if z_score > 1 else ("Ниже нормы" if z_score < -1 else "В норме")
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df = comparison_df.sort_values("Z-score", key=abs, ascending=False)
                        
                        # Показываем топ-15 с наибольшими отклонениями
                        st.dataframe(comparison_df.head(15), use_container_width=True, hide_index=True)
                        
                        # Быстрое исключение признаков
                        st.markdown("**🚫 Быстрое исключение признаков:**")
                        st.markdown("Выберите признаки с аномально высокими/низкими значениями для исключения:")
                        
                        # Находим признаки с большими отклонениями
                        high_z_features = comparison_df[comparison_df["Z-score"].abs() > 2]["Признак"].tolist()
                        
                        if high_z_features:
                            st.info(f"⚠️ Обнаружены признаки с большими отклонениями (|Z-score| > 2): {', '.join(high_z_features[:5])}")
                            
                            # Мультиселект для быстрого исключения
                            features_to_exclude = st.multiselect(
                                "Выберите признаки для исключения",
                                numeric_cols,
                                default=high_z_features[:3],  # По умолчанию предлагаем топ-3 с наибольшими отклонениями
                                key=f"exclude_{selected_sample}",
                                help="Эти признаки будут исключены из PCA анализа. Обновите страницу после выбора."
                            )
                            
                            if features_to_exclude:
                                st.warning(
                                    f"⚠️ Выбрано {len(features_to_exclude)} признаков для исключения: {', '.join(features_to_exclude)}\n\n"
                                    f"**Чтобы применить исключение:**\n"
                                    f"1. Перейдите в раздел '🎯 Выбор признаков' в боковой панели\n"
                                    f"2. Выберите эти же признаки там\n"
                                    f"3. Данные пересчитаются автоматически"
                                )
                                
                                # Сохраняем в session state для удобства
                                if "suggested_exclusions" not in st.session_state:
                                    st.session_state.suggested_exclusions = []
                                st.session_state.suggested_exclusions = features_to_exclude
                    
                    # Если есть результаты спектрального анализа
                    if "analyzer" in st.session_state and use_spectral_analysis:
                        st.subheader("🎯 Результаты спектрального анализа")
                        
                        if "df_spectrum" in locals() or "df_spectrum" in st.session_state:
                            if "df_spectrum" not in locals():
                                analyzer = st.session_state.analyzer
                                df_pca = analyzer.transform_pca(df_features)
                                df_spectrum = analyzer.transform_to_spectrum(df_pca)
                            
                            sample_spectrum = df_spectrum[df_spectrum["image"] == selected_sample]
                            
                            if len(sample_spectrum) > 0:
                                spectrum_row = sample_spectrum.iloc[0]
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    pc1_val = spectrum_row["PC1"]
                                    st.metric("PC1", f"{pc1_val:.3f}")
                                
                                with col2:
                                    spectrum_val = spectrum_row["PC1_spectrum"]
                                    st.metric("PC1_spectrum", f"{spectrum_val:.3f}")
                                
                                with col3:
                                    if "PC1_mode" in spectrum_row:
                                        mode = spectrum_row["PC1_mode"]
                                        st.metric("Ближайшая мода", mode)
                                
                                with col4:
                                    # Интерпретация
                                    if spectrum_val < 0.3:
                                        interpretation = "🔵 Низкая патология (ближе к норме)"
                                        color = "blue"
                                    elif spectrum_val < 0.7:
                                        interpretation = "🟡 Средняя патология"
                                        color = "orange"
                                    else:
                                        interpretation = "🔴 Высокая патология"
                                        color = "red"
                                    
                                    st.markdown(f"**{interpretation}**")
                                
                                # Визуализация на шкале
                                st.subheader("📍 Расположение на спектральной шкале")
                                fig, ax = plt.subplots(figsize=(12, 2))
                                
                                # Все образцы
                                all_spectrum = df_spectrum["PC1_spectrum"].values
                                ax.scatter(all_spectrum, [0.5] * len(all_spectrum), 
                                          alpha=0.3, s=50, c='gray', label='Все образцы')
                                
                                # Выбранный образец
                                ax.scatter([spectrum_val], [0.5], 
                                          s=300, c=color, marker='*', 
                                          edgecolors='black', linewidth=2, 
                                          label=f'{selected_sample} (score={spectrum_val:.3f})',
                                          zorder=10)
                                
                                # Моды
                                if analyzer.modes:
                                    for mode in analyzer.modes:
                                        mode_spectrum = (mode["position"] - analyzer.pc1_p1) / (
                                            analyzer.pc1_p99 - analyzer.pc1_p1
                                        )
                                        mode_spectrum = np.clip(mode_spectrum, 0.0, 1.0)
                                        ax.axvline(mode_spectrum, color='red', linestyle='--', 
                                                  alpha=0.5, linewidth=1)
                                
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                                ax.set_xlabel("Спектральная шкала (0-1)")
                                ax.set_ylabel("")
                                ax.set_yticks([])
                                ax.set_title(f"Расположение {selected_sample} на шкале патологии")
                                ax.legend(loc='upper right')
                                ax.grid(True, alpha=0.3, axis='x')
                                st.pyplot(fig)
                                
                                # Рекомендации
                                if spectrum_val < 0.3:
                                    st.warning(
                                        f"⚠️ Образец {selected_sample} имеет низкий score ({spectrum_val:.3f}), "
                                        f"что может не соответствовать диагнозу. "
                                        f"Рекомендуется:\n"
                                        f"1. Проверить исходные предсказания патологий\n"
                                        f"2. Исключить некоторые признаки, которые могут мешать\n"
                                        f"3. Проверить значения признаков этого образца"
                                    )
                    
                    # Сравнение с похожими образцами
                    st.subheader("🔬 Сравнение с другими образцами")
                    
                    # Находим похожие образцы по PC1_spectrum
                    if "df_spectrum" in locals() or "df_spectrum" in st.session_state:
                        if "df_spectrum" not in locals():
                            analyzer = st.session_state.analyzer
                            df_pca = analyzer.transform_pca(df_features)
                            df_spectrum = analyzer.transform_to_spectrum(df_pca)
                        
                        sample_spectrum_val = df_spectrum[df_spectrum["image"] == selected_sample]["PC1_spectrum"].iloc[0]
                        
                        # Находим ближайшие образцы
                        df_spectrum_sorted = df_spectrum.sort_values("PC1_spectrum")
                        sample_idx = df_spectrum_sorted[df_spectrum_sorted["image"] == selected_sample].index[0]
                        
                        # Берем 2 образца до и 2 после
                        start_idx = max(0, sample_idx - 2)
                        end_idx = min(len(df_spectrum_sorted), sample_idx + 3)
                        similar_samples = df_spectrum_sorted.iloc[start_idx:end_idx]
                        
                        st.dataframe(similar_samples[["image", "PC1", "PC1_spectrum"]], 
                                   use_container_width=True, hide_index=True)
                    
                    # Сравнение с образцами того же типа (ibd_mod, hp_ и т.д.)
                    st.subheader("🔍 Сравнение с образцами того же типа")
                    
                    # Пытаемся найти образцы с похожим паттерном в имени
                    sample_name_lower = selected_sample.lower()
                    similar_type_samples = []
                    
                    # Ищем образцы с похожими паттернами
                    if "ibd_mod" in sample_name_lower:
                        pattern = "ibd_mod"
                        pattern_name = "IBD moderate"
                    elif "hp_" in sample_name_lower:
                        pattern = "hp_"
                        pattern_name = "HP (Helicobacter pylori)"
                    else:
                        pattern = None
                        pattern_name = None
                    
                    if pattern:
                        similar_type = df_features[df_features["image"].str.contains(pattern, case=False, na=False)]
                        if len(similar_type) > 1:  # Больше чем сам образец
                            st.markdown(f"**Образцы типа '{pattern_name}' ({pattern}):**")
                            
                            # Сравнение признаков с группой
                            comparison_features = []
                            for feat in numeric_cols:
                                sample_val = sample_data[feat]
                                other_vals = similar_type[similar_type["image"] != selected_sample][feat].dropna()
                                
                                if len(other_vals) > 0:
                                    other_mean = other_vals.mean()
                                    other_std = other_vals.std()
                                    
                                    if other_std > 0:
                                        z_vs_group = (sample_val - other_mean) / other_std
                                    else:
                                        z_vs_group = 0
                                    
                                    # Процент от среднего в группе
                                    if other_mean != 0:
                                        pct_of_group = (sample_val / other_mean) * 100
                                    else:
                                        pct_of_group = 0
                                    
                                    comparison_features.append({
                                        "Признак": feat,
                                        f"{selected_sample}": f"{sample_val:.4f}",
                                        "Среднее в группе": f"{other_mean:.4f}",
                                        "Z-score vs группа": f"{z_vs_group:.2f}",
                                        "% от среднего": f"{pct_of_group:.1f}%",
                                        "Разница": f"{sample_val - other_mean:.4f}"
                                    })
                            
                            if comparison_features:
                                comp_df = pd.DataFrame(comparison_features)
                                # Сортируем по абсолютному Z-score
                                comp_df["Z_abs"] = comp_df["Z-score vs группа"].str.replace('%', '').astype(float).abs()
                                comp_df = comp_df.sort_values("Z_abs", ascending=False).drop(columns=["Z_abs"])
                                
                                st.markdown("**Признаки, где образец сильно отличается от группы:**")
                                st.dataframe(comp_df.head(20), use_container_width=True, hide_index=True)
                                
                                # Рекомендации по исключению
                                st.markdown("**💡 Рекомендации:**")
                                
                                # Находим признаки, где образец сильно отличается
                                comp_df_numeric = pd.DataFrame(comparison_features)
                                comp_df_numeric["Z_abs"] = pd.to_numeric(comp_df_numeric["Z-score vs группа"].str.replace('%', ''), errors='coerce').abs()
                                comp_df_numeric["Разница_num"] = pd.to_numeric(comp_df_numeric["Разница"], errors='coerce')
                                
                                low_features = comp_df_numeric[comp_df_numeric["Z_abs"] > 1.5]["Признак"].tolist()
                                high_features = comp_df_numeric[comp_df_numeric["Z_abs"] > 1.5]["Признак"].tolist()
                                
                                if low_features:
                                    st.warning(
                                        f"⚠️ У образца **сильно отличающиеся** значения по сравнению с группой:\n"
                                        f"{', '.join(low_features[:5])}\n\n"
                                        f"**Если эти признаки не связаны с тяжестью патологии, попробуйте их исключить.**"
                                    )
                                
                                # Показываем таблицу сравнения всех образцов группы
                                st.markdown("**📊 Сравнение всех образцов группы:**")
                                if "df_spectrum" in locals() or "df_spectrum" in st.session_state:
                                    if "df_spectrum" not in locals():
                                        analyzer = st.session_state.analyzer
                                        df_pca = analyzer.transform_pca(df_features)
                                        df_spectrum = analyzer.transform_to_spectrum(df_pca)
                                    
                                    similar_type_with_spectrum = similar_type.merge(
                                        df_spectrum[["image", "PC1", "PC1_spectrum"]], 
                                        on="image", how="left"
                                    )
                                    display_cols = ["image", "PC1", "PC1_spectrum"]
                                    # Добавляем ключевые патологические признаки
                                    key_features = ["Dysplasia_relative_count", "Mild_relative_count", 
                                                 "Moderate_relative_count", "Neutrophils_relative_count",
                                                 "Plasma Cells_relative_count"]
                                    for kf in key_features:
                                        if kf in similar_type_with_spectrum.columns:
                                            display_cols.append(kf)
                                    
                                    st.dataframe(
                                        similar_type_with_spectrum[display_cols].sort_values("PC1_spectrum", ascending=False),
                                        use_container_width=True, 
                                        hide_index=True
                                    )
                                    
                                    # Анализ: почему 6mod имеет низкий score
                                    if selected_sample == "9_ibd_mod_6mod" or "6mod" in selected_sample:
                                        st.error(
                                            "**🔴 Проблема с образцом 6mod:**\n\n"
                                            "Образец имеет низкий score, хотя должен быть патологическим. "
                                            "Возможные причины:\n\n"
                                            "1. **Низкие значения патологических признаков** (Dysplasia, Mild, Moderate) "
                                            "по сравнению с другими ibd_mod образцами\n"
                                            "2. **Высокие значения других признаков** (Neutrophils, Plasma Cells), "
                                            "которые могут 'перетягивать' в другую сторону\n"
                                            "3. **Проблема в исходных предсказаниях** - возможно, модель не детектировала "
                                            "патологии для этого образца\n\n"
                                            "**Рекомендации:**\n"
                                            "- Проверьте исходный JSON файл с предсказаниями\n"
                                            "- Попробуйте исключить признаки с высокими значениями (Neutrophils, Plasma Cells)\n"
                                            "- Или используйте только патологические признаки (Dysplasia, Mild, Moderate)"
                                        )
            else:
                st.info("Загрузите данные для анализа образцов")

        with tab5:
            st.header("Статистика")

            if len(df_features) > 0:
                numeric_cols = df_features.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

                if numeric_cols:
                    st.subheader("Описательная статистика")
                    st.dataframe(
                        df_features[numeric_cols].describe(),
                        use_container_width=True,
                    )

                    # Корреляционная матрица
                    if len(numeric_cols) > 1:
                        st.subheader("Корреляционная матрица")
                        
                        # Используем функцию из preprocessing
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                            tmp_path = Path(tmp_file.name)
                        
                        try:
                            preprocessing.visualize_correlations(
                                df_features,
                                feature_columns=numeric_cols,
                                save_path=tmp_path
                            )
                            if tmp_path.exists():
                                st.image(str(tmp_path))
                        finally:
                            if tmp_path.exists():
                                tmp_path.unlink()
                        
                        # Анализ высоко коррелированных признаков
                        with st.expander("🔍 Анализ высоко коррелированных признаков"):
                            threshold = st.slider("Порог корреляции", 0.7, 0.99, 0.95, 0.01)
                            highly_corr = preprocessing.find_highly_correlated_features(
                                df_features,
                                threshold=threshold,
                                feature_columns=numeric_cols
                            )
                            
                            if highly_corr:
                                st.warning(f"Найдено {len(highly_corr)} пар признаков с корреляцией >= {threshold}")
                                corr_df = pd.DataFrame(
                                    highly_corr,
                                    columns=["Признак 1", "Признак 2", "Корреляция"]
                                )
                                st.dataframe(corr_df, use_container_width=True)
                                
                                if st.button("Удалить избыточные признаки"):
                                    df_cleaned, removed = preprocessing.remove_redundant_features(
                                        df_features,
                                        threshold=threshold,
                                        feature_columns=numeric_cols
                                    )
                                    if removed:
                                        st.success(f"Удалено {len(removed)} признаков: {', '.join(removed)}")
                                        st.session_state.df_results = df_cleaned
                                        st.rerun()
                            else:
                                st.info("Нет высоко коррелированных признаков")

        # Вкладка сравнения методов
        if tab_comparison is not None and enable_comparison:
            with tab_comparison:
                st.header("⚖️ Сравнение методов построения шкалы")
                
                # Проверка, какие методы выбраны
                selected_methods = []
                if use_pca_simple:
                    selected_methods.append(("pca_simple", "PCA Scoring"))
                if use_spectral_p1_p99:
                    selected_methods.append(("spectral_p1_p99", "Spectral [1, 99]"))
                if use_spectral_p05_p995:
                    selected_methods.append(("spectral_p05_p995", "Spectral [0.5, 99.5]"))
                if use_spectral_p5_p95:
                    selected_methods.append(("spectral_p5_p95", "Spectral [5, 95]"))
                if use_spectral_gmm:
                    selected_methods.append(("spectral_gmm", "Spectral + GMM"))
                if use_custom_spectral:
                    selected_methods.append((
                        f"spectral_custom_{custom_percentile_low}_{custom_percentile_high}",
                        f"Spectral [{custom_percentile_low}, {custom_percentile_high}]"
                    ))
                
                if not selected_methods:
                    st.warning("Выберите хотя бы один метод для сравнения в боковой панели")
                else:
                    st.info(f"Сравнивается {len(selected_methods)} методов")
                    
                    # Инициализация сравнения
                    comparison = scale_comparison.ScaleComparison()
                    
                    # Запуск выбранных методов
                    with st.spinner("Запуск методов..."):
                        progress_bar = st.progress(0)
                        total_methods = len(selected_methods)
                        
                        for idx, (method_key, method_name) in enumerate(selected_methods):
                            try:
                                if method_key == "pca_simple":
                                    comparison.test_pca_scoring(df_features, name=method_key)
                                
                                elif method_key == "spectral_p1_p99":
                                    comparison.test_spectral_analysis(
                                        df_features,
                                        name=method_key,
                                        percentile_low=1.0,
                                        percentile_high=99.0,
                                        use_gmm=False
                                    )
                                
                                elif method_key == "spectral_p05_p995":
                                    comparison.test_spectral_analysis(
                                        df_features,
                                        name=method_key,
                                        percentile_low=0.5,
                                        percentile_high=99.5,
                                        use_gmm=False
                                    )
                                
                                elif method_key == "spectral_p5_p95":
                                    comparison.test_spectral_analysis(
                                        df_features,
                                        name=method_key,
                                        percentile_low=5.0,
                                        percentile_high=95.0,
                                        use_gmm=False
                                    )
                                
                                elif method_key == "spectral_gmm":
                                    comparison.test_spectral_analysis(
                                        df_features,
                                        name=method_key,
                                        percentile_low=1.0,
                                        percentile_high=99.0,
                                        use_gmm=True
                                    )
                                
                                elif method_key.startswith("spectral_custom_"):
                                    comparison.test_spectral_analysis(
                                        df_features,
                                        name=method_key,
                                        percentile_low=custom_percentile_low,
                                        percentile_high=custom_percentile_high,
                                        use_gmm=False
                                    )
                                
                                progress_bar.progress((idx + 1) / total_methods)
                                
                            except Exception as e:
                                st.error(f"Ошибка при выполнении {method_name}: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    progress_bar.empty()
                    
                    # Сравнение результатов
                    try:
                        comparison_df = comparison.compare_results()
                        stats_df = comparison.get_statistics()
                        
                        # Отображение статистики
                        st.subheader("📊 Статистика по методам")
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Таблица сравнения
                        st.subheader("📋 Сравнение шкал для каждого образца")
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Визуализация
                        st.subheader("📈 Визуализация сравнения")
                        
                        # Создание временного файла для графика
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                            tmp_path = Path(tmp_file.name)
                        
                        try:
                            comparison.visualize_comparison(save_path=tmp_path)
                            if tmp_path.exists():
                                st.image(str(tmp_path))
                                # Скачивание графика
                                with open(tmp_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Скачать график сравнения",
                                        data=f.read(),
                                        file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png"
                                    )
                        finally:
                            if tmp_path.exists():
                                tmp_path.unlink()
                        
                        # Скачивание результатов
                        st.subheader("💾 Скачивание результатов")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_comparison = comparison_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Скачать сравнение (CSV)",
                                data=csv_comparison,
                                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            csv_stats = stats_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Скачать статистику (CSV)",
                                data=csv_stats,
                                file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        # Сохранение в session state для возможности сохранения эксперимента
                        st.session_state.comparison = comparison
                        st.session_state.comparison_df = comparison_df
                        st.session_state.stats_df = stats_df
                        
                    except Exception as e:
                        st.error(f"Ошибка при сравнении результатов: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # Вкладка кластеризации
        with tab_clustering:
            st.header("🔗 Кластеризация данных")
            st.markdown("Выявление скрытых паттернов и патологических фенотипов через кластеризацию.")
            
            if len(df_features) > 0:
                # Настройки кластеризации
                st.subheader("⚙️ Настройки кластеризации")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    clustering_method = st.selectbox(
                        "Метод кластеризации",
                        ["hdbscan", "agglomerative", "kmeans"],
                        help="HDBSCAN: автоматическое определение числа кластеров. Agglomerative/KMeans: требуется указать число кластеров."
                    )
                
                with col2:
                    if clustering_method == "hdbscan":
                        min_cluster_size = st.slider("Минимальный размер кластера", 2, 10, 2)
                        use_pca = st.checkbox("Использовать PCA", value=True)
                        n_clusters = None
                    elif clustering_method == "agglomerative":
                        n_clusters = st.slider("Число кластеров", 2, 10, 3)
                        use_pca = st.checkbox("Использовать PCA", value=True)
                        min_cluster_size = None
                    else:  # kmeans
                        n_clusters = st.slider("Число кластеров", 2, 10, 3)
                        use_pca = st.checkbox("Использовать PCA", value=True)
                        min_cluster_size = None
                
                with col3:
                    if use_pca:
                        pca_components = st.slider("Число компонент PCA", 2, 20, 10)
                    else:
                        pca_components = None
                
                # Запуск кластеризации
                if st.button("🚀 Запустить кластеризацию", type="primary"):
                    with st.spinner("Выполняется кластеризация..."):
                        try:
                            clusterer = clustering.ClusterAnalyzer(
                                method=clustering_method,
                                n_clusters=n_clusters,
                                random_state=42,
                            )
                            
                            clusterer.fit(
                                df_features,
                                use_pca=use_pca,
                                pca_components=pca_components if use_pca else None,
                                min_cluster_size=min_cluster_size if clustering_method == "hdbscan" else 2,
                            )
                            
                            # Сохраняем в session state
                            st.session_state.clusterer = clusterer
                            
                            st.success("✅ Кластеризация завершена!")
                            
                        except Exception as e:
                            st.error(f"Ошибка при кластеризации: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Отображение результатов
                if "clusterer" in st.session_state:
                    clusterer = st.session_state.clusterer
                    
                    # Метрики
                    st.subheader("📊 Метрики качества кластеризации")
                    metrics = clusterer.get_metrics(df_features)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Число кластеров", metrics["n_clusters"])
                    with col2:
                        st.metric("Шум (outliers)", metrics["n_noise"])
                    with col3:
                        if not np.isnan(metrics.get("silhouette_score", np.nan)):
                            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
                        else:
                            st.metric("Silhouette Score", "N/A")
                    with col4:
                        if not np.isnan(metrics.get("calinski_harabasz_score", np.nan)):
                            st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.1f}")
                        else:
                            st.metric("Calinski-Harabasz", "N/A")
                    
                    # Интерпретация кластеров
                    st.subheader("🔍 Интерпретация кластеров")
                    interpretation = clusterer.get_cluster_interpretation()
                    
                    if interpretation:
                        for cluster_id, info in interpretation.items():
                            with st.expander(f"Кластер {cluster_id} ({info['n_samples']} образцов)"):
                                st.markdown(f"**Интерпретация:** {info['interpretation']}")
                                st.markdown(f"**Топ признаки:** {info['features_str']}")
                                
                                # Показываем средние значения признаков
                                if clusterer.cluster_stats_:
                                    cluster_means = clusterer.cluster_stats_["means"].loc[cluster_id]
                                    top_features = cluster_means.nlargest(10)
                                    st.dataframe(
                                        pd.DataFrame({
                                            "Признак": top_features.index,
                                            "Среднее значение": top_features.values
                                        }),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                    else:
                        st.warning("Не удалось интерпретировать кластеры")
                    
                    # Визуализация
                    st.subheader("📈 Визуализация кластеров")
                    
                    # UMAP визуализация
                    if st.checkbox("Показать UMAP визуализацию", value=True):
                        with st.spinner("Обучение UMAP..."):
                            try:
                                clusterer.fit_umap(df_features, n_neighbors=5, min_dist=0.1)
                                
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                                    tmp_path = Path(tmp_file.name)
                                
                                clusterer.visualize_clusters(df_features, save_path=tmp_path)
                                
                                if tmp_path.exists():
                                    st.image(str(tmp_path))
                                    tmp_path.unlink()
                            except Exception as e:
                                st.error(f"Ошибка при визуализации: {e}")
                    
                    # Таблица с результатами
                    st.subheader("📋 Результаты кластеризации")
                    df_with_clusters = clusterer.transform(df_features)
                    
                    # Показываем распределение по кластерам
                    cluster_counts = df_with_clusters["cluster"].value_counts().sort_index()
                    st.markdown("**Распределение по кластерам:**")
                    st.dataframe(
                        pd.DataFrame({
                            "Кластер": cluster_counts.index,
                            "Число образцов": cluster_counts.values
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Таблица с образцами
                    display_cols = ["image", "cluster"]
                    if "PC1" in df_with_clusters.columns:
                        display_cols.append("PC1")
                    if "PC1_spectrum" in df_with_clusters.columns:
                        display_cols.append("PC1_spectrum")
                    
                    st.dataframe(
                        df_with_clusters[display_cols].sort_values("cluster"),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Скачивание результатов
                    csv_clusters = df_with_clusters.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать результаты кластеризации (CSV)",
                        data=csv_clusters,
                        file_name=f"clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Загрузите данные для кластеризации")

    else:
        st.info("👈 Загрузите JSON файлы с предсказаниями в боковой панели")


if __name__ == "__main__":
    render_dashboard()

