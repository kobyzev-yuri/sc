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

from scale import aggregate, spectral_analysis, domain


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

        uploaded_files = st.file_uploader(
            "Загрузите JSON файлы с предсказаниями",
            type=["json"],
            accept_multiple_files=True,
        )

        st.markdown("---")

        st.header("⚙️ Настройки")

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
                st.success(f"Эксперимент сохранен: {exp_dir}")
            else:
                st.warning("Нет данных для сохранения")

    # Основная область
    if uploaded_files:
        # Загрузка предсказаний
        with st.spinner("Загрузка предсказаний..."):
            predictions = load_predictions_from_upload(uploaded_files)

        if not predictions:
            st.error("Не удалось загрузить предсказания")
            return

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

        st.session_state.df_results = df_features
        st.session_state.settings = {
            "use_relative_features": use_relative_features,
            "use_spectral_analysis": use_spectral_analysis,
            "percentile_low": percentile_low,
            "percentile_high": percentile_high,
        }

        # Вкладки для визуализации
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Данные", "📈 Распределения", "🔬 Спектральный анализ", "📋 Статистика"]
        )

        with tab1:
            st.header("Загруженные данные")
            st.dataframe(df_features, use_container_width=True)

            # Скачивание CSV
            csv = df_features.to_csv(index=False)
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

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

                col1, col2, col3 = st.columns(3)
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

                # Визуализация спектра
                st.subheader("Визуализация спектра")

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
                display_cols = ["image", "PC1", "PC1_spectrum"]
                if "PC1_mode" in df_spectrum.columns:
                    display_cols.append("PC1_mode")

                st.dataframe(
                    df_spectrum[display_cols].sort_values(
                        by="PC1_spectrum", ascending=False
                    ),
                    use_container_width=True,
                )

                # Важность признаков
                st.subheader("Важность признаков (PC1 loadings)")
                feature_importance = analyzer.get_feature_importance()

                fig, ax = plt.subplots(figsize=(10, 6))
                top_n = st.slider("Показать топ N признаков", 5, 30, 15)
                top_features = feature_importance.head(top_n)

                ax.barh(
                    range(len(top_features)),
                    top_features.values,
                    align="center",
                )
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features.index)
                ax.set_xlabel("Loading value")
                ax.set_title("Важность признаков в PC1")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)

            else:
                st.info("Включите спектральный анализ в настройках")

        with tab4:
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
                        corr_matrix = df_features[numeric_cols].corr()

                        fig, ax = plt.subplots(figsize=(12, 10))
                        im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
                        ax.set_xticks(range(len(corr_matrix.columns)))
                        ax.set_yticks(range(len(corr_matrix.columns)))
                        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
                        ax.set_yticklabels(corr_matrix.columns)
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)

    else:
        st.info("👈 Загрузите JSON файлы с предсказаниями в боковой панели")


if __name__ == "__main__":
    render_dashboard()

