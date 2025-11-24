"""
Минимальный Dashboard для развертывания в Google Cloud.

Упрощенная версия дашборда с базовым функционалом:
- Загрузка JSON файлов с предсказаниями
- Загрузка из Google Drive (расшаренные папки)
- Сохранение в Google Drive
- Визуализация данных
- Базовый анализ

Использует общие функции из scale.dashboard_common для синхронизации с dashboard.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Добавляем путь к проекту для импортов
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
except ImportError as e:
    raise ImportError(
        "Требуются зависимости для дашборда. Установите: pip install streamlit matplotlib pandas numpy"
    ) from e

# Импорт общих функций из dashboard_common
from scale.dashboard_common import (
    safe_session_get,
    safe_session_set,
    load_predictions_from_files,
    load_predictions_from_gdrive,
    render_gdrive_upload_section,
    render_gdrive_load_section,
    GDRIVE_ENABLED,
)


def aggregate_predictions(predictions: Dict[str, dict]) -> pd.DataFrame:
    """Агрегирует предсказания в DataFrame."""
    rows = []
    for image_name, pred_data in predictions.items():
        if isinstance(pred_data, dict):
            # Простая агрегация: суммируем все числовые значения
            row = {'image_name': image_name}
            for key, value in pred_data.items():
                if isinstance(value, (int, float)):
                    row[key] = value
                elif isinstance(value, list) and len(value) > 0:
                    # Если список чисел, берем среднее
                    if all(isinstance(x, (int, float)) for x in value):
                        row[key] = np.mean(value)
                    else:
                        row[key] = len(value)  # Количество элементов
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Основная функция дашборда."""
    st.set_page_config(
        page_title="Dashboard - Анализ данных",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Dashboard - Анализ данных")
    
    # Информация о вариантах хранения
    with st.expander("ℹ️ О вариантах хранения данных", expanded=False):
        st.info("""
        **Два варианта работы с данными:**
        
        1. **Локальные директории** (ephemeral storage)
           - Файлы загружаются внутрь контейнера
           - Пропадают при перезапуске (это нормально!)
           - ✅ Идеально для быстрого тестирования и временной работы
        
        2. **Google Drive** (постоянное хранилище)
           - Файлы хранятся в Google Drive
           - Доступны после перезапуска
           - ✅ Идеально для постоянного хранения и совместной работы
        
        Оба варианта доступны в боковой панели. Выберите нужный в зависимости от задачи!
        
        Подробнее: см. `docs/STORAGE_OPTIONS.md`
        """)
    
    st.markdown("---")

    # Боковая панель для загрузки данных
    with st.sidebar:
        st.header("📁 Загрузка данных")
        
        # Опции источника данных
        data_source_options = ["Использовать директорию"]
        if GDRIVE_ENABLED:
            data_source_options.append("Google Drive")
        
        data_source = st.radio(
            "Источник данных",
            data_source_options,
            index=0
        )

        predictions = {}
        
        if data_source == "Использовать директорию":
            data_dir = st.text_input(
                "Путь к директории с JSON файлами",
                value="results/predictions",
                placeholder="results/predictions"
            )
            
            if st.button("Загрузить из директории"):
                data_path = Path(data_dir)
                if data_path.exists():
                    json_files = list(data_path.glob("*.json"))
                    if json_files:
                        predictions = load_predictions_from_files(json_files)
                        st.success(f"✓ Загружено {len(predictions)} файлов из {data_dir}")
                    else:
                        st.warning(f"⚠ В директории {data_dir} нет JSON файлов")
                else:
                    st.error(f"❌ Директория {data_dir} не найдена")
        
        elif data_source == "Google Drive" and GDRIVE_ENABLED:
            # Используем общую функцию для загрузки из Google Drive
            drive_folder_url, gdrive_predictions = render_gdrive_load_section()
            if gdrive_predictions:
                predictions = gdrive_predictions
        
        elif data_source == "Google Drive" and not GDRIVE_ENABLED:
            st.error("❌ Интеграция с Google Drive недоступна")
            st.caption("Установите зависимости: `pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`")

    # Основная область
    if not predictions:
        st.info("👈 Загрузите данные через боковую панель")
        return

    # Агрегация данных
    df = aggregate_predictions(predictions)
    
    if df.empty:
        st.warning("⚠ Не удалось создать DataFrame из загруженных данных")
        return

    st.header("📈 Обзор данных")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего образцов", len(df))
    with col2:
        st.metric("Признаков", len(df.columns) - 1)  # -1 для image_name
    with col3:
        st.metric("Заполненность", f"{df.notna().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%")

    # Таблица данных
    st.subheader("📋 Данные")
    st.dataframe(df, use_container_width=True)

    # Статистика
    st.subheader("📊 Статистика")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    else:
        st.info("Нет числовых колонок для статистики")

    # Визуализация
    if numeric_cols:
        st.subheader("📈 Визуализация")
        
        selected_col = st.selectbox(
            "Выберите признак для визуализации",
            numeric_cols
        )
        
        if selected_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[selected_col].dropna(), bins=30, edgecolor='black')
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Частота")
            ax.set_title(f"Распределение {selected_col}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Корреляция (если есть несколько числовых колонок)
            if len(numeric_cols) > 1:
                st.subheader("🔗 Корреляционная матрица")
                corr_matrix = df[numeric_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                im = ax2.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax2.set_xticks(range(len(corr_matrix.columns)))
                ax2.set_yticks(range(len(corr_matrix.columns)))
                ax2.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax2.set_yticklabels(corr_matrix.columns)
                ax2.set_title("Корреляционная матрица")
                plt.colorbar(im, ax=ax2)
                st.pyplot(fig2)

    # Экспорт данных
    st.subheader("💾 Экспорт")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать данные как CSV",
        data=csv,
        file_name="dashboard_data.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()

