"""
Общие функции для dashboard.py и dashboard_minimal.py.

Этот модуль содержит базовые функции, которые используются обоими dashboard'ами,
чтобы обеспечить синхронизацию кода между полной и минимальной версиями.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import re

# Настройка логирования
logger = logging.getLogger(__name__)

# Добавляем путь к проекту для импортов
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import streamlit as st
except ImportError:
    st = None

# Настройка логирования для отладки (определяем ДО импортов)
DEBUG_MODE = False

# Импорт интеграции с Google Drive (опционально)
try:
    from scale.gdrive_integration import (
        is_gdrive_available,
        extract_folder_id_from_url,
        load_json_from_drive_folder,
        upload_file_to_drive,
        get_credentials,
        create_oauth_flow,
        SCOPES,
    )
    GDRIVE_ENABLED = is_gdrive_available()
    if DEBUG_MODE:
        logger.debug(f"[GDRIVE] GDRIVE_ENABLED = {GDRIVE_ENABLED}")
except ImportError as e:
    GDRIVE_ENABLED = False
    create_oauth_flow = None
    SCOPES = []
    if DEBUG_MODE:
        logger.debug(f"[GDRIVE] ImportError: {e}")
except Exception as e:
    GDRIVE_ENABLED = False
    create_oauth_flow = None
    SCOPES = []
    if DEBUG_MODE:
        logger.debug(f"[GDRIVE] Exception при импорте: {e}")

# Импорт интеграции с Google Cloud Storage (опционально)
try:
    from scale.gcs_integration import (
        is_gcs_available,
        load_json_from_gcs_bucket,
    )
    GCS_ENABLED = is_gcs_available()
except ImportError:
    GCS_ENABLED = False


def safe_session_get(key, default=None):
    """
    Безопасное получение значения из session_state.
    
    Используется в обоих dashboard'ах для предотвращения ошибок инициализации.
    """
    if st is None:
        return default
    
    try:
        if not hasattr(st, 'session_state'):
            return default
        try:
            _ = st.session_state
        except (RuntimeError, AttributeError):
            return default
        return st.session_state.get(key, default)
    except (RuntimeError, AttributeError, KeyError, TypeError) as e:
        if DEBUG_MODE:
            print(f"⚠️ DEBUG: Ошибка доступа к session_state['{key}']: {e}")
        return default


def safe_session_set(key, value):
    """Безопасная установка значения в session_state."""
    if st is None:
        return
    
    try:
        if not hasattr(st, 'session_state'):
            if DEBUG_MODE:
                print(f"⚠️ DEBUG: session_state не инициализирован, пропускаем установку '{key}'")
            return
        try:
            _ = st.session_state
        except (RuntimeError, AttributeError):
            if DEBUG_MODE:
                print(f"⚠️ DEBUG: session_state не инициализирован, пропускаем установку '{key}'")
            return
        st.session_state[key] = value
    except (RuntimeError, AttributeError, TypeError) as e:
        if DEBUG_MODE:
            print(f"⚠️ DEBUG: Ошибка установки session_state['{key}']: {e}")


def safe_session_del(key):
    """Безопасное удаление ключа из session_state."""
    if st is None:
        return
    
    try:
        if not hasattr(st, 'session_state'):
            return
        try:
            _ = st.session_state
        except (RuntimeError, AttributeError):
            return
        if key in st.session_state:
            del st.session_state[key]
    except (RuntimeError, AttributeError, KeyError, TypeError):
        pass


def safe_session_has(key):
    """Безопасная проверка наличия ключа в session_state."""
    if st is None:
        return False
    
    try:
        if not hasattr(st, 'session_state'):
            return False
        try:
            _ = st.session_state
        except (RuntimeError, AttributeError):
            return False
        return key in st.session_state
    except (RuntimeError, AttributeError, TypeError):
        return False


def load_predictions_from_files(json_files: List[Path]) -> Dict[str, dict]:
    """
    Загружает предсказания из JSON файлов.
    
    Используется в обоих dashboard'ах для загрузки данных из локальных файлов.
    """
    predictions = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            predictions[json_file.stem] = data
        except Exception as e:
            if st:
                st.error(f"Ошибка при загрузке {json_file.name}: {e}")
            else:
                print(f"Ошибка при загрузке {json_file.name}: {e}")
    return predictions


def load_predictions_from_upload_basic(uploaded_files) -> Dict[str, dict]:
    """
    Базовая загрузка предсказаний из загруженных файлов (без domain.predictions_from_dict).
    
    Используется в dashboard_minimal.py для простой загрузки JSON.
    """
    predictions = {}
    for uploaded_file in uploaded_files:
        try:
            data = json.load(uploaded_file)
            predictions[Path(uploaded_file.name).stem] = data
        except Exception as e:
            if st:
                st.error(f"Ошибка при загрузке {uploaded_file.name}: {e}")
            else:
                print(f"Ошибка при загрузке {uploaded_file.name}: {e}")
    return predictions


def load_predictions_from_upload(uploaded_files) -> Dict[str, dict]:
    """
    Загружает предсказания из загруженных файлов с использованием domain.predictions_from_dict.
    
    Используется в dashboard.py для полной обработки данных.
    """
    try:
        from scale import domain
    except ImportError:
        # Fallback на базовую загрузку, если domain недоступен
        return load_predictions_from_upload_basic(uploaded_files)
    
    predictions = {}
    for uploaded_file in uploaded_files:
        try:
            data = json.load(uploaded_file)
            image_name = Path(uploaded_file.name).stem
            predictions[image_name] = domain.predictions_from_dict(data)
        except Exception as e:
            if st:
                st.error(f"Ошибка при загрузке {uploaded_file.name}: {e}")
            else:
                print(f"Ошибка при загрузке {uploaded_file.name}: {e}")
    return predictions


def load_predictions_from_gdrive(
    drive_folder_url: str,
    credentials_path: Optional[str] = None,
    log_callback: Optional[callable] = None
) -> Dict[str, dict]:
    """
    Загружает предсказания из папки Google Drive.
    
    Используется в обоих dashboard'ах для загрузки данных из Google Drive.
    
    Args:
        drive_folder_url: URL папки Google Drive
        credentials_path: Путь к credentials.json
        log_callback: Функция для логирования (message)
    """
    if not GDRIVE_ENABLED:
        error_msg = "❌ Интеграция с Google Drive недоступна"
        if log_callback:
            log_callback(error_msg)
        if st:
            st.error(error_msg)
        return {}
    
    folder_id = extract_folder_id_from_url(drive_folder_url)
    if not folder_id:
        error_msg = "❌ Не удалось извлечь ID папки из ссылки"
        if log_callback:
            log_callback(error_msg)
        if st:
            st.error(error_msg)
        return {}
    
    credentials = get_credentials(credentials_path=credentials_path)
    if not credentials:
        error_msg = "⚠️ Требуется авторизация Google Drive"
        if log_callback:
            log_callback(error_msg)
        if st:
            st.warning(error_msg)
        return {}
    
    return load_json_from_drive_folder(
        drive_folder_url,
        credentials=credentials,
        credentials_path=credentials_path,
        log_callback=log_callback
    )


def save_files_to_gdrive(
    files: List[tuple],  # List of (file_path: Path, file_name: str) or (uploaded_file, None)
    folder_id: str,
    credentials_path: Optional[str] = None
) -> int:
    """
    Сохраняет файлы в Google Drive с индикацией прогресса.
    
    Args:
        files: Список кортежей (file_path или uploaded_file, file_name)
        folder_id: ID папки в Google Drive
        credentials_path: Путь к credentials.json
        
    Returns:
        Количество успешно сохраненных файлов
    """
    if not GDRIVE_ENABLED:
        if st:
            st.error("❌ Интеграция с Google Drive недоступна")
        return 0
    
    credentials = get_credentials(credentials_path=credentials_path)
    if not credentials:
        if st:
            st.warning("⚠️ Требуется авторизация Google Drive")
        return 0
    
    saved_count = 0
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    total_files = len(files)
    
    # Создаем progress bar если доступен Streamlit
    progress_bar = None
    status_text = None
    if st and total_files > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        for idx, file_item in enumerate(files):
            try:
                if isinstance(file_item, tuple):
                    file_obj, file_name = file_item
                else:
                    file_obj = file_item
                    file_name = None
                
                # Определяем путь к файлу
                if hasattr(file_obj, 'name'):  # Streamlit UploadedFile
                    temp_path = temp_dir / file_obj.name
                    temp_path.write_bytes(file_obj.getbuffer())
                    file_name = file_name or file_obj.name
                elif isinstance(file_obj, (str, Path)):  # Путь к файлу
                    temp_path = Path(file_obj)
                    file_name = file_name or temp_path.name
                else:
                    continue
                
                if not temp_path.exists():
                    continue
                
                # Обновляем статус
                if status_text:
                    file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                    status_text.text(f"📤 Загрузка файла {idx + 1}/{total_files}: {file_name} ({file_size_mb:.2f} MB)")
                
                # Callback для прогресса загрузки одного файла
                def update_progress(progress, total):
                    if progress_bar:
                        # Общий прогресс: завершенные файлы + прогресс текущего файла
                        overall_progress = (idx / total_files) + (progress / 100 / total_files)
                        progress_bar.progress(min(overall_progress, 1.0))
                
                # Загружаем в Drive
                file_id = upload_file_to_drive(
                    temp_path,
                    folder_id,
                    credentials=credentials,
                    credentials_path=credentials_path,
                    file_name=file_name,
                    progress_callback=update_progress
                )
                
                if file_id:
                    saved_count += 1
                    if status_text:
                        status_text.text(f"✅ Файл {idx + 1}/{total_files} загружен: {file_name}")
                
                # Обновляем общий прогресс
                if progress_bar:
                    progress_bar.progress((idx + 1) / total_files)
                
                # Удаляем временный файл, если он был создан
                if hasattr(file_obj, 'name') and temp_path.exists():
                    temp_path.unlink()
                    
            except Exception as e:
                error_msg = f"Ошибка при сохранении файла {file_name if 'file_name' in locals() else 'unknown'}: {e}"
                if st:
                    st.error(error_msg)
                else:
                    print(error_msg)
                # Продолжаем с следующим файлом
                continue
        
        # Финальный статус
        if status_text:
            status_text.text(f"✅ Загружено {saved_count} из {total_files} файлов")
        if progress_bar:
            progress_bar.progress(1.0)
            
    finally:
        # Очищаем временную директорию если пуста
        try:
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except:
            pass
    
    return saved_count


def render_gdrive_upload_section(
    uploaded_files: Optional[List] = None,
    predictions: Optional[Dict] = None
) -> Optional[str]:
    """
    Рендерит секцию сохранения в Google Drive для боковой панели.
    
    Returns:
        Folder ID если файлы были сохранены, иначе None
    """
    if not GDRIVE_ENABLED or not st:
        return None
    
    if not (uploaded_files or predictions):
        return None
    
    st.markdown("---")
    st.subheader("💾 Сохранение в Google Drive")
    
    drive_folder_url = st.text_input(
        "Ссылка на папку Google Drive для сохранения",
        placeholder="https://drive.google.com/drive/folders/...",
        help="Вставьте ссылку на папку, куда сохранить файлы",
        key="gdrive_save_folder_url"
    )
    
    # Проверяем авторизацию
    import os
    default_creds_path = os.path.join(os.path.expanduser('~'), '.config', 'gdrive', 'credentials.json')
    creds_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH', default_creds_path)
    credentials = get_credentials(credentials_path=creds_path)
    
    if not credentials:
        # Пытаемся авторизоваться
        authorize_gdrive(creds_path)
        credentials = get_credentials(credentials_path=creds_path)
    
    if st.button("Сохранить в Google Drive", key="gdrive_save_button") and drive_folder_url:
        if not credentials:
            st.error("❌ Требуется авторизация Google Drive. Используйте кнопку авторизации выше.")
            return None
            
        folder_id = extract_folder_id_from_url(drive_folder_url)
        if folder_id:
            # Подготавливаем файлы для сохранения
            files_to_save = []
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    files_to_save.append((uploaded_file, None))
            
            saved_count = save_files_to_gdrive(
                files_to_save,
                folder_id,
                credentials_path=creds_path
            )
            
            if saved_count > 0:
                st.success(f"✅ Сохранено {saved_count} файлов в Google Drive")
                return folder_id
            else:
                st.error("❌ Не удалось сохранить файлы")
        else:
            st.error("❌ Не удалось извлечь ID папки из ссылки")
    
    return None


def authorize_gdrive(creds_path: str) -> bool:
    """
    Авторизует пользователя через OAuth flow для Google Drive.
    
    Args:
        creds_path: Путь к credentials.json
        
    Returns:
        True если авторизация успешна, False иначе
    """
    if not GDRIVE_ENABLED or not st or not create_oauth_flow:
        return False
    
    import os
    from pathlib import Path
    from urllib.parse import urlencode
    
    # Проверяем наличие credentials.json
    if not Path(creds_path).exists():
        st.error(f"❌ **Файл credentials.json не найден!**")
        st.caption(f"""
        Ожидаемый путь: `{creds_path}`
        
        **Что делать:**
        1. Создайте OAuth credentials в [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
        2. Скачайте credentials.json
        3. Положите файл в: `~/.config/gdrive/credentials.json`
           ```bash
           mkdir -p ~/.config/gdrive
           cp ~/Downloads/credentials.json ~/.config/gdrive/credentials.json
           ```
        4. Или установите переменную окружения:
           ```bash
           export GOOGLE_DRIVE_CREDENTIALS_PATH="/path/to/credentials.json"
           ```
        
        Подробнее: см. `docs/GOOGLE_OAUTH_SETUP_RU.md`
        """)
        return False
    
    # Получаем текущий URL для redirect URI
    try:
        # Пытаемся определить, работаем ли мы на Cloud Run или локально
        # Cloud Run устанавливает переменную K_SERVICE
        if os.getenv('K_SERVICE'):
            # На Cloud Run - используем URL сервиса СО слешем (как в credentials.json)
            redirect_uri = "https://dashboard-gia5jttcaq-uc.a.run.app/"
        else:
            # Локальная разработка
            port = os.getenv('STREAMLIT_SERVER_PORT', '8501')
            redirect_uri = f"http://localhost:{port}"
    except:
        # Fallback для Cloud Run
        redirect_uri = "https://dashboard-gia5jttcaq-uc.a.run.app/"
    
    # Создаем OAuth flow
    flow = create_oauth_flow(creds_path, redirect_uri=redirect_uri)
    if not flow:
        st.error("❌ Не удалось создать OAuth flow. Проверьте credentials.json")
        return False
    
    # Проверяем callback (query параметры)
    query_params = st.query_params
    code = query_params.get("code")
    
    if code:
        # Обрабатываем callback
        try:
            flow.fetch_token(code=code)
            credentials = flow.credentials
            
            # Сохраняем токен
            token_path = Path('.gdrive_token.json')
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, 'w') as token_file:
                token_file.write(credentials.to_json())
            
            st.success("✅ Авторизация успешна! Теперь вы можете использовать Google Drive.")
            st.rerun()
            return True
        except Exception as e:
            error_msg = str(e)
            if "access_denied" in error_msg or "403" in error_msg:
                st.error("❌ **Ошибка 403: Доступ запрещен**")
                st.caption("""
                **Эта ошибка обычно возникает из-за настроек OAuth Consent Screen:**
                
                **Решение:**
                
                1. **Откройте OAuth Consent Screen:**
                   - Перейдите в [Google Cloud Console](https://console.cloud.google.com/apis/credentials/consent)
                   - Выберите ваш проект
                
                2. **Если приложение в режиме "Testing":**
                   - Найдите раздел "Test users"
                   - Нажмите "+ ADD USERS"
                   - Добавьте ваш email адрес (тот, которым вы авторизуетесь)
                   - Сохраните
                
                3. **Или переведите приложение в Production:**
                   - В разделе "Publishing status" нажмите "PUBLISH APP"
                   - Заполните форму (название приложения, email поддержки)
                   - Отправьте на проверку (может занять несколько дней)
                
                4. **Проверьте настройки:**
                   - Убедитесь, что Google Drive API включен
                   - Проверьте, что redirect URI `http://localhost:8501` добавлен в OAuth credentials
                
                **Быстрое решение для тестирования:**
                - Добавьте себя в "Test users" в OAuth Consent Screen
                - Это позволит авторизоваться сразу без ожидания верификации
                
                Подробнее: см. `docs/GOOGLE_OAUTH_SETUP_RU.md`
                """)
            else:
                st.error(f"❌ Ошибка при авторизации: {e}")
            return False
    else:
        # Генерируем authorization URL
        try:
            auth_url, state = flow.authorization_url(prompt='consent', access_type='offline')
            
            # Сохраняем state в session для проверки
            safe_session_set('gdrive_oauth_state', state)
            
            st.info("🔐 **Нажмите на кнопку ниже для авторизации в Google Drive**")
            st.link_button("🔐 Авторизоваться в Google Drive", auth_url)
            st.caption(f"""
            **Инструкция:**
            1. Нажмите на кнопку выше
            2. Войдите в свой Google аккаунт
            3. Разрешите доступ к Google Drive
            4. Вы будете перенаправлены обратно в dashboard
            
            **Redirect URI:** `{redirect_uri}`
            
            ⚠️ Убедитесь, что этот URI добавлен в Google Cloud Console в настройках OAuth credentials!
            """)
            return False
        except Exception as e:
            st.error(f"❌ Ошибка при создании authorization URL: {e}")
            st.caption(f"Проверьте, что redirect URI `{redirect_uri}` добавлен в Google Cloud Console.")
            return False


def render_gdrive_load_section(data_source_selected: str = None) -> tuple:
    """
    Рендерит секцию загрузки из Google Drive или GCS для боковой панели.
    
    Args:
        data_source_selected: Выбранный источник данных из основного меню ("Google Drive" или "Google Cloud Storage (GCS)")
    
    Returns:
        Tuple (source_info, predictions_dict)
    """
    if not st:
        return None, {}
    
    # Определяем, какой источник использовать на основе переданного параметра или session state
    if data_source_selected:
        source = data_source_selected
    else:
        # Fallback: используем session state
        source = safe_session_get("data_source", None)
    
    # КРИТИЧНО: Если источник не определен, пытаемся определить его из session state или доступных опций
    if not source:
        # Проверяем, есть ли загруженные данные из cloud storage
        cloud_predictions = safe_session_get("predictions_cloud", None)
        if cloud_predictions and len(cloud_predictions) > 0:
            # Есть данные - определяем источник по data_source или по доступным опциям
            saved_source = safe_session_get("data_source", None)
            if saved_source in ["Google Drive", "Google Cloud Storage (GCS)"]:
                source = saved_source
            elif GCS_ENABLED:
                source = "Google Cloud Storage (GCS)"
            elif GDRIVE_ENABLED:
                source = "Google Drive"
        else:
            # Нет данных - используем первый доступный источник
            if GCS_ENABLED:
                source = "Google Cloud Storage (GCS)"
            elif GDRIVE_ENABLED:
                source = "Google Drive"
    
    # Показываем соответствующий интерфейс в зависимости от выбранного источника
    if source == "Google Drive" and GDRIVE_ENABLED:
        return _render_gdrive_load()
    elif source == "Google Cloud Storage (GCS)" and GCS_ENABLED:
        result = _render_gcs_load()
        # Сохраняем информацию об источнике
        if result and result[0]:
            safe_session_set("gdrive_load_source_info", f"gcs://{result[0]}")
        return result
    else:
        # Если источник не определен или недоступен, показываем предупреждение
        if not GDRIVE_ENABLED and not GCS_ENABLED:
            st.warning("⚠️ Ни Google Drive, ни GCS не доступны.")
        elif source == "Google Drive" and not GDRIVE_ENABLED:
            st.error("❌ Google Drive недоступен")
        elif source == "Google Cloud Storage (GCS)" and not GCS_ENABLED:
            st.error("❌ Google Cloud Storage недоступен")
        return None, {}


def _render_gdrive_load() -> tuple:
    """Рендерит секцию загрузки из Google Drive - только кнопка загрузки (поля ввода в sidebar)."""
    if not GDRIVE_ENABLED or not st:
        return None, {}
    
    st.markdown("---")
    st.subheader("📥 Google Drive")
    
    # КРИТИЧНО: URL берем из session state (устанавливается в sidebar)
    drive_folder_url = safe_session_get("gdrive_load_url", "")
    
    import os
    default_creds_path = os.path.join(os.path.expanduser('~'), '.config', 'gdrive', 'credentials.json')
    creds_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH', default_creds_path)
    credentials = get_credentials(credentials_path=creds_path)
    
    # Показываем информацию о текущих настройках
    if drive_folder_url:
        st.info(f"📂 Папка: `{drive_folder_url[:50]}...`" if len(drive_folder_url) > 50 else f"📂 Папка: `{drive_folder_url}`")
    else:
        st.info("👈 Введите ссылку на папку Google Drive в боковой панели")
    
    # КРИТИЧНО: ВСЕГДА показываем кнопку загрузки, даже если данные уже загружены
    # Это позволяет пользователю перезагрузить данные или загрузить другие данные
    load_button_clicked = st.button("📥 Загрузить из Google Drive", key="gdrive_load_button", type="primary")
    
    # Проверяем наличие URL и credentials только при нажатии кнопки
    if load_button_clicked:
        if not drive_folder_url:
            st.error("❌ Пожалуйста, введите ссылку на папку Google Drive в боковой панели")
            return None, {}
        
        if not credentials:
            st.error("❌ Требуется авторизация Google Drive. Используйте кнопку авторизации в боковой панели.")
            return None, {}
        # Устанавливаем флаг, что загрузка была запрошена
        safe_session_set("gdrive_load_triggered", True)
        safe_session_set("gdrive_load_url", drive_folder_url)
        logger.debug(f"[GDRIVE] Кнопка нажата, начинаю загрузку из: {drive_folder_url}")
        
        # Создаем прогрессбар
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        def log_to_ui(message):
            # Парсим прогресс из сообщений типа "[1/36]" или "📥 Начинаю загрузку 36 JSON файлов..."
            progress_match = re.search(r'\[(\d+)/(\d+)\]', message)
            if progress_match:
                current = int(progress_match.group(1))
                total = int(progress_match.group(2))
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                progress_text.text(f"📥 Загрузка файлов: {current}/{total}")
            elif "Начинаю загрузку" in message:
                total_match = re.search(r'(\d+)\s+JSON файлов', message)
                if total_match:
                    total = int(total_match.group(1))
                    progress_text.text(f"📥 Найдено файлов: {total}")
        
        predictions = load_predictions_from_gdrive(
            drive_folder_url,
            credentials_path=creds_path,
            log_callback=log_to_ui
        )
        
        # Завершаем прогрессбар
        progress_bar.progress(1.0)
        if predictions:
            progress_text.text(f"✅ Загружено {len(predictions)} файлов")
        else:
            progress_text.text("⚠️ Файлы не найдены")
        
        if predictions:
            st.success(f"✅ Загружено {len(predictions)} файлов из Google Drive")
            # Сохраняем predictions в session state для использования в dashboard
            # Конвертируем в формат domain.predictions_from_dict
            try:
                from scale import domain
                predictions_converted = {}
                for name, data in predictions.items():
                    predictions_converted[name] = domain.predictions_from_dict(data)
                
                # КРИТИЧНО: Сохраняем данные ПЕРЕД любыми st.write() или st.rerun()
                # Это гарантирует, что данные будут в session state даже если rerun произойдет сразу
                safe_session_set("predictions_cloud", predictions_converted)
                safe_session_set("use_cloud_storage", True)
                safe_session_set("data_source", "Google Drive")
                
                # КРИТИЧНО: Проверяем, не был ли уже вызван rerun для этих данных
                last_loaded_hash = safe_session_get("gdrive_last_loaded_hash", None)
                current_hash = hash(str(sorted(predictions_converted.keys())))
                
                if last_loaded_hash != current_hash:
                    safe_session_set("gdrive_last_loaded_hash", current_hash)
                    # Сбрасываем флаг загрузки
                    safe_session_set("gdrive_load_triggered", False)
                    st.success("✅ Данные загружены и готовы к использованию!")
                    logger.debug(f"[GDRIVE] Данные успешно загружены и сохранены: {len(predictions_converted)} файлов")
                else:
                    st.info("ℹ️ Данные уже загружены ранее")
                    # Сбрасываем флаг, так как данные уже были загружены
                    safe_session_set("gdrive_load_triggered", False)
                
                # Возвращаем данные - они будут использованы в dashboard
                return drive_folder_url, predictions_converted
            except Exception as e:
                st.error(f"Ошибка при обработке данных из Google Drive: {e}")
                return drive_folder_url, {}
        else:
            st.warning("⚠️ Не найдено JSON файлов в указанной папке")
            return drive_folder_url, {}
    
    # КРИТИЧНО: Если данные уже загружены, возвращаем их, но интерфейс все равно показывается выше
    # Это позволяет пользователю видеть интерфейс и загружать другие данные
    # Кнопка уже была показана выше, поэтому она всегда видна
    existing_predictions = safe_session_get("predictions_cloud", None)
    if existing_predictions and len(existing_predictions) > 0:
        # Проверяем, что это данные из Google Drive (не из GCS)
        current_data_source = safe_session_get("data_source", None)
        if current_data_source == "Google Drive":
            # Показываем информацию о загруженных данных
            st.info(f"✅ Данные уже загружены: {len(existing_predictions)} файлов. Вы можете загрузить другие данные, нажав кнопку выше.")
            return drive_folder_url, existing_predictions
    
    return None, {}


def _render_gcs_load() -> tuple:
    """Рендерит секцию загрузки из Google Cloud Storage - только кнопка загрузки (поля ввода в sidebar)."""
    if not GCS_ENABLED or not st:
        return None, {}
    
    st.markdown("---")
    st.subheader("📥 Google Cloud Storage")
    
    st.info("⚡ **GCS быстрее на Cloud Run!** Используйте GCS для лучшей производительности.")
    
    # КРИТИЧНО: bucket_name берем из session state (устанавливается в sidebar)
    bucket_name = safe_session_get("gcs_bucket_name", "scalebucket")
    
    # Prefix не используется - всегда пустая строка
    prefix = ""
    
    # Показываем информацию о текущих настройках
    if bucket_name:
        st.info(f"📦 Bucket: `{bucket_name}`")
    else:
        st.info("👈 Введите имя GCS bucket в боковой панели")
    
    # КРИТИЧНО: ВСЕГДА показываем кнопку загрузки, даже если данные уже загружены
    # Это позволяет пользователю перезагрузить данные или загрузить другие данные
    load_button_clicked = st.button("📥 Загрузить из GCS", key="gcs_load_button", type="primary")
    
    # Проверяем наличие bucket и авторизации только при нажатии кнопки
    if load_button_clicked:
        if not bucket_name:
            st.error("❌ Пожалуйста, введите имя GCS bucket в боковой панели")
            return None, {}
        
        # Проверка авторизации
        auth_ok = False
        try:
            from google.cloud import storage
            client = storage.Client()
            try:
                _ = list(client.list_buckets(max_results=1))
                auth_ok = True
            except Exception:
                auth_ok = False
        except Exception:
            auth_ok = False
        
        if not auth_ok:
            st.error("❌ Требуется авторизация GCS. См. инструкции в боковой панели.")
            return None, {}
        
        # Устанавливаем флаг, что загрузка была запрошена
        safe_session_set("gcs_load_triggered", True)
        safe_session_set("gcs_load_bucket", bucket_name)
        logger.debug(f"[GCS] Кнопка нажата, начинаю загрузку из bucket: {bucket_name}")
        # Вызываем загрузку сразу (prefix всегда пустой)
        return _load_from_gcs(bucket_name, "")
    
    # КРИТИЧНО: Если данные уже загружены, возвращаем их, но интерфейс все равно показывается выше
    # Это позволяет пользователю видеть интерфейс и загружать другие данные
    # Кнопка уже была показана выше, поэтому она всегда видна
    existing_predictions = safe_session_get("predictions_cloud", None)
    if existing_predictions and len(existing_predictions) > 0:
        # Проверяем, что это данные из GCS (не из Google Drive)
        current_data_source = safe_session_get("data_source", None)
        if current_data_source == "Google Cloud Storage (GCS)":
            # Показываем информацию о загруженных данных
            st.info(f"✅ Данные уже загружены: {len(existing_predictions)} файлов. Вы можете загрузить другие данные, нажав кнопку выше.")
            return f"gcs://{bucket_name}", existing_predictions
    
    return None, {}


def _load_from_gcs(bucket_name: str, prefix: str = "") -> tuple:
    """Загружает данные из Google Cloud Storage."""
    from scale.gcs_integration import load_json_from_gcs_bucket
    
    # Создаем прогрессбар
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    def log_to_ui(message):
        # Парсим прогресс из сообщений типа "[1/36]" или "📥 Начинаю загрузку 36 JSON файлов..."
        progress_match = re.search(r'\[(\d+)/(\d+)\]', message)
        if progress_match:
            current = int(progress_match.group(1))
            total = int(progress_match.group(2))
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            progress_text.text(f"📥 Загрузка файлов: {current}/{total}")
        elif "Начинаю загрузку" in message:
            total_match = re.search(r'(\d+)\s+JSON файлов', message)
            if total_match:
                total = int(total_match.group(1))
                progress_text.text(f"📥 Найдено файлов: {total}")
    
    predictions = load_json_from_gcs_bucket(
        bucket_name,
        prefix=prefix,
        log_callback=log_to_ui
    )
    
    # Завершаем прогрессбар
    progress_bar.progress(1.0)
    if predictions:
        progress_text.text(f"✅ Загружено {len(predictions)} файлов")
    else:
        progress_text.text("⚠️ Файлы не найдены")
    
    if predictions:
        st.success(f"✅ Загружено {len(predictions)} файлов из GCS bucket")
        
        # Сохраняем predictions в session state для использования в dashboard
        # Конвертируем в формат domain.predictions_from_dict
        try:
            from scale import domain
            predictions_converted = {}
            for name, data in predictions.items():
                predictions_converted[name] = domain.predictions_from_dict(data)
            
            # КРИТИЧНО: Сохраняем данные ПЕРЕД любыми st.write() или st.rerun()
            # Это гарантирует, что данные будут в session state даже если rerun произойдет сразу
            safe_session_set("predictions_cloud", predictions_converted)
            safe_session_set("use_cloud_storage", True)
            safe_session_set("data_source", "Google Cloud Storage (GCS)")
            
            # КРИТИЧНО: Проверяем, не был ли уже вызван rerun для этих данных
            last_loaded_hash = safe_session_get("gcs_last_loaded_hash", None)
            current_hash = hash(str(sorted(predictions_converted.keys())))
            
            if last_loaded_hash != current_hash:
                safe_session_set("gcs_last_loaded_hash", current_hash)
                # Сбрасываем флаг загрузки
                safe_session_set("gcs_load_triggered", False)
                st.success("✅ Данные загружены и готовы к использованию!")
                logger.debug(f"[GCS] Данные успешно загружены и сохранены: {len(predictions_converted)} файлов")
            else:
                st.info("ℹ️ Данные уже загружены ранее")
                # Сбрасываем флаг, так как данные уже были загружены
                safe_session_set("gcs_load_triggered", False)
            
            # Возвращаем данные - они будут использованы в dashboard
            return f"gcs://{bucket_name}", predictions_converted
        except Exception as e:
            st.error(f"Ошибка при обработке данных из GCS: {e}")
            return f"gcs://{bucket_name}", {}
    else:
        st.warning("⚠️ Не найдено JSON файлов в указанном bucket")
        return f"gcs://{bucket_name}", {}

