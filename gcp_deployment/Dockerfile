# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости (если нужны)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (включая credentials из .config/)
COPY . .

# Создаем необходимые директории (если их нет)
RUN mkdir -p experiments results/visualization
RUN mkdir -p .config/gdrive .config/gcs
RUN mkdir -p results/predictions results/inference

# Копируем файлы из results/inference (2 файла, которые есть на сервере)
# COPY скопирует только существующие файлы
COPY results/inference/ results/inference/

# Проверяем наличие credentials (для отладки)
RUN echo "🔍 Проверка credentials в Docker образе:" && \
    echo "   CWD: $(pwd)" && \
    echo "   HOME: $HOME" && \
    echo "   Содержимое .config/:" && \
    ls -la .config/ 2>&1 || echo "   .config/ не существует" && \
    echo "   Содержимое .config/gdrive/:" && \
    ls -la .config/gdrive/ 2>&1 || echo "   .config/gdrive/ не существует" && \
    echo "   Содержимое .config/gcs/:" && \
    ls -la .config/gcs/ 2>&1 || echo "   .config/gcs/ не существует" && \
    if [ -f .config/gdrive/credentials.json ]; then echo "✅ Google Drive credentials найдены в .config/gdrive/"; else echo "⚠️ Google Drive credentials НЕ найдены в .config/gdrive/"; fi && \
    if [ -f .config/gcs/service-account-key.json ]; then echo "✅ GCS credentials найдены в .config/gcs/"; else echo "⚠️ GCS credentials НЕ найдены в .config/gcs/"; fi && \
    if [ -f "$HOME/.config/gdrive/credentials.json" ]; then echo "✅ Google Drive credentials найдены в $HOME/.config/gdrive/"; else echo "⚠️ Google Drive credentials НЕ найдены в $HOME/.config/gdrive/"; fi

# Открываем порт для Streamlit
EXPOSE 8080

# Настройка переменных окружения для Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Запускаем Streamlit приложение
CMD ["streamlit", "run", "scale/dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]

