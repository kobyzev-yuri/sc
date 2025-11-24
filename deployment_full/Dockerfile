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

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p experiments results/predictions results/visualization

# Создаем директории для Google Drive и GCS credentials
RUN mkdir -p /root/.config/gdrive /root/.config/gcs

# Копируем credentials файлы если они есть в проекте
COPY .config/gdrive/ /root/.config/gdrive/
COPY .config/gcs/ /root/.config/gcs/

# Открываем порт для Streamlit
EXPOSE 8080

# Настройка переменных окружения для Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Настройка путей к credentials по умолчанию
# Google Drive OAuth credentials
ENV GOOGLE_DRIVE_CREDENTIALS_PATH=/root/.config/gdrive/credentials.json

# Google Cloud Storage Service Account key (приоритетный путь)
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcs/service-account-key.json

# Запускаем Streamlit приложение
CMD ["streamlit", "run", "scale/dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]

