# Dockerfile для FastAPI сервера
FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements для API
COPY api/requirements.txt .

# Устанавливаем зависимости API
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (для импорта scale модулей)
COPY . .

# Создаем необходимые директории
RUN mkdir -p experiments results/predictions results/inference results/visualization
RUN mkdir -p .config/gdrive .config/gcs

# Открываем порт для FastAPI
EXPOSE 8080

# Переменные окружения
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/.config/gcs/service-account-key.json"

# Запускаем FastAPI сервер с увеличенными таймаутами для длительных операций
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300", "--timeout-graceful-shutdown", "300"]

