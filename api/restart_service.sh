#!/bin/bash
# Скрипт для быстрого перезапуска сервиса в Google Cloud Run

set -e

echo "🔄 Перезапуск сервиса в Google Cloud Run"
echo "=========================================="
echo ""

# Параметры
PROJECT_ID=${PROJECT_ID:-scalepathology}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-pathology-api}

echo "📋 Параметры:"
echo "   Проект: $PROJECT_ID"
echo "   Регион: $REGION"
echo "   Сервис: $SERVICE_NAME"
echo ""

# Проверка gcloud
if ! command -v gcloud &> /dev/null; then
    echo "❌ Ошибка: gcloud CLI не установлен"
    exit 1
fi

# Установка проекта
gcloud config set project $PROJECT_ID

# Вариант 1: Быстрый перезапуск через обновление конфигурации
# (изменяем переменную окружения, чтобы заставить сервис перезапуститься)
echo "🔄 Принудительный перезапуск сервиса..."
TIMESTAMP=$(date +%s)
gcloud run services update $SERVICE_NAME \
    --region $REGION \
    --update-env-vars "RESTART_TIMESTAMP=$TIMESTAMP" \
    --project=$PROJECT_ID \
    --quiet

echo ""
echo "✅ Сервис перезапущен!"
echo ""
echo "📝 URL сервиса:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)" --project=$PROJECT_ID
echo ""
echo "💡 Примечание: Если изменения не отобразились, возможно данные загружаются из кэша или внешнего источника (GCS/Google Drive)."
echo "   В этом случае выполните полный передеплой: ./deploy_api.sh"



