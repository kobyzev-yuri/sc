#!/bin/bash
# Скрипт для развертывания FastAPI сервера в Google Cloud Run

set -e

echo "🚀 Развертывание FastAPI сервера в Google Cloud Run"
echo "=================================================="
echo ""

# Параметры
PROJECT_ID=${PROJECT_ID:-scalepathology}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-pathology-api}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "📋 Параметры:"
echo "   Проект: $PROJECT_ID"
echo "   Регион: $REGION"
echo "   Сервис: $SERVICE_NAME"
echo "   Образ: $IMAGE_NAME"
echo ""

# Проверка gcloud
if ! command -v gcloud &> /dev/null; then
    echo "❌ Ошибка: gcloud CLI не установлен"
    exit 1
fi

# Установка проекта
gcloud config set project $PROJECT_ID

# Сборка Docker образа
echo "🔨 Сборка Docker образа..."
gcloud builds submit --tag $IMAGE_NAME --project=$PROJECT_ID

# Развертывание в Cloud Run
echo "🚀 Развертывание в Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --min-instances 0 \
    --port 8080 \
    --timeout 300 \
    --project=$PROJECT_ID

echo ""
echo "✅ Развертывание завершено!"
echo ""
echo "📝 URL сервиса:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)" --project=$PROJECT_ID

