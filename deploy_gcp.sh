#!/bin/bash
# Скрипт для быстрого развертывания дашборда в Google Cloud Run

set -e  # Остановка при ошибке

echo "🚀 Развертывание Dashboard в Google Cloud Run"
echo "=============================================="
echo ""

# Проверка наличия gcloud
if ! command -v gcloud &> /dev/null; then
    echo "❌ Ошибка: gcloud CLI не установлен"
    echo "   Установите: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Проверка аутентификации
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  Вы не авторизованы в gcloud"
    echo "   Выполните: gcloud auth login"
    exit 1
fi

# Получение текущего проекта
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Ошибка: проект не установлен"
    echo "   Выполните: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "📋 Текущий проект: $PROJECT_ID"
echo ""

# Параметры развертывания (можно изменить)
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-dashboard}
MEMORY=${MEMORY:-2Gi}
CPU=${CPU:-2}
MAX_INSTANCES=${MAX_INSTANCES:-10}
MIN_INSTANCES=${MIN_INSTANCES:-0}

echo "⚙️  Параметры развертывания:"
echo "   Регион: $REGION"
echo "   Имя сервиса: $SERVICE_NAME"
echo "   Память: $MEMORY"
echo "   CPU: $CPU"
echo "   Макс. инстансов: $MAX_INSTANCES"
echo "   Мин. инстансов: $MIN_INSTANCES"
echo ""

# Включение необходимых API
echo "🔧 Проверка и включение необходимых API..."
gcloud services enable cloudbuild.googleapis.com --quiet 2>/dev/null || true
gcloud services enable run.googleapis.com --quiet 2>/dev/null || true
gcloud services enable containerregistry.googleapis.com --quiet 2>/dev/null || true
echo "✅ API включены"
echo ""

# Проверка наличия Dockerfile
if [ ! -f "Dockerfile" ]; then
    echo "❌ Ошибка: Dockerfile не найден"
    echo "   Убедитесь, что вы находитесь в корневой директории проекта"
    exit 1
fi

# Подтверждение развертывания
read -p "Продолжить развертывание? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Развертывание отменено"
    exit 1
fi

echo ""
echo "📦 Начало развертывания..."
echo ""

# Развертывание в Cloud Run
gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory $MEMORY \
  --cpu $CPU \
  --timeout 300 \
  --max-instances $MAX_INSTANCES \
  --min-instances $MIN_INSTANCES \
  --quiet

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Развертывание завершено успешно!"
echo ""

# Получение URL сервиса
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")

echo "🌐 Ваш дашборд доступен по адресу:"
echo ""
echo "   $SERVICE_URL"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Полезные команды
echo "💡 Полезные команды:"
echo ""
echo "   Просмотр логов:"
echo "   gcloud run services logs read $SERVICE_NAME --region $REGION --follow"
echo ""
echo "   Обновление сервиса:"
echo "   ./deploy_gcp.sh"
echo ""
echo "   Удаление сервиса:"
echo "   gcloud run services delete $SERVICE_NAME --region $REGION"
echo ""

