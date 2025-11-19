#!/bin/bash
# Скрипт для развертывания Dashboard в Google Cloud Run

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Развертывание Dashboard в Google Cloud Run${NC}"
echo "=========================================="
echo ""

# Проверка наличия gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI не установлен${NC}"
    echo "Установите: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker не установлен${NC}"
    echo "Установите: https://docs.docker.com/get-docker/"
    exit 1
fi

# Получение проекта GCP
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}

if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}⚠️  GCP проект не установлен${NC}"
    read -p "Введите GCP Project ID: " PROJECT_ID
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}❌ Project ID обязателен${NC}"
        exit 1
    fi
    gcloud config set project $PROJECT_ID
fi

echo -e "${GREEN}✓${NC} Используется проект: ${PROJECT_ID}"

# Регион (можно изменить)
REGION=${GCP_REGION:-"us-central1"}
echo -e "${GREEN}✓${NC} Регион: ${REGION}"

# Имя сервиса
SERVICE_NAME=${SERVICE_NAME:-"dashboard"}
echo -e "${GREEN}✓${NC} Имя сервиса: ${SERVICE_NAME}"

# Включение необходимых API
echo ""
echo -e "${YELLOW}📋 Проверка и включение необходимых API...${NC}"
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID

# Настройка Docker для GCR
echo ""
echo -e "${YELLOW}🐳 Настройка Docker для Google Container Registry...${NC}"
gcloud auth configure-docker

# Сборка Docker образа
echo ""
echo -e "${YELLOW}🔨 Сборка Docker образа...${NC}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

docker build -f Dockerfile.dashboard -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Ошибка при сборке Docker образа${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker образ собран"

# Отправка образа в GCR
echo ""
echo -e "${YELLOW}📤 Отправка образа в Google Container Registry...${NC}"
docker push $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Ошибка при отправке образа${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Образ отправлен в GCR"

# Развертывание в Cloud Run
echo ""
echo -e "${YELLOW}🚀 Развертывание в Cloud Run...${NC}"

gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Ошибка при развертывании${NC}"
    exit 1
fi

# Получение URL сервиса
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Dashboard успешно развернут!${NC}"
echo ""
echo -e "${GREEN}🌐 URL: ${SERVICE_URL}${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "💡 Полезные команды:"
echo "   Просмотр логов: gcloud run services logs read $SERVICE_NAME --region $REGION"
echo "   Обновление: ./scripts/deploy_gcp.sh"
echo "   Удаление: gcloud run services delete $SERVICE_NAME --region $REGION"

