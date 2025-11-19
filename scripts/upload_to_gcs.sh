#!/bin/bash
# Скрипт для загрузки архива в Google Cloud Storage

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}☁️  Загрузка архива в Google Cloud Storage${NC}"
echo "=============================================="
echo ""

# Проверка наличия gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI не установлен${NC}"
    echo "Установите: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Получение проекта
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

# Поиск архива
ARCHIVE_NAME=${1:-$(ls -t dashboard_deployment_*.tar.gz 2>/dev/null | head -1)}

if [ -z "$ARCHIVE_NAME" ] || [ ! -f "$ARCHIVE_NAME" ]; then
    echo -e "${YELLOW}⚠️  Архив не найден. Создаю новый...${NC}"
    ./scripts/package_for_deployment.sh
    ARCHIVE_NAME=$(ls -t dashboard_deployment_*.tar.gz | head -1)
fi

echo -e "${GREEN}✓${NC} Архив: $ARCHIVE_NAME"

# Имя bucket (создадим если нужно)
BUCKET_NAME="${PROJECT_ID}-dashboard-deployment"
echo -e "${GREEN}✓${NC} Bucket: $BUCKET_NAME"

# Проверка и создание bucket
echo ""
echo -e "${YELLOW}📦 Проверка bucket...${NC}"
if ! gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    echo "Создание bucket..."
    gsutil mb -p $PROJECT_ID -l us-central1 "gs://$BUCKET_NAME" || true
    echo -e "${GREEN}✓${NC} Bucket создан"
else
    echo -e "${GREEN}✓${NC} Bucket существует"
fi

# Загрузка архива
echo ""
echo -e "${YELLOW}📤 Загрузка архива в Cloud Storage...${NC}"
gsutil cp "$ARCHIVE_NAME" "gs://$BUCKET_NAME/"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ Архив успешно загружен!${NC}"
    echo ""
    echo "📦 URL архива: gs://$BUCKET_NAME/$ARCHIVE_NAME"
    echo ""
    echo "💡 Следующие шаги:"
    echo ""
    echo "1. Откройте Cloud Shell:"
    echo "   https://console.cloud.google.com/cloudshell"
    echo ""
    echo "2. Скачайте архив:"
    echo "   gsutil cp gs://$BUCKET_NAME/$ARCHIVE_NAME ."
    echo ""
    echo "3. Распакуйте:"
    echo "   tar -xzf $ARCHIVE_NAME"
    echo ""
    echo "4. Разверните:"
    echo "   cd dashboard_deployment"
    echo "   ./scripts/setup_gcp.sh"
    echo "   ./scripts/deploy_gcp.sh"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${RED}❌ Ошибка при загрузке${NC}"
    exit 1
fi

