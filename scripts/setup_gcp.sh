#!/bin/bash
# Скрипт для первоначальной настройки Google Cloud проекта

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}⚙️  Настройка Google Cloud проекта${NC}"
echo "=========================================="
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
fi

echo -e "${GREEN}✓${NC} Используется проект: ${PROJECT_ID}"
gcloud config set project $PROJECT_ID

# Регион
REGION=${GCP_REGION:-"us-central1"}
echo -e "${GREEN}✓${NC} Регион: ${REGION}"

# Включение необходимых API
echo ""
echo -e "${YELLOW}📋 Включение необходимых API...${NC}"

APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "containerregistry.googleapis.com"
    "cloudresourcemanager.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo -n "  Включение $api... "
    if gcloud services enable $api --project=$PROJECT_ID 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC} (возможно уже включен)"
    fi
done

# Настройка Docker для GCR
echo ""
echo -e "${YELLOW}🐳 Настройка Docker для Google Container Registry...${NC}"
gcloud auth configure-docker

# Настройка Cloud Build
echo ""
echo -e "${YELLOW}🔨 Настройка Cloud Build...${NC}"
echo "Проверка прав доступа..."

# Получение текущего пользователя
CURRENT_USER=$(gcloud config get-value account 2>/dev/null)
if [ -z "$CURRENT_USER" ]; then
    echo -e "${YELLOW}⚠️  Необходима авторизация${NC}"
    gcloud auth login
    CURRENT_USER=$(gcloud config get-value account)
fi

echo -e "${GREEN}✓${NC} Авторизован как: ${CURRENT_USER}"

# Проверка billing
echo ""
echo -e "${YELLOW}💳 Проверка billing аккаунта...${NC}"
BILLING_ENABLED=$(gcloud beta billing projects describe $PROJECT_ID --format="value(billingAccountName)" 2>/dev/null || echo "")

if [ -z "$BILLING_ENABLED" ]; then
    echo -e "${YELLOW}⚠️  Billing не подключен${NC}"
    echo "Для использования Cloud Run необходим billing аккаунт"
    echo "Подключите: https://console.cloud.google.com/billing"
else
    echo -e "${GREEN}✓${NC} Billing подключен"
fi

# Создание файла конфигурации
echo ""
echo -e "${YELLOW}📝 Создание файла конфигурации...${NC}"

cat > .gcp_config.env << EOF
# Google Cloud Configuration
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION
SERVICE_NAME=dashboard
IMAGE_NAME=gcr.io/$PROJECT_ID/dashboard:latest
EOF

echo -e "${GREEN}✓${NC} Конфигурация сохранена в .gcp_config.env"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Настройка завершена!${NC}"
echo ""
echo "💡 Следующие шаги:"
echo "   1. Загрузите конфигурацию: source .gcp_config.env"
echo "   2. Соберите и разверните: ./scripts/deploy_gcp.sh"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

