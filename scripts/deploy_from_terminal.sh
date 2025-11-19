#!/bin/bash
# Универсальный скрипт для развертывания через терминал
# Поддерживает загрузку архива в Cloud Storage и работу через Cloud Shell

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Развертывание Dashboard через терминал${NC}"
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
    gcloud config set project $PROJECT_ID
fi

echo -e "${GREEN}✓${NC} Проект: ${PROJECT_ID}"
echo ""

# Выбор метода
echo -e "${BLUE}Выберите метод развертывания:${NC}"
echo "1) Загрузить архив в Cloud Storage (затем использовать в Cloud Shell)"
echo "2) Развернуть напрямую (если есть Docker и все зависимости)"
echo "3) Показать команды для ручного выполнения"
read -p "Ваш выбор (1-3): " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}📦 Создание архива...${NC}"
        ./scripts/package_for_deployment.sh
        
        ARCHIVE_NAME=$(ls -t dashboard_deployment_*.tar.gz 2>/dev/null | head -1)
        
        if [ -z "$ARCHIVE_NAME" ]; then
            echo -e "${RED}❌ Архив не найден${NC}"
            exit 1
        fi
        
        echo ""
        echo -e "${YELLOW}☁️  Загрузка в Cloud Storage...${NC}"
        ./scripts/upload_to_gcs.sh "$ARCHIVE_NAME"
        
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✅ Архив загружен!${NC}"
        echo ""
        echo -e "${BLUE}📋 Следующие шаги в Cloud Shell:${NC}"
        echo ""
        echo "1. Откройте Cloud Shell:"
        echo "   https://console.cloud.google.com/cloudshell"
        echo ""
        echo "2. Выполните команды:"
        echo ""
        echo "   gsutil cp gs://${PROJECT_ID}-dashboard-deployment/${ARCHIVE_NAME} ."
        echo "   tar -xzf ${ARCHIVE_NAME}"
        echo "   cd dashboard_deployment"
        echo "   chmod +x scripts/*.sh"
        echo "   ./scripts/setup_gcp.sh"
        echo "   ./scripts/deploy_gcp.sh"
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        ;;
        
    2)
        echo ""
        echo -e "${YELLOW}🔨 Прямое развертывание...${NC}"
        
        # Проверка Docker
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}❌ Docker не установлен${NC}"
            exit 1
        fi
        
        ./scripts/setup_gcp.sh
        ./scripts/deploy_gcp.sh
        ;;
        
    3)
        echo ""
        echo -e "${BLUE}📋 Команды для ручного выполнения:${NC}"
        echo ""
        echo -e "${YELLOW}1. Создать архив:${NC}"
        echo "   ./scripts/package_for_deployment.sh"
        echo ""
        echo -e "${YELLOW}2. Загрузить в Cloud Storage:${NC}"
        echo "   gsutil mb -p $PROJECT_ID -l us-central1 gs://${PROJECT_ID}-dashboard-deployment 2>/dev/null || true"
        echo "   gsutil cp dashboard_deployment_*.tar.gz gs://${PROJECT_ID}-dashboard-deployment/"
        echo ""
        echo -e "${YELLOW}3. В Cloud Shell скачать и развернуть:${NC}"
        echo "   gsutil cp gs://${PROJECT_ID}-dashboard-deployment/dashboard_deployment_*.tar.gz ."
        echo "   tar -xzf dashboard_deployment_*.tar.gz"
        echo "   cd dashboard_deployment"
        echo "   chmod +x scripts/*.sh"
        echo "   ./scripts/setup_gcp.sh"
        echo "   ./scripts/deploy_gcp.sh"
        echo ""
        ;;
        
    *)
        echo -e "${RED}❌ Неверный выбор${NC}"
        exit 1
        ;;
esac

