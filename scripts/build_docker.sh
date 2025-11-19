#!/bin/bash
# Скрипт для локальной сборки Docker образа Dashboard

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🐳 Сборка Docker образа Dashboard${NC}"
echo "======================================"
echo ""

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен"
    echo "Установите: https://docs.docker.com/get-docker/"
    exit 1
fi

# Имя образа
IMAGE_NAME=${IMAGE_NAME:-"dashboard:latest"}

echo -e "${YELLOW}📦 Сборка образа: ${IMAGE_NAME}${NC}"
docker build -f Dockerfile.dashboard -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Образ успешно собран!${NC}"
    echo ""
    echo "💡 Полезные команды:"
    echo "   Запуск: docker run -p 8080:8080 $IMAGE_NAME"
    echo "   Или используйте: docker-compose up"
    echo "   Просмотр образов: docker images"
else
    echo "❌ Ошибка при сборке образа"
    exit 1
fi

