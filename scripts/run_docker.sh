#!/bin/bash
# Скрипт для запуска Dashboard в Docker

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Запуск Dashboard в Docker${NC}"
echo "=================================="
echo ""

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен"
    exit 1
fi

# Проверка наличия образа
IMAGE_NAME=${IMAGE_NAME:-"dashboard:latest"}

if ! docker images | grep -q "^dashboard"; then
    echo -e "${YELLOW}⚠️  Образ не найден. Собираю...${NC}"
    ./scripts/build_docker.sh
fi

# Порт
PORT=${PORT:-8080}

echo -e "${YELLOW}📦 Запуск контейнера на порту ${PORT}...${NC}"

docker run -d \
    --name dashboard-app \
    -p ${PORT}:8080 \
    -v "$(pwd)/results:/app/results:ro" \
    -v "$(pwd)/data:/app/data:ro" \
    --restart unless-stopped \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Dashboard запущен!${NC}"
    echo ""
    echo "🌐 Доступен по адресу: http://localhost:${PORT}"
    echo ""
    echo "💡 Полезные команды:"
    echo "   Логи: docker logs -f dashboard-app"
    echo "   Остановка: docker stop dashboard-app"
    echo "   Удаление: docker rm dashboard-app"
else
    echo "❌ Ошибка при запуске контейнера"
    exit 1
fi

