#!/bin/bash
# Скрипт для настройки Google Cloud API для проекта

set -e

echo "🔧 Настройка Google Cloud API"
echo "=============================="
echo ""

# Получаем текущий проект
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Ошибка: проект не установлен"
    echo "   Выполните: gcloud config set project scalepathology"
    exit 1
fi

echo "📋 Текущий проект: $PROJECT_ID"
echo ""

# Список необходимых API
APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "containerregistry.googleapis.com"
    "compute.googleapis.com"
    "storage-api.googleapis.com"
    "storage-component.googleapis.com"
)

echo "📦 Включаем необходимые API..."
echo ""

for api in "${APIS[@]}"; do
    echo -n "   ⏳ $api ... "
    if gcloud services enable "$api" --quiet 2>/dev/null; then
        echo "✅ включен"
    else
        echo "⚠️  уже включен или ошибка"
    fi
done

echo ""
echo "✅ Настройка API завершена"
echo ""

# Проверка статуса
echo "📊 Статус включенных API:"
gcloud services list --enabled --filter="name:cloudbuild.googleapis.com OR name:run.googleapis.com OR name:containerregistry.googleapis.com" --format="table(name,title)" 2>/dev/null || echo "   (не удалось получить список)"

echo ""
echo "✅ Готово! API настроены для проекта $PROJECT_ID"

