#!/bin/bash
# Простой скрипт для поиска и подключения к виртуальной машине

set -e

VM_IP="35.225.95.250"

echo "🔍 Поиск виртуальной машины с IP: $VM_IP"
echo ""

# Находим виртуальную машину
VM_INFO=$(gcloud compute instances list --filter="EXTERNAL_IP=$VM_IP" --format="value(name,zone)" 2>/dev/null)

if [ -z "$VM_INFO" ]; then
    echo "❌ Виртуальная машина не найдена"
    echo ""
    echo "💡 Список всех виртуальных машин:"
    gcloud compute instances list
    exit 1
fi

VM_NAME=$(echo "$VM_INFO" | cut -d$'\t' -f1)
VM_ZONE=$(echo "$VM_INFO" | cut -d$'\t' -f2)

echo "✅ Найдена виртуальная машина:"
echo "   Имя: $VM_NAME"
echo "   Зона: $VM_ZONE"
echo ""

echo "🔌 Подключение через gcloud compute ssh..."
echo ""

# Подключаемся
gcloud compute ssh "ai8049520@$VM_NAME" --zone="$VM_ZONE"

