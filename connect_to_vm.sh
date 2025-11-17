#!/bin/bash
# Скрипт для настройки и подключения к виртуальной машине

set -e

echo "🔗 Настройка подключения к виртуальной машине"
echo "=============================================="
echo ""

# Параметры виртуальной машины
VM_IP="35.225.95.250"
VM_USER="ai8049520"

echo "📋 Параметры:"
echo "   IP адрес: $VM_IP"
echo "   Пользователь: $VM_USER"
echo ""

# Находим имя виртуальной машины по IP
echo "🔍 Поиск виртуальной машины по IP..."
VM_NAME=$(gcloud compute instances list --filter="EXTERNAL_IP=$VM_IP" --format="value(name)" 2>/dev/null | head -1)
VM_ZONE=$(gcloud compute instances list --filter="EXTERNAL_IP=$VM_IP" --format="value(zone)" 2>/dev/null | head -1)

if [ -n "$VM_NAME" ] && [ -n "$VM_ZONE" ]; then
    echo "✅ Найдена виртуальная машина: $VM_NAME (зона: $VM_ZONE)"
    echo ""
    
    # Используем gcloud compute ssh (рекомендуемый способ)
    echo "💡 Использование gcloud compute ssh (автоматическое управление ключами)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Подключение к виртуальной машине..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Подключаемся через gcloud
    gcloud compute ssh "$VM_USER@$VM_NAME" --zone="$VM_ZONE"
    
else
    echo "⚠️  Не удалось найти виртуальную машину по IP"
    echo "   Используем прямое подключение через SSH"
    echo ""
    
    # Директория для SSH
    SSH_DIR="$HOME/.ssh"
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
    
    # Проверяем наличие ключей Google Cloud
    GCE_KEY="$HOME/.ssh/google_compute_engine"
    if [ ! -f "$GCE_KEY" ]; then
        echo "🔧 Генерация ключа Google Compute Engine..."
        ssh-keygen -t rsa -f "$GCE_KEY" -N "" -C "$VM_USER" 2>/dev/null || true
        echo "✅ Ключ создан"
        echo ""
    fi
    
    # Настройка SSH config
    SSH_CONFIG="$SSH_DIR/config"
    if [ ! -f "$SSH_CONFIG" ]; then
        touch "$SSH_CONFIG"
        chmod 600 "$SSH_CONFIG"
    fi
    
    # Удаляем старую запись если есть
    if grep -q "Host gcp-vm" "$SSH_CONFIG" 2>/dev/null; then
        sed -i '/^Host gcp-vm$/,/^$/d' "$SSH_CONFIG"
    fi
    
    # Добавляем запись в SSH config
    cat >> "$SSH_CONFIG" << EOF

# Google Cloud VM - $VM_IP
Host gcp-vm
    HostName $VM_IP
    User $VM_USER
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
    IdentityFile $GCE_KEY
EOF
    
    echo "✅ SSH config обновлен"
    echo ""
    echo "💡 Попытка подключения..."
    echo ""
    
    # Пробуем подключиться
    ssh -i "$GCE_KEY" "$VM_USER@$VM_IP" || {
        echo ""
        echo "❌ Не удалось подключиться"
        echo ""
        echo "💡 Решения:"
        echo "   1. Найдите имя виртуальной машины:"
        echo "      gcloud compute instances list"
        echo ""
        echo "   2. Используйте gcloud compute ssh:"
        echo "      gcloud compute ssh VM_NAME --zone=ZONE"
        echo ""
        echo "   3. Или добавьте ключ вручную на виртуальную машину"
    }
fi
