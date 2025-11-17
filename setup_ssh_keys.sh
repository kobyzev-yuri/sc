#!/bin/bash
# Скрипт для настройки SSH ключей для доступа к виртуальной машине

set -e

echo "🔑 Настройка SSH ключей для виртуальной машины"
echo "=============================================="
echo ""

# Проверка наличия ssh-keygen
if ! command -v ssh-keygen &> /dev/null; then
    echo "❌ Ошибка: ssh-keygen не установлен"
    echo "   Установите: apt-get update && apt-get install -y openssh-client"
    exit 1
fi

# Директория для SSH ключей
SSH_DIR="$HOME/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Имя ключа
KEY_NAME="gcp_vm_key"
PRIVATE_KEY="$SSH_DIR/$KEY_NAME"
PUBLIC_KEY="$SSH_DIR/$KEY_NAME.pub"

echo "📋 Настройка SSH ключа: $KEY_NAME"
echo ""

# Проверка существования ключа
if [ -f "$PRIVATE_KEY" ]; then
    echo "⚠️  SSH ключ уже существует: $PRIVATE_KEY"
    read -p "   Пересоздать ключ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Удаление старого ключа..."
        rm -f "$PRIVATE_KEY" "$PUBLIC_KEY"
    else
        echo "✅ Используем существующий ключ"
        KEY_EXISTS=true
    fi
else
    KEY_EXISTS=false
fi

# Генерация нового ключа если нужно
if [ "$KEY_EXISTS" = false ]; then
    echo "🔧 Генерация нового SSH ключа..."
    echo ""
    
    # Запрашиваем email для ключа (опционально)
    echo "Введите email для ключа (опционально, можно оставить пустым):"
    read -r KEY_EMAIL
    
    if [ -z "$KEY_EMAIL" ]; then
        ssh-keygen -t ed25519 -f "$PRIVATE_KEY" -N "" -C "gcp-vm-key"
    else
        ssh-keygen -t ed25519 -f "$PRIVATE_KEY" -N "" -C "$KEY_EMAIL"
    fi
    
    echo ""
    echo "✅ SSH ключ создан"
fi

# Показываем публичный ключ
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 ПУБЛИЧНЫЙ КЛЮЧ (скопируйте его):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat "$PUBLIC_KEY"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Определяем имя пользователя (не root)
if [ "$USER" = "root" ]; then
    DEFAULT_USER="ai8049520"
else
    DEFAULT_USER="$USER"
fi

# Инструкции по добавлению ключа в виртуальную машину
echo "💡 Следующие шаги:"
echo ""
echo "1️⃣  Скопируйте публичный ключ выше"
echo ""
echo "2️⃣  Добавьте ключ в метаданные проекта Google Cloud:"
echo ""
echo "   gcloud compute project-info add-metadata \\"
echo "     --metadata-from-file ssh-keys=<(echo \"$DEFAULT_USER:$(cat $PUBLIC_KEY)\")"
echo ""

# Создаем скрипт для автоматического добавления ключа
AUTO_ADD_SCRIPT="$SSH_DIR/add_key_to_gcp.sh"
cat > "$AUTO_ADD_SCRIPT" << EOF
#!/bin/bash
# Автоматическое добавление SSH ключа в метаданные проекта

USERNAME=\${1:-$DEFAULT_USER}
echo "Добавление SSH ключа для пользователя: \$USERNAME"
gcloud compute project-info add-metadata \\
  --metadata-from-file ssh-keys=<(echo "\$USERNAME:$(cat $PUBLIC_KEY)")
echo "✅ Ключ добавлен в метаданные проекта"
EOF

chmod +x "$AUTO_ADD_SCRIPT"

echo "📝 Создан скрипт для автоматического добавления:"
echo "   $AUTO_ADD_SCRIPT"
echo ""
echo "   Использование:"
echo "   $AUTO_ADD_SCRIPT [USERNAME]"
echo ""

# Настройка SSH config для удобства
SSH_CONFIG="$SSH_DIR/config"
if [ ! -f "$SSH_CONFIG" ]; then
    touch "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"
fi

echo "💾 Настройка SSH config..."
echo ""

# Запрашиваем информацию о виртуальной машине
echo "Для настройки SSH config введите информацию о виртуальной машине:"
echo "   (можно пропустить и настроить позже)"
echo ""
read -p "Имя виртуальной машины (или Enter для пропуска): " VM_NAME
read -p "Зона (например, us-central1-a, или Enter для пропуска): " VM_ZONE

if [ -n "$VM_NAME" ] && [ -n "$VM_ZONE" ]; then
    # Получаем IP виртуальной машины
    VM_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$VM_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "")
    
    if [ -n "$VM_IP" ]; then
        # Добавляем запись в SSH config
        cat >> "$SSH_CONFIG" << EOF

# Google Cloud VM: $VM_NAME
Host gcp-vm
    HostName $VM_IP
    User $DEFAULT_USER
    IdentityFile $PRIVATE_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
        
        echo "✅ SSH config обновлен"
        echo ""
        echo "   Теперь можно подключаться:"
        echo "   ssh gcp-vm"
    else
        echo "⚠️  Не удалось получить IP виртуальной машины"
    fi
fi

echo ""
echo "✅ Настройка SSH ключей завершена!"
echo ""
