#!/bin/bash
# Добавление существующего SSH ключа ноутбука в GCP

SSH_DIR="$HOME/.ssh"
KEY_FILE=""

# Ищем существующий ключ
for key in id_ed25519 id_rsa id_ecdsa; do
    if [ -f "$SSH_DIR/$key.pub" ]; then
        KEY_FILE="$SSH_DIR/$key.pub"
        break
    fi
done

if [ -z "$KEY_FILE" ]; then
    echo "Ключ не найден. Укажите путь к публичному ключу:"
    read -r KEY_FILE
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "Ошибка: файл не найден: $KEY_FILE"
    exit 1
fi

echo "Ключ: $KEY_FILE"
echo ""
cat "$KEY_FILE"
echo ""

gcloud compute project-info add-metadata \
  --metadata-from-file ssh-keys=<(echo "ai8049520:$(cat $KEY_FILE)")

echo "✅ Ключ добавлен в GCP"

