# Развертывание Dashboard для внешнего доступа

## Вариант 1: ngrok (самый простой для демонстрации) ⭐ Рекомендуется

### Установка ngrok:

```bash
# Linux
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/

# Или через snap
sudo snap install ngrok
```

### Использование:

1. **Запустите Streamlit dashboard:**
   ```bash
   cd /mnt/ai/cnn/sc
   streamlit run scale/dashboard.py --server.port 8501
   ```

2. **В другом терминале запустите ngrok:**
   ```bash
   ngrok http 8501
   ```

3. **Получите публичный URL:**
   - ngrok покажет URL вида: `https://xxxx-xx-xx-xx-xx.ngrok-free.app`
   - Этот URL можно использовать для доступа извне
   - URL будет работать пока запущен ngrok

### Регистрация в ngrok (для постоянного URL):

1. Зарегистрируйтесь на https://ngrok.com
2. Получите authtoken
3. Настройте:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```
4. Используйте зарегистрированный домен (опционально)

**Преимущества:**
- ✅ Быстрая настройка (2 минуты)
- ✅ HTTPS автоматически
- ✅ Не требует изменения firewall
- ✅ Идеально для демонстрации

**Недостатки:**
- ❌ URL меняется при каждом перезапуске (если не зарегистрирован)
- ❌ Бесплатный план имеет ограничения

---

## Вариант 2: Настройка Streamlit для внешнего доступа

### Создайте конфигурационный файл:

```bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
port = 8501
address = "0.0.0.0"  # Слушать на всех интерфейсах
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF
```

### Настройте firewall:

```bash
# Ubuntu/Debian
sudo ufw allow 8501/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

### Запустите dashboard:

```bash
cd /mnt/ai/cnn/sc
streamlit run scale/dashboard.py
```

### Получите IP адрес сервера:

```bash
# Внутренний IP
hostname -I

# Или внешний IP (если есть)
curl ifconfig.me
```

### Доступ:

- Из локальной сети: `http://SERVER_IP:8501`
- Из интернета: `http://EXTERNAL_IP:8501` (если порт открыт)

**Преимущества:**
- ✅ Прямой доступ без посредников
- ✅ Можно использовать свой домен

**Недостатки:**
- ❌ Требует настройки firewall
- ❌ Нет HTTPS (нужен reverse proxy)
- ❌ Нужен статический IP или домен

---

## Вариант 3: SSH туннель (для безопасного доступа)

### На сервере:

```bash
# Запустите dashboard на localhost
streamlit run scale/dashboard.py --server.port 8501 --server.address localhost
```

### На клиентской машине:

```bash
# Создайте SSH туннель
ssh -L 8501:localhost:8501 user@server_ip

# Или в фоне
ssh -N -L 8501:localhost:8501 user@server_ip
```

### Доступ:

Откройте в браузере: `http://localhost:8501`

**Преимущества:**
- ✅ Безопасно (шифрование через SSH)
- ✅ Не требует открытия портов
- ✅ Работает через NAT

**Недостатки:**
- ❌ Требует SSH доступ к серверу
- ❌ Каждый клиент должен создавать туннель

---

## Вариант 4: Streamlit Cloud (бесплатный хостинг)

### Подготовка:

1. **Создайте файл `requirements.txt`** (уже есть):
   ```bash
   # Проверьте, что все зависимости указаны
   cat requirements.txt
   ```

2. **Создайте `.streamlit/config.toml`** (опционально):
   ```bash
   mkdir -p .streamlit
   cat > .streamlit/config.toml << EOF
   [server]
   port = 8501
   EOF
   ```

3. **Закоммитьте в GitHub:**
   ```bash
   git add requirements.txt .streamlit/
   git commit -m "Подготовка для Streamlit Cloud"
   git push
   ```

### Развертывание:

1. Зайдите на https://streamlit.io/cloud
2. Войдите через GitHub
3. Нажмите "New app"
4. Выберите репозиторий и ветку
5. Укажите путь к файлу: `scale/dashboard.py`
6. Нажмите "Deploy"

**Преимущества:**
- ✅ Бесплатно
- ✅ HTTPS автоматически
- ✅ Публичный URL
- ✅ Автоматическое обновление при push

**Недостатки:**
- ❌ Ограничения на ресурсы (CPU/RAM)
- ❌ Требует публичный репозиторий (или платный план)

---

## Вариант 5: Docker + облачный сервис

### Создайте Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "scale/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Создайте .dockerignore:

```
__pycache__/
*.pyc
.env
.git/
experiments/
results/visualization/
```

### Развертывание:

**Railway (рекомендуется):**
1. Зайдите на https://railway.app
2. Создайте проект из GitHub
3. Добавьте Dockerfile
4. Railway автоматически развернет

**Heroku:**
```bash
heroku create your-app-name
heroku container:push web
heroku container:release web
```

**Преимущества:**
- ✅ Изолированная среда
- ✅ Легко масштабировать
- ✅ Автоматическое развертывание

**Недостатки:**
- ❌ Требует Docker знания
- ❌ Может быть платно (зависит от сервиса)

---

## Рекомендации по безопасности

### Для демонстрации (ngrok):
- ✅ Используйте ngrok - самый простой вариант
- ✅ Временный доступ, безопасно для демо

### Для продакшена:
- ✅ Используйте HTTPS (через nginx reverse proxy)
- ✅ Настройте аутентификацию (Streamlit поддерживает)
- ✅ Ограничьте доступ по IP (если возможно)
- ✅ Используйте переменные окружения для секретов

### Настройка аутентификации в Streamlit:

Создайте `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[server.enableXsrfProtection]
enabled = true

# Опционально: базовая аутентификация через nginx
```

---

## Быстрый старт для демонстрации

```bash
# 1. Установите ngrok
sudo snap install ngrok

# 2. Запустите dashboard
cd /mnt/ai/cnn/sc
streamlit run scale/dashboard.py &

# 3. Запустите ngrok
ngrok http 8501

# 4. Скопируйте URL из ngrok (например: https://xxxx.ngrok-free.app)
# 5. Откройте URL в браузере
```

---

## Troubleshooting

### Порт уже занят:
```bash
# Найдите процесс
lsof -i :8501

# Убейте процесс
kill -9 PID
```

### Dashboard не открывается:
```bash
# Проверьте, что Streamlit запущен
ps aux | grep streamlit

# Проверьте логи
# Streamlit показывает URL в консоли
```

### ngrok не работает:
```bash
# Проверьте, что dashboard доступен локально
curl http://localhost:8501

# Проверьте версию ngrok
ngrok version
```

---

## Сравнение вариантов

| Вариант | Сложность | Стоимость | Время настройки | Для демо | Для продакшена |
|---------|-----------|-----------|-----------------|----------|----------------|
| ngrok | ⭐ Легко | Бесплатно | 2 мин | ✅ Да | ❌ Нет |
| Прямой доступ | ⭐⭐ Средне | Бесплатно | 10 мин | ✅ Да | ⚠️ С HTTPS |
| SSH туннель | ⭐⭐ Средне | Бесплатно | 5 мин | ✅ Да | ✅ Да |
| Streamlit Cloud | ⭐ Легко | Бесплатно | 15 мин | ✅ Да | ⚠️ Ограничения |
| Docker + Cloud | ⭐⭐⭐ Сложно | Платно | 30 мин | ✅ Да | ✅ Да |

**Для демонстрации: используйте ngrok** ⭐




