# -*- coding: utf-8 -*-
"""
Тестирование базового workflow (обработка заявок с AI-классификацией)
Лабораторная работа №4
"""

import requests
import json
from datetime import datetime

# Настройки
N8N_URL = "http://localhost:5678"
WEBHOOK_PATH = "application"
WEBHOOK_SECRET = None  # Укажите ваш секретный ключ, если задавали в .env

webhook_url = f"{N8N_URL}/webhook/{WEBHOOK_PATH}"

# Тестовая заявка, которая должна попасть в категорию "техническая_поддержка"
test_payload = {
    "message": "Не работает вход в систему, ошибка 403",
    "contact": "petrov@example.com",
    "priority": "high",
    "timestamp": datetime.now().isoformat()
}

headers = {
    "Content-Type": "application/json"
}
if WEBHOOK_SECRET:
    headers["X-Webhook-Secret"] = WEBHOOK_SECRET

print("Отправка тестовой заявки...")
print(f"URL: {webhook_url}")
print(f"Payload: {json.dumps(test_payload, indent=2, ensure_ascii=False)}")

try:
    response = requests.post(webhook_url, headers=headers, json=test_payload, timeout=30)
    print(f"\nСтатус ответа: {response.status_code}")
    if response.status_code == 200:
        print("✅ Workflow успешно выполнен!")
        print("Ответ:", json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"⚠️ Workflow вернул статус {response.status_code}")
        print("Тело ответа:", response.text)
except requests.exceptions.ConnectionError:
    print("❌ Не удалось подключиться к n8n. Проверьте, что Docker-контейнер запущен.")
except Exception as e:
    print(f"❌ Ошибка: {e}")
