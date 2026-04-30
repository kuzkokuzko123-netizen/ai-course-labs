# -*- coding: utf-8 -*-
"""
Клиент для тестирования API YandexGPT напрямую.
Используется для проверки доступности и формата запросов.
Лабораторная работа №4
"""

import os
import json
import requests
from dotenv import load_dotenv

# Явно указываем путь к .env, который находится в папке docker рядом с src
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'docker', '.env')
load_dotenv(dotenv_path)

class YandexGPTClient:
    """Простой клиент к YandexGPT API (foundationModels/v1)."""
    def __init__(self, iam_token=None, folder_id=None):
        self.iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        # Отладочный вывод — можно убрать после проверки
        print(f"Token starts with: {self.iam_token[:20] if self.iam_token else 'NOT FOUND'}...")
        print(f"Folder ID: {self.folder_id}")

    def classify_defect(self, measurements: str) -> dict:
        """Отправляет промпт для классификации дефектов бутылок."""
        body = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite/latest",
            "completionOptions": {
                "temperature": 0.2,
                "maxTokens": "150"
            },
            "messages": [
                {
                    "role": "system",
                    "text": (
                        "Ты эксперт по неразрушающему контролю металлических бутылок. "
                        "На основе предоставленных измерений классифицируй изделие как: "
                        "Годен, Брак_трещина, Брак_вмятина, Брак_утонение. "
                        "Ответь в формате JSON: {\"verdict\": \"<категория>\", \"confidence\": <число>, \"reason\": \"<обоснование>\"}"
                    )
                },
                {
                    "role": "user",
                    "text": measurements
                }
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id
        }
        resp = requests.post(self.url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

if __name__ == "__main__":
    client = YandexGPTClient()
    sample = "d=65.1mm, h=169.9mm, wall_min=0.22mm, wall_nom=0.30mm, ect_amplitude=18.7mV, anomaly_flag=1"
    result = client.classify_defect(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))