# -*- coding: utf-8 -*-
"""
Клиент для работы с Yandex GPT API
"""

import os
import json
import logging
import requests
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class YandexGPTClient:
    """
    Клиент для работы с Yandex GPT через официальное API.
    """

    def __init__(self):
        self.iam_token = os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.temperature = float(os.getenv("TEMPERATURE", 0.7))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 2000))

        # Правильный URL
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        self._validate_config()

    def _validate_config(self):
        """Проверка конфигурации."""
        if not self.iam_token or self.iam_token == "your_iam_token_here":
            logger.warning("YANDEX_IAM_TOKEN не настроен")
        if not self.folder_id or self.folder_id == "your_folder_id_here":
            logger.warning("YANDEX_FOLDER_ID не настроен")

    def _get_headers(self) -> Dict[str, str]:
        """Получение заголовков для запроса."""
        return {
            "Authorization": f"Bearer {self.iam_token}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Отправка запроса к Yandex GPT и получение ответа.
        """
        if not self.is_available():
            error_msg = "Yandex GPT не настроен. Проверьте .env файл"
            logger.error(error_msg)
            return error_msg

        # Формируем массив сообщений
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "text": system_prompt
            })

        messages.append({
            "role": "user",
            "text": prompt
        })

        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            },
            "messages": messages
        }

        try:
            response = requests.post(
                self.url,
                headers=self._get_headers(),
                json=data,
                timeout=30
            )

            logger.info(f"Статус ответа Yandex GPT: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                # 🔧 ИСПРАВЛЕННЫЙ ПАРСИНГ ОТВЕТА
                # Формат ответа: {'result': {'alternatives': [{'message': {'text': ...}}]}}
                if "result" in result:
                    result_data = result["result"]

                    if "alternatives" in result_data and len(result_data["alternatives"]) > 0:
                        alternative = result_data["alternatives"][0]

                        if "message" in alternative and "text" in alternative["message"]:
                            generated_text = alternative["message"]["text"]
                            logger.info(f"✅ Yandex GPT успешно ответил ({len(generated_text)} символов)")
                            return generated_text

                        elif "text" in alternative:  # Альтернативный формат
                            generated_text = alternative["text"]
                            logger.info(f"✅ Yandex GPT ответил (альтернативный формат)")
                            return generated_text

                # Если структура другая, пробуем альтернативный парсинг
                elif "alternatives" in result and len(result["alternatives"]) > 0:
                    generated_text = result["alternatives"][0].get("message", {}).get("text", "")
                    if generated_text:
                        logger.info(f"✅ Yandex GPT ответил (альтернативная структура)")
                        return generated_text

                # Если ничего не нашли, выводим ошибку
                logger.error(f"Неожиданный формат ответа: {json.dumps(result, ensure_ascii=False)[:500]}")
                return f"Ошибка парсинга ответа API. Получен формат: {list(result.keys())}"

            elif response.status_code == 401:
                logger.error("Ошибка 401: Неверный IAM токен")
                return "Ошибка авторизации: обновите IAM токен командой 'yc iam create-token'"

            elif response.status_code == 403:
                logger.error("Ошибка 403: Нет доступа")
                return "Ошибка доступа: проверьте folder_id и права сервисного аккаунта"

            elif response.status_code == 429:
                logger.error("Ошибка 429: Превышен лимит")
                return "Превышен лимит запросов. Попробуйте позже."

            else:
                logger.error(f"Ошибка {response.status_code}: {response.text}")
                return f"Ошибка API: {response.status_code}"

        except requests.exceptions.Timeout:
            logger.error("Таймаут запроса")
            return "Таймаут соединения с API"

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return f"Ошибка: {e}"

    def is_available(self) -> bool:
        """Проверка настройки клиента."""
        return bool(
            self.iam_token and
            self.folder_id and
            self.iam_token != "your_iam_token_here" and
            self.folder_id != "your_folder_id_here"
        )


# Глобальный экземпляр
yandex_gpt = YandexGPTClient()

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("ТЕСТ YANDEX GPT API")
    print("=" * 60)

    if yandex_gpt.is_available():
        print("✅ Клиент настроен")

        print("\n🔄 Тест 1: Простой запрос")
        response1 = yandex_gpt.generate("Напиши одно предложение о неразрушающем контроле металлических бутылок.")
        print(f"\n📝 Ответ:\n{response1}")

        print("\n" + "-" * 60)
        print("\n🔄 Тест 2: Запрос с системным промтом")
        response2 = yandex_gpt.generate(
            prompt="Перечисли 3 ключевых факта о вихретоковом контроле.",
            system_prompt="Ты - эксперт по неразрушающему контролю. Отвечай кратко, по пунктам."
        )
        print(f"\n📝 Ответ:\n{response2}")

    else:
        print("❌ Клиент не настроен")
        print("\nНастройка:")
        print("1. yc iam create-token")
        print("2. Обновить .env файл")