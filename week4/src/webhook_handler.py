# -*- coding: utf-8 -*-
"""
Обработчик webhook для тестирования workflow
Лабораторная работа №4
Дисциплина: Искусственный интеллект
Автор: [ФИО]
Группа: [НОМЕР ГРУППЫ]
Дата: 2026
"""
import os
import json
import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


class WorkflowClient:
    """
    Клиент для взаимодействия с n8n workflow.

    Атрибуты:
        base_url: URL n8n сервера
        webhook_path: Путь webhook для базового workflow
        secret: Секретный ключ для аутентификации
    """

    def __init__(
            self,
            base_url: str = "http://localhost:5678",
            webhook_path: str = "application",
            secret: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.webhook_path = webhook_path
        self.secret = secret or os.getenv("WEBHOOK_SECRET")
        self.webhook_url = f"{self.base_url}/webhook/{webhook_path}"

        logger.info(f"WorkflowClient инициализирован: {self.webhook_url}")

    def send_application(
            self,
            message: str,
            contact: str,
            priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Отправка заявки в базовый workflow.

        Args:
            message: Текст заявки
            contact: Контактная информация
            priority: Приоритет (low, normal, high)

        Returns:
            Dict: Ответ от workflow
        """
        payload = {
            "message": message,
            "contact": contact,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }

        headers = {
            "Content-Type": "application/json"
        }
        if self.secret:
            headers["X-Webhook-Secret"] = self.secret

        logger.info(f"Отправка заявки: {message[:50]}...")
        try:
            response = requests.post(
                self.webhook_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Workflow выполнил обработку")
            return {
                "success": True,
                "response": result,
                "status_code": response.status_code
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def send_inspection_data(
            self,
            measurements: str,
            serial_number: str
    ) -> Dict[str, Any]:
        """
        Отправка результатов замера бутылки в специализированный workflow
        для неразрушающего контроля.

        Args:
            measurements: Строка с параметрами бутылки
                         (например, "d=65.1mm, h=169.8mm, wall_min=0.22mm, ...")
            serial_number: Серийный номер бутылки

        Returns:
            Dict: Ответ от workflow
        """
        payload = {
            "serial": serial_number,
            "measurements": measurements,
            "timestamp": datetime.now().isoformat()
        }

        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["X-Webhook-Secret"] = self.secret

        # URL webhook'а из специализированного workflow
        webhook_url = f"{self.base_url}/webhook/bottle-inspection"

        logger.info(f"Отправка данных бутылки {serial_number}: {measurements[:50]}...")
        try:
            response = requests.post(
                webhook_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Вердикт: {result}")
            return {
                "success": True,
                "response": result,
                "status_code": response.status_code
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def check_workflow_status(self) -> Dict[str, Any]:
        """Проверка доступности n8n."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return {
                "available": response.status_code == 200,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }


# Точка входа для тестирования специализированного workflow
if __name__ == "__main__":
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №4")
    print("Тестирование workflow неразрушающего контроля бутылок")
    print("=" * 80)

    # Инициализация клиента (путь для базового workflow не важен)
    client = WorkflowClient()

    # Проверка доступности
    print("\nПроверка доступности n8n...")
    status = client.check_workflow_status()
    if status.get("available"):
        print("✅ n8n доступен")
    else:
        print(f"❌ n8n недоступен: {status.get('error')}")
        exit(1)

    # Тестовые данные бутылки с дефектом утонения стенки
    print("\n" + "=" * 80)
    print("ТЕСТОВАЯ ИНСПЕКЦИЯ БУТЫЛКИ")
    print("=" * 80)

    sample_data = (
        "d=65.1mm, h=169.8mm, wall_min=0.22mm, wall_nom=0.30mm, "
        "ect_amplitude=18.7mV, anomaly_flag=1"
    )
    serial = "BTL-2026-0425-001"

    result = client.send_inspection_data(sample_data, serial)

    if result["success"]:
        print("✅ Инспекция выполнена успешно")
        print(f"Статус код: {result['status_code']}")
        if result.get("response"):
            print(f"Ответ workflow: {json.dumps(result['response'], indent=2, ensure_ascii=False)}")
    else:
        print(f"❌ Ошибка: {result['error']}")

    print("=" * 80)
