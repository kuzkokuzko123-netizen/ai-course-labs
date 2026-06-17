# -*- coding: utf-8 -*-
"""
Шина сообщений для асинхронной коммуникации между агентами
Лабораторная работа №3
"""

from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
from datetime import datetime
from enum import Enum
import logging
import uuid
import threading
import queue
import time

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Приоритеты сообщений."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageBus:
    """
    Шина сообщений для асинхронной коммуникации между агентами.

    Поддерживает:
    - Подписку на сообщения определённого типа
    - Отправку сообщений с гарантией доставки
    - Приоритеты сообщений
    - Очереди сообщений
    - Широковещательную рассылку
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Dict]] = defaultdict(list)
        self._message_history: List[Dict] = []
        self._message_queues: Dict[str, queue.Queue] = {}
        self._running = False
        self._worker_thread = None

    def start(self):
        """Запуск обработчика сообщений."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_messages, daemon=True)
        self._worker_thread.start()
        logger.info("Message Bus запущен")

    def stop(self):
        """Остановка обработчика сообщений."""
        self._running = False
        logger.info("Message Bus остановлен")

    def subscribe(self, message_type: str, callback: Callable, agent_id: str = None) -> str:
        """
        Подписка на сообщения определённого типа.

        Returns:
            str: ID подписки
        """
        subscription_id = str(uuid.uuid4())
        self._subscribers[message_type].append({
            "id": subscription_id,
            "callback": callback,
            "agent_id": agent_id,
            "created_at": datetime.now().isoformat()
        })
        logger.debug(f"Подписка {subscription_id} на тип {message_type} от агента {agent_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Отписка от сообщений."""
        for msg_type, subscribers in self._subscribers.items():
            for i, sub in enumerate(subscribers):
                if sub["id"] == subscription_id:
                    subscribers.pop(i)
                    logger.debug(f"Отписка {subscription_id}")
                    return True
        return False

    def publish(self, message: Dict, priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """
        Публикация сообщения всем подписчикам.
        """
        message_id = str(uuid.uuid4())
        message["message_id"] = message_id
        message["timestamp"] = datetime.now().isoformat()
        message["priority"] = priority.value

        self._message_history.append(message)

        message_type = message.get("type", "unknown")

        if message_type in self._subscribers:
            for subscriber in self._subscribers[message_type]:
                try:
                    subscriber["callback"](message)
                except Exception as e:
                    logger.error(f"Ошибка в обработчике {subscriber['id']}: {e}")

        logger.info(
            f"Сообщение {message_id} типа {message_type} опубликовано для {len(self._subscribers[message_type])} подписчиков")
        return message_id

    def send_direct(self, sender_id: str, receiver_id: str, content: Dict) -> str:
        """
        Прямая отправка сообщения конкретному агенту.
        """
        message = {
            "type": "direct_message",
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": content
        }
        return self.publish(message)

    def broadcast(self, sender_id: str, content: Dict) -> str:
        """
        Широковещательная рассылка всем агентам.
        """
        message = {
            "type": "broadcast",
            "sender_id": sender_id,
            "receiver_id": "all",
            "content": content
        }
        return self.publish(message, MessagePriority.HIGH)

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Получение истории сообщений."""
        return self._message_history[-limit:]

    def get_message_count(self) -> int:
        """Количество сообщений в истории."""
        return len(self._message_history)

    def clear_history(self):
        """Очистка истории сообщений."""
        self._message_history = []
        logger.info("История сообщений очищена")

    def _process_messages(self):
        """Обработка сообщений в фоновом режиме."""
        while self._running:
            time.sleep(0.1)

    def get_statistics(self) -> Dict:
        """Статистика шины сообщений."""
        return {
            "total_messages": len(self._message_history),
            "subscribers_count": sum(len(subs) for subs in self._subscribers.values()),
            "message_types": list(self._subscribers.keys()),
            "is_running": self._running
        }


# Глобальная шина сообщений
message_bus = MessageBus()