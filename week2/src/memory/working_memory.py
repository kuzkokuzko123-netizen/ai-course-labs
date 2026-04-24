# -*- coding: utf-8 -*
"""
Оперативная память агента
"""
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Краткосрочная память для хранения диалога."""

    def __init__(self, max_messages: int = 100):
        self.messages: List[Dict] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """Добавление сообщения в память."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)

        # Ограничение размера
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        logger.debug(f"Добавлено сообщение от {role}")

    def get_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """Получение сообщений."""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def clear(self):
        """Очистка памяти."""
        self.messages = []
        logger.info("Память очищена")

    def get_conversation_history(self) -> str:
        """Получение истории диалога в текстовом формате."""
        history = []
        for msg in self.messages:
            history.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(history)