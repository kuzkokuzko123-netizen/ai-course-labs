# -*- coding: utf-8 -*-
"""
Тест коммуникации между агентами
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)

from agents.base_agent import AgentConfig
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent


def test_direct_communication():
    """Прямой тест передачи сообщений между агентами."""

    print("\n" + "=" * 60)
    print("ТЕСТ ПРЯМОЙ КОММУНИКАЦИИ МЕЖДУ АГЕНТАМИ")
    print("=" * 60)

    # Создание агентов
    researcher = ResearcherAgent()
    analyst = AnalystAgent()

    print(f"\n1️⃣ Созданы агенты:")
    print(f"   - {researcher.config.role} (ID: {researcher.state.agent_id})")
    print(f"   - {analyst.config.role} (ID: {analyst.state.agent_id})")

    # Шаг 1: Researcher выполняет задачу
    print("\n2️⃣ Researcher выполняет задачу...")
    research_result = researcher.execute_task(
        "Исследование методов вихретокового контроля"
    )
    print(f"   ✅ Результат: {len(research_result.get('sources_found', []))} источников найдено")

    # Шаг 2: Отправка сообщения от Researcher к Analyst
    print("\n3️⃣ Отправка сообщения от Researcher к Analyst...")
    message = researcher.send_message(
        receiver_id=analyst.state.agent_id,
        content={"research_data": research_result, "type": "research_complete"}
    )
    print(f"   📨 Сообщение отправлено: {message['type']}")

    # Шаг 3: Analyst получает сообщение
    print("\n4️⃣ Analyst получает сообщение...")
    analyst.receive_message(message)
    print(f"   📬 Сообщение получено")

    # Шаг 4: Проверка истории сообщений
    print("\n5️⃣ Проверка истории сообщений:")
    print(f"   📤 Researcher отправлено: {researcher.statistics['messages_sent']}")
    print(f"   📥 Analyst получено: {analyst.statistics['messages_received']}")

    # Шаг 5: Analyst обрабатывает данные
    print("\n6️⃣ Analyst обрабатывает полученные данные...")
    analysis_result = analyst.execute_task(
        "Анализ исследовательских данных",
        context={"research_data": research_result}
    )
    print(f"   ✅ Анализ завершён: {len(analysis_result.get('insights', []))} инсайтов")

    # Итог
    print("\n" + "=" * 60)
    print("ИТОГ ТЕСТА КОММУНИКАЦИИ")
    print("=" * 60)

    # Проверка что данные переданы
    data_transferred = "research_data" in analyst.state.message_history[-1]["message"]["content"] \
        if analyst.state.message_history else False

    if data_transferred:
        print("✅ Данные успешно переданы между агентами!")
        print("✅ Агенты скоординированы!")
    else:
        print("❌ Передача данных не обнаружена")

    print(f"\n📊 Статистика:")
    print(f"   Researcher: {researcher.statistics['tasks_completed']} задач, "
          f"{researcher.statistics['messages_sent']} сообщений")
    print(f"   Analyst: {analyst.statistics['tasks_completed']} задач, "
          f"{analyst.statistics['messages_received']} сообщений")


if __name__ == "__main__":
    test_direct_communication()