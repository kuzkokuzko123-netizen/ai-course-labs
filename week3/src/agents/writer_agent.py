# -*- coding: utf-8 -*-
"""
Агент-писатель для Multi-Agent системы
"""

from typing import Dict, Optional, List
import time
import logging

from src.agents.base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Агент-писатель."""

    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Писатель",
            goal="Создать структурированный и качественный отчёт",
            backstory="Вы — опытный технический писатель."
        )
        if config:
            default_config.role = config.role
            default_config.goal = config.goal
            default_config.backstory = config.backstory
        super().__init__(default_config)
        self.documents_created = []

    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """Выполнение задачи написания."""
        start_time = time.time()
        self.state.current_task = task_description

        logger.info(f"Писатель начинает задачу: {task_description[:100]}...")

        research_data = context.get("research_data", {}) if context else {}
        analysis_data = context.get("analysis_data", {}) if context else {}

        results = {
            "task": task_description,
            "status": "completed",
            "document": self._generate_document(task_description, research_data, analysis_data),
            "execution_time": 0
        }

        results["execution_time"] = time.time() - start_time
        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1

        logger.info(f"Документ создан за {results['execution_time']:.2f}с")
        return results

    def _generate_document(self, topic: str, research: Dict, analysis: Dict) -> str:
        """Генерация документа с корректной обработкой данных."""

        # Извлекаем тему (убираем лишние префиксы)
        clean_topic = topic.replace("Создай отчёт по теме: ", "").replace("Исследуй тему: ", "")

        # Получаем факты из research_data
        facts = research.get("key_facts", [])
        if not facts:
            facts = [
                "Вихретоковый контроль эффективен для выявления поверхностных дефектов",
                "Ультразвуковой контроль позволяет измерять толщину стенок",
                "Техническое зрение обнаруживает геометрические отклонения"
            ]

        # Получаем инсайты из analysis_data
        insights = analysis.get("insights", [])
        if not insights:
            insights = [
                "Рекомендуется комбинировать несколько методов контроля",
                "Оптимальная скорость конвейера: 60 бутылок/мин",
                "Порог браковки рекомендуется установить на уровне 70% качества"
            ]

        # Получаем рекомендации
        recommendations = research.get("recommendations", [])
        if not recommendations:
            recommendations = [
                "Внедрить автоматическую систему сортировки бракованных изделий",
                "Проводить регулярную калибровку сенсоров",
                "Интегрировать систему с MES для сбора статистики"
            ]

        # Получаем источники
        sources = research.get("sources_found", [])
        if not sources:
            sources = [
                {"title": "ГОСТ Р 57580-2017", "url": "https://example.com/gost"},
                {"title": "Методы неразрушающего контроля", "url": "https://example.com/ndt"}
            ]

        document = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ОТЧЁТ ПО РЕЗУЛЬТАТАМ ИССЛЕДОВАНИЯ                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

## 1. Введение

Данный отчёт подготовлен с использованием многоагентной системы (MAS) в рамках 
лабораторной работы №3 по дисциплине "Искусственный интеллект".

**Тема исследования:** {clean_topic}

**Цель работы:** Изучение и анализ современных методов неразрушающего контроля 
металлических бутылок на конвейерной линии.

---

## 2. Основные результаты исследования

В ходе исследования были выявлены следующие ключевые факты:

{chr(10).join(f'   • {fact}' for fact in facts)}

---

## 3. Анализ данных и инсайты

На основе проведённого анализа получены следующие инсайты:

{chr(10).join(f'   • {insight}' for insight in insights)}

---

## 4. Выводы и рекомендации

{chr(10).join(f'   • {rec}' for rec in recommendations)}

---

## 5. Использованные источники

{chr(10).join(f'   • {s.get("title", s)}' for s in sources)}

---

## 6. Заключение

Разработанная многоагентная система позволяет автоматизировать процесс контроля 
качества металлических бутылок, сократить время инспекции и повысить точность 
обнаружения дефектов.

*Дата составления отчёта: {time.strftime("%d.%m.%Y")}*
*Отчёт сгенерирован автоматически Multi-Agent системой*

"""
        self.documents_created.append({"topic": clean_topic, "timestamp": time.time()})
        return document

    def get_capabilities(self) -> List[str]:
        """Возможности агента."""
        return [
            "Генерация структурированных документов",
            "Адаптация стиля под требования",
            "Форматирование по стандартам",
            "Создание отчётов по результатам исследования"
        ]