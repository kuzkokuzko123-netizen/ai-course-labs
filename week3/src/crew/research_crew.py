# -*- coding: utf-8 -*-
"""
Основная команда агентов (Crew) для исследовательских задач
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime
import time
import logging
import sys
import os

# Добавляем путь к родительской директории
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем агентов
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.writer_agent import WriterAgent

logger = logging.getLogger(__name__)


@dataclass
class CrewConfig:
    """Конфигурация команды агентов."""
    name: str = "ResearchCrew"
    process_type: str = "sequential"
    verbose: bool = True
    memory_enabled: bool = True


@dataclass
class CrewResult:
    """Результат работы команды."""
    success: bool
    final_output: str
    agent_results: Dict
    execution_time: float
    timestamp: str


class ResearchCrew:
    """Команда агентов для исследовательских задач."""

    def __init__(self, config: Optional[CrewConfig] = None):
        self.config = config or CrewConfig()

        # Инициализация агентов
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

        # Общий контекст
        self.shared_context = {}

        # Статистика
        self.statistics = {
            "crews_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0
        }

        logger.info(f"Команда агентов инициализирована: {self.config.name}")

    def execute(self, task: str, context: Optional[Dict] = None) -> CrewResult:
        """Выполнение задачи командой агентов."""
        start_time = time.time()
        logger.info(f"Команда начинает выполнение задачи: {task[:100]}...")

        agent_results = {}
        shared_context = context or {}

        try:
            # Этап 1: Исследование - передаём чистую тему
            logger.info("📚 Этап 1: Исследование")
            researcher_result = self.researcher.execute_task(
                task_description=task,  # Передаём тему без лишних слов
                context=shared_context
            )
            agent_results["researcher"] = researcher_result
            shared_context["research_data"] = researcher_result

            # Этап 2: Анализ
            logger.info("📊 Этап 2: Анализ")
            analyst_result = self.analyst.execute_task(
                task_description=f"Анализ данных по теме: {task}",
                context=shared_context
            )
            agent_results["analyst"] = analyst_result
            shared_context["analysis_data"] = analyst_result

            # Этап 3: Написание отчёта - передаём чистую тему
            logger.info("✍️ Этап 3: Написание отчёта")
            writer_result = self.writer.execute_task(
                task_description=task,  # Передаём тему без лишних слов
                context=shared_context
            )
            agent_results["writer"] = writer_result

            execution_time = time.time() - start_time
            self.statistics["crews_executed"] += 1
            self.statistics["successful_executions"] += 1
            self.statistics["total_execution_time"] += execution_time

            final_output = writer_result.get("document", "Отчёт сгенерирован")

            logger.info(f"✅ Команда успешно завершила работу за {execution_time:.2f}с")

            return CrewResult(
                success=True,
                final_output=final_output,
                agent_results=agent_results,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"❌ Ошибка при выполнении команды: {str(e)}")
            self.statistics["failed_executions"] += 1

            return CrewResult(
                success=False,
                final_output=f"Ошибка: {str(e)}",
                agent_results=agent_results,
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

    def get_statistics(self) -> Dict:
        """Получение статистики работы команды."""
        return {
            "statistics": self.statistics,
            "success_rate": (self.statistics["successful_executions"] /
                             max(self.statistics["crews_executed"], 1)) * 100
        }

    def get_agent_statistics(self) -> Dict:
        """Получение статистики каждого агента."""
        return {
            "researcher": self.researcher.get_statistics(),
            "analyst": self.analyst.get_statistics(),
            "writer": self.writer.get_statistics()
        }
