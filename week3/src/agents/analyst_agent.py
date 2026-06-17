# -*- coding: utf-8 -*-
from typing import Dict, Optional, List
import time
import logging
from src.agents.base_agent import BaseAgent, AgentConfig
from src.llm.yandex_gpt_client import yandex_gpt

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Аналитик (Yandex GPT)",
            goal="Проанализировать данные и выявить ключевые инсайты",
            backstory="Вы - аналитик данных, использующий AI для анализа"
        )
        super().__init__(config or default_config)

    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        start_time = time.time()

        research_data = context.get("research_data", {}) if context else {}
        facts = research_data.get("key_facts", [])

        # Анализ данных через Yandex GPT
        prompt = f"""
        На основе следующих фактов:
        {chr(10).join(f'- {f}' for f in facts)}

        Задача: {task_description}

        Предоставь:
        1. 3-4 ключевых инсайта
        2. Рекомендации по улучшению
        """

        try:
            response = yandex_gpt.generate(prompt, "Ты - аналитик данных, специализирующийся на контроле качества")
            insights = self._parse_insights(response)
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            insights = ["Оптимизировать контроль качества", "Увеличить скорость проверки"]

        results = {
            "task": task_description,
            "status": "completed",
            "insights": insights,
            "execution_time": time.time() - start_time
        }

        self.statistics["tasks_completed"] += 1
        return results

    def _parse_insights(self, response: str) -> List[str]:
        insights = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                insight = line.lstrip('1234567890. •-').strip()
                if len(insight) > 10:
                    insights.append(insight)
        return insights[:4] if insights else [response[:150]]

    def get_capabilities(self) -> List[str]:
        return ["Анализ данных через AI", "Выявление инсайтов", "Формирование выводов"]