# -*- coding: utf-8 -*-
from typing import Dict, Optional, List
import time
import logging
from src.agents.base_agent import BaseAgent, AgentConfig
from src.llm.yandex_gpt_client import yandex_gpt

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Исследователь (Yandex GPT)",
            goal="Найти актуальную информацию по заданной теме",
            backstory="Вы - эксперт по неразрушающему контролю, использующий Yandex GPT"
        )
        super().__init__(config or default_config)

    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        start_time = time.time()

        # Формируем запрос к Yandex GPT
        prompt = f"""
        Тема исследования: {task_description}

        Предоставь 4-5 ключевых фактов по этой теме.
        Факты должны быть конкретными, содержательными и соответствовать теме.
        """

        system_prompt = "Ты - эксперт по неразрушающему контролю металлических изделий и конвейерным линиям."

        try:
            llm_response = yandex_gpt.generate(prompt, system_prompt)
            facts = self._parse_facts(llm_response)

            # Генерируем рекомендации
            rec_prompt = f"Предоставь 3-4 практические рекомендации по теме: {task_description}"
            rec_response = yandex_gpt.generate(rec_prompt, system_prompt)
            recommendations = self._parse_recommendations(rec_response)

        except Exception as e:
            logger.error(f"Ошибка Yandex GPT: {e}")
            facts = self._get_mock_facts()
            recommendations = self._get_mock_recommendations()

        results = {
            "task": task_description,
            "status": "completed",
            "sources_found": self._get_sources(),
            "key_facts": facts,
            "recommendations": recommendations,
            "execution_time": time.time() - start_time
        }

        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1

        return results

    def _parse_facts(self, response: str) -> List[str]:
        facts = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                fact = line.lstrip('1234567890. •-').strip()
                if len(fact) > 10:
                    facts.append(fact)
        return facts[:5] if facts else [response[:200]]

    def _parse_recommendations(self, response: str) -> List[str]:
        recs = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                rec = line.lstrip('1234567890. •-').strip()
                if len(rec) > 10:
                    recs.append(rec)
        return recs[:4] if recs else ["Внедрить систему контроля качества"]

    def _get_sources(self) -> List[Dict]:
        return [
            {"title": "Yandex GPT (AI)", "type": "ai", "relevance": 0.95},
            {"title": "База знаний Yandex GPT", "type": "llm", "relevance": 0.90}
        ]

    def _get_mock_facts(self) -> List[str]:
        return ["Вихретоковый контроль точность 95-98%"]

    def _get_mock_recommendations(self) -> List[str]:
        return ["Используйте гибридную систему контроля"]

    def get_capabilities(self) -> List[str]:
        return ["Поиск информации через AI", "Анализ данных", "Генерация рекомендаций"]