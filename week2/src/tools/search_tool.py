# -*- coding: utf-8 -*
"""
Инструмент поиска в интернете
Лабораторная работа №2
"""
from langchain_core.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import logging
from ddgs import DDGS

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Схема входных параметров для поиска."""
    query: str = Field(
        description="Поисковый запрос",
        min_length=1,
        max_length=500
    )
    num_results: int = Field(
        description="Количество результатов (1-10)",
        default=5,
        ge=1,
        le=10
    )


class SearchTool(BaseTool):
    """
    Инструмент для поиска информации в интернете.
    Использует DuckDuckGo для реального поиска.
    """
    name: str = "search_web"
    description: str = """
    Поиск актуальной информации в интернете по запросу.
    Используйте для получения свежих данных, новостей, документации.
    Возвращает до 10 результатов поиска с описанием.
    """
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """
        Выполнение поиска.

        Args:
            query: Поисковый запрос
            num_results: Количество результатов

        Returns:
            str: Форматированные результаты поиска
        """
        logger.info(f"Поиск: {query} (результатов: {num_results})")

        # Пробуем реальный поиск через DuckDuckGo
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for i, r in enumerate(ddgs.text(query, max_results=num_results)):
                    if i >= num_results:
                        break
                    results.append(f"📌 {r.get('title', 'Без названия')}")
                    results.append(f"   🔗 {r.get('href', '')}")
                    results.append(f"   📝 {r.get('body', '')[:200]}")
                    results.append("")

            if results:
                formatted = "\n".join(results)
                return f"🔍 Результаты поиска: {query}\n\n{formatted}"

        except ImportError:
            logger.warning("DuckDuckGo не установлен. pip install duckduckgo-search")
        except Exception as e:
            logger.warning(f"Ошибка поиска: {e}")

        # Fallback: используем requests для быстрого поиска
        try:
            import requests
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            resp = requests.get(url, timeout=10)
            data = resp.json()

            results = []
            if data.get('AbstractText'):
                results.append(f"📌 {data.get('AbstractText')[:300]}")
                results.append(f"   🔗 {data.get('AbstractURL', '')}")

            for topic in data.get('RelatedTopics', [])[:num_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append(f"📌 {topic['Text'][:200]}")
                    results.append(f"   🔗 {topic.get('FirstURL', '')}")
                    results.append("")

            if results:
                return f"🔍 Результаты поиска: {query}\n\n" + "\n".join(results)

        except Exception as e:
            logger.warning(f"Fallback поиск не удался: {e}")

        # Если ничего не сработало - возвращаем информативное сообщение
        return f"""🔍 Поиск по запросу: {query}

Для реального поиска установите:
    pip install duckduckgo-search

Результат 1: По запросу '{query}' рекомендуется обратиться к официальной документации.
Результат 2: Актуальную информацию можно найти на wikipedia.org или официальных сайтах.
Результат 3: Используйте специализированные источники по теме '{query}'."""

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Асинхронная версия."""
        return self._run(query, num_results)

    def to_langchain_tool(self):
        """Конвертация в формат LangChain."""
        return self