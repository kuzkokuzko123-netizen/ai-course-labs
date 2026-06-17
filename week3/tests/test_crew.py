# -*- coding: utf-8 -*-
"""
Тесты для многоагентной системы
Лабораторная работа №3
"""

import unittest
import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.base_agent import BaseAgent, AgentConfig
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.writer_agent import WriterAgent
from crew.research_crew import ResearchCrew, CrewConfig


class TestBaseAgent(unittest.TestCase):
    """Тесты базового агента."""

    def test_agent_initialization(self):
        """Тест инициализации агента."""
        config = AgentConfig(role="Tester", goal="Test")
        agent = ResearcherAgent(config)
        self.assertEqual(agent.config.role, "Tester")

    def test_agent_statistics(self):
        """Тест статистики агента."""
        agent = ResearcherAgent()
        stats = agent.get_statistics()
        self.assertIn("agent_id", stats)
        self.assertIn("statistics", stats)


class TestResearcherAgent(unittest.TestCase):
    """Тесты агента-исследователя."""

    def test_execute_task(self):
        """Тест выполнения задачи."""
        agent = ResearcherAgent()
        result = agent.execute_task("Test research topic")
        self.assertEqual(result["status"], "completed")
        self.assertIn("sources_found", result)
        self.assertIn("key_facts", result)

    def test_capabilities(self):
        """Тест получения возможностей."""
        agent = ResearcherAgent()
        capabilities = agent.get_capabilities()
        self.assertIsInstance(capabilities, list)
        self.assertTrue(len(capabilities) > 0)


class TestAnalystAgent(unittest.TestCase):
    """Тесты агента-аналитика."""

    def test_execute_task_with_context(self):
        """Тест выполнения задачи с контекстом."""
        agent = AnalystAgent()
        context = {"research_data": {"sources_found": ["source1"], "key_facts": ["fact1"]}}
        result = agent.execute_task("Test analysis", context)
        self.assertEqual(result["status"], "completed")
        self.assertIn("structured_data", result)

    def test_capabilities(self):
        """Тест получения возможностей."""
        agent = AnalystAgent()
        capabilities = agent.get_capabilities()
        self.assertIsInstance(capabilities, list)
        self.assertTrue(len(capabilities) > 0)


class TestWriterAgent(unittest.TestCase):
    """Тесты агента-писателя."""

    def test_execute_task(self):
        """Тест выполнения задачи."""
        agent = WriterAgent()
        context = {
            "research_data": {"key_facts": ["fact1", "fact2"], "recommendations": ["rec1"]},
            "analysis_data": {"insights": ["insight1"]}
        }
        result = agent.execute_task("Test writing", context)
        self.assertEqual(result["status"], "completed")
        self.assertIn("document", result)

    def test_capabilities(self):
        """Тест получения возможностей."""
        agent = WriterAgent()
        capabilities = agent.get_capabilities()
        self.assertIsInstance(capabilities, list)
        self.assertTrue(len(capabilities) > 0)


class TestResearchCrew(unittest.TestCase):
    """Тесты команды агентов."""

    def test_crew_initialization(self):
        """Тест инициализации команды."""
        crew = ResearchCrew()
        self.assertIsNotNone(crew.researcher)
        self.assertIsNotNone(crew.analyst)
        self.assertIsNotNone(crew.writer)

    def test_crew_execution(self):
        """Тест выполнения команды."""
        crew = ResearchCrew()
        result = crew.execute("Test research topic for testing")
        self.assertIsNotNone(result)
        self.assertIn("success", result.__dict__)

    def test_crew_statistics(self):
        """Тест статистики команды."""
        crew = ResearchCrew()
        stats = crew.get_statistics()
        self.assertIn("statistics", stats)
        self.assertIn("config", stats)


class TestNDTInspectorAgent(unittest.TestCase):
    """Тесты агента неразрушающего контроля."""

    def setUp(self):
        """Подготовка к тестам."""
        try:
            from agents.ndt_inspector_agent import NDTInspectorAgent
            self.agent_class = NDTInspectorAgent
            self.available = True
        except ImportError:
            self.available = False

    def test_agent_creation(self):
        """Тест создания агента НК."""
        if not self.available:
            self.skipTest("NDTInspectorAgent не импортирован")

        from agents.ndt_inspector_agent import NDTInspectorAgent
        agent = NDTInspectorAgent()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.config.role, "Инспектор неразрушающего контроля")

    def test_inspection_execution(self):
        """Тест выполнения инспекции."""
        if not self.available:
            self.skipTest("NDTInspectorAgent не импортирован")

        from agents.ndt_inspector_agent import NDTInspectorAgent
        agent = NDTInspectorAgent()
        context = {"bottle_id": "TEST_001"}
        result = agent.execute_task("Test inspection", context)
        self.assertEqual(result["status"], "completed")
        self.assertIn("quality_decision", result)


def run_tests():
    """Запуск всех тестов."""
    # Настройка логирования для тестов
    import logging
    logging.basicConfig(level=logging.ERROR)

    # Создание тестового набора
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestBaseAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestResearcherAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalystAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestWriterAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestResearchCrew))
    suite.addTests(loader.loadTestsFromTestCase(TestNDTInspectorAgent))

    # Запуск тестов
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
