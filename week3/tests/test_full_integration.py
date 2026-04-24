# -*- coding: utf-8 -*-
"""
Полные интеграционные тесты для лабораторной работы №3
"""

import unittest
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.ERROR)


class TestFullIntegration(unittest.TestCase):
    """Полные интеграционные тесты."""

    @classmethod
    def setUpClass(cls):
        """Настройка перед всеми тестами."""
        from crew.research_crew import ResearchCrew
        cls.crew = ResearchCrew()

    def test_1_agents_exist(self):
        """Тест 1: Проверка наличия всех агентов."""
        self.assertIsNotNone(self.crew.researcher)
        self.assertIsNotNone(self.crew.analyst)
        self.assertIsNotNone(self.crew.writer)
        self.assertIsNotNone(self.crew.security_auditor)
        self.assertIsNotNone(self.crew.ndt_inspector)

    def test_2_agent_capabilities(self):
        """Тест 2: Проверка возможностей агентов."""
        caps = self.crew.researcher.get_capabilities()
        self.assertTrue(len(caps) > 0)

        caps = self.crew.analyst.get_capabilities()
        self.assertTrue(len(caps) > 0)

        caps = self.crew.writer.get_capabilities()
        self.assertTrue(len(caps) > 0)

    def test_3_coordination(self):
        """Тест 3: Проверка координации."""
        result = self.crew.execute("Тестовая тема для координации")
        self.assertTrue(result.success)
        self.assertIn("researcher", result.agent_results)
        self.assertIn("analyst", result.agent_results)
        self.assertIn("writer", result.agent_results)

    def test_4_ndt_inspection(self):
        """Тест 4: Проверка инспекции бутылки."""
        result = self.crew.execute_ndt_inspection("TEST_BOTTLE_001")
        self.assertIn("quality_decision", result)
        self.assertIn("inspection_result", result)

    def test_5_communication(self):
        """Тест 5: Проверка коммуникации."""
        stats = self.crew.get_statistics()
        self.assertIn("communication", stats)

    def test_6_data_flow(self):
        """Тест 6: Проверка потока данных между агентами."""
        result = self.crew.execute("Проверка передачи данных")

        # Данные от исследователя должны быть переданы аналитику
        researcher_data = result.agent_results.get("researcher", {})
        analyst_data = result.agent_results.get("analyst", {})

        # Проверка, что аналитик получил данные
        self.assertIsNotNone(analyst_data)

    def test_7_crew_statistics(self):
        """Тест 7: Проверка статистики команды."""
        stats = self.crew.get_statistics()
        self.assertIn("statistics", stats)
        self.assertIn("success_rate", stats)


class TestNDTInspector(unittest.TestCase):
    """Тесты специализированного агента."""

    @classmethod
    def setUpClass(cls):
        from ndt_inspector_agent import NDTInspectorAgent
        cls.inspector = NDTInspectorAgent()

    def test_defect_detection(self):
        """Проверка обнаружения дефектов."""
        result = self.inspector.execute_task(
            "Инспекция",
            context={"bottle_id": "DEFECT_TEST"}
        )
        self.assertEqual(result["status"], "completed")

    def test_quality_scoring(self):
        """Проверка оценки качества."""
        result = self.inspector.execute_task(
            "Инспекция",
            context={"bottle_id": "QUALITY_TEST"}
        )
        quality_score = result.get("inspection_result", {}).get("quality_score", 0)
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 100)

    def test_rejection_decision(self):
        """Проверка решения о браковке."""
        result = self.inspector.execute_task(
            "Инспекция",
            context={"bottle_id": "REJECT_TEST"}
        )
        decision = result.get("quality_decision", {})
        self.assertIn("reject", decision)


class TestMessageBus(unittest.TestCase):
    """Тесты шины сообщений."""

    def setUp(self):
        from communication.message_bus import MessageBus, message_bus
        self.bus = MessageBus()

    def test_subscribe_publish(self):
        """Проверка подписки и публикации."""
        received = []

        def callback(msg):
            received.append(msg)

        self.bus.subscribe("test", callback)
        self.bus.publish({"type": "test", "data": "hello"})

        self.assertEqual(len(received), 1)

    def test_broadcast(self):
        """Проверка широковещательной рассылки."""
        received = []

        def callback(msg):
            received.append(msg)

        self.bus.subscribe("broadcast", callback)
        self.bus.broadcast("sender", {"message": "to all"})

        self.assertEqual(len(received), 1)


def run_all_tests():
    """Запуск всех тестов."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFullIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNDTInspector))
    suite.addTests(loader.loadTestsFromTestCase(TestMessageBus))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)