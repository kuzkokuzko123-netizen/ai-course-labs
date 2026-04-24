# -*- coding: utf-8 -*
"""
Юнит-тесты для AI-агента
Лабораторная работа №2
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_core import AIAgent, AgentConfig
from tools.search_tool import SearchTool
from tools.calc_tool import CalculateTool
from tools.custom_tool import CustomTool
from guardrails.input_validator import InputGuardrails, RiskLevel
from memory.working_memory import WorkingMemory


class TestTools:
    """Тестирование инструментов."""

    def test_search_tool(self):
        """Тест поискового инструмента."""
        tool = SearchTool()
        result = tool._run("test query", num_results=3)
        assert "Результаты поиска" in result

    def test_calc_tool_addition(self):
        """Тест калькулятора - сложение."""
        tool = CalculateTool()
        result = tool._run("2 + 3")
        assert "5" in result

    def test_calc_tool_multiplication(self):
        """Тест калькулятора - умножение."""
        tool = CalculateTool()
        result = tool._run("4 * 5")
        assert "20" in result

    def test_calc_tool_complex(self):
        """Тест калькулятора - сложное выражение."""
        tool = CalculateTool()
        result = tool._run("(10 + 5) * 2")
        assert "30" in result

    def test_calc_tool_error(self):
        """Тест калькулятора - деление на ноль."""
        tool = CalculateTool()
        result = tool._run("2 / 0")
        assert "Ошибка" in result

    def test_custom_tool(self):
        """Тест кастомного инструмента (контроль качества бутылок)."""
        tool = CustomTool()
        result = tool._run("TEST-001", "ультразвук")
        assert "РЕЗУЛЬТАТ" in result


class TestGuardrails:
    """Тестирование guardrails."""

    def setup_method(self):
        self.guardrails = InputGuardrails()

    def test_valid_input(self):
        """Тест валидного ввода."""
        result = self.guardrails.validate_input("Привет, как дела?")
        assert result.is_safe == True
        assert result.risk_level == RiskLevel.LOW

    def test_sql_injection(self):
        """Тест блокировки SQL-инъекции."""
        result = self.guardrails.validate_input("SELECT * FROM users; --")
        assert result.is_safe == False
        assert result.risk_level == RiskLevel.HIGH

    def test_instruction_bypass(self):
        """Тест блокировки попытки обхода инструкций."""
        result = self.guardrails.validate_input("<script>alert('hack')</script>")
        assert result.is_safe == False

    def test_long_input(self):
        """Тест блокировки слишком длинного ввода."""
        long_text = "A" * 50000
        result = self.guardrails.validate_input(long_text)
        assert result.is_safe == False

    def test_tool_validation(self):
        """Тест валидации вызовов инструментов."""
        # Разрешенный инструмент
        result = self.guardrails.validate_tool_call("search_web", {"query": "test"})
        assert result.is_safe == True

        # Запрещенный инструмент
        result = self.guardrails.validate_tool_call("delete_database", {})
        assert result.is_safe == False


class TestMemory:
    """Тестирование рабочей памяти."""

    def test_add_message(self):
        """Тест добавления сообщения."""
        memory = WorkingMemory(max_messages=5)
        memory.add_message("user", "Тестовое сообщение")
        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Тестовое сообщение"

    def test_memory_limit(self):
        """Тест ограничения размера памяти."""
        memory = WorkingMemory(max_messages=3)
        for i in range(5):
            memory.add_message("user", f"Message {i}")
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "Message 2"

    def test_memory_clear(self):
        """Тест очистки памяти."""
        memory = WorkingMemory()
        memory.add_message("user", "Test")
        memory.clear()
        assert len(memory.get_messages()) == 0

    def test_memory_history(self):
        """Тест получения истории диалога."""
        memory = WorkingMemory()
        memory.add_message("user", "Привет")
        memory.add_message("assistant", "Здравствуйте")
        history = memory.get_conversation_history()
        assert "user: Привет" in history
        assert "assistant: Здравствуйте" in history


class TestAgent:
    """Тестирование агента."""

    def test_init(self):
        """Тест инициализации агента."""
        config = AgentConfig(
            name="TestAgent",
            verbose=False,
            memory_enabled=False,
            guardrails_enabled=False
        )
        agent = AIAgent(config)
        assert agent.config.name == "TestAgent"
        assert len(agent.tools) == 3

    def test_stats(self):
        """Тест статистики агента."""
        config = AgentConfig(
            memory_enabled=False,
            guardrails_enabled=False,
            verbose=False
        )
        agent = AIAgent(config)
        stats = agent.get_stats()
        assert stats["name"] == "TechnicalAssistant"
        assert stats["tools_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])