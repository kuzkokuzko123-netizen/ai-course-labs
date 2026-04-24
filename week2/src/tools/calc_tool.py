# -*- coding: utf-8 -*
"""
Инструмент математических вычислений
Лабораторная работа №2
"""
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import ast
import operator
import logging

logger = logging.getLogger(__name__)


class CalculateInput(BaseModel):
    """Схема входных параметров для вычислений."""
    expression: str = Field(
        description="Математическое выражение (например: 2+2*3)",
        min_length=1,
        max_length=200
    )
    precision: int = Field(
        description="Точность результата (знаков после запятой)",
        default=2,
        ge=0,
        le=10
    )


class SafeCalculator:
    """Безопасный калькулятор без eval()."""

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_expr(self, expr: str) -> float:
        """Безопасное вычисление выражения."""
        try:
            node = ast.parse(expr, mode='eval').body
            return self._eval_node(node)
        except Exception as e:
            raise ValueError(f"Ошибка вычисления: {e}")

    def _eval_node(self, node) -> float:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Для старых версий Python
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            return self.OPERATORS[type(node.op)](operand)
        else:
            raise ValueError(f"Неподдерживаемая операция: {type(node)}")


class CalculateTool(BaseTool):
    """Инструмент для безопасных математических вычислений."""

    # ✅ ЯВНОЕ УКАЗАНИЕ ТИПОВ ДЛЯ PYDANTIC V2
    name: str = "calculate"
    description: str = """
    Выполнение математических вычислений.
    Используйте для расчётов, формул, статистики.
    Поддерживает: +, -, *, /, **, скобки.
    """
    args_schema: Type[BaseModel] = CalculateInput

    calculator: SafeCalculator = SafeCalculator()

    def _run(self, expression: str, precision: int = 2) -> str:
        """
        Выполнение вычислений.

        Args:
            expression: Математическое выражение
            precision: Точность результата

        Returns:
            str: Результат вычислений
        """
        logger.info(f"Вычисление: {expression}")

        try:
            result = self.calculator.eval_expr(expression)
            return f"🧮 {expression} = {result:.{precision}f}"
        except Exception as e:
            return f"❌ Ошибка вычисления: {e}"

    async def _arun(self, expression: str, precision: int = 2) -> str:
        """Асинхронная версия."""
        return self._run(expression, precision)

    def to_langchain_tool(self):
        """Конвертация в формат LangChain."""
        return self
