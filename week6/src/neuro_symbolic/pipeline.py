# -*- coding: utf-8 -*-
"""
Нейро-символьный гибридный пайплайн
Лабораторная работа №6
Дисциплина: Искусственный интеллект

Автор: Мыльников Александр Русланович
Группа: ФИТ-221
Дата: 2026
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import sys
import os

# Добавляем папку src в sys.path, чтобы работали импорты
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.llm_client import LLMClient
from symbolic.rule_engine import RuleEngine, Rule, RulePriority, InferenceResult
from symbolic.knowledge_base import KnowledgeBase, KnowledgeFact

logger = logging.getLogger(__name__)


class NeuroSymbolicPipeline:
    """
    Нейро-символьный гибридный пайплайн.

    Архитектура:
        1. Входные данные → Нейронная компонента (LLM)
        2. Извлечённые факты → Символьная компонента (Rules)
        3. Комбинация выводов → Финальный ответ с объяснением

    Атрибуты:
        llm: LLM клиент
        rule_engine: Движок правил
        knowledge_base: База знаний
        neural_weight: Вес нейронного вывода (0-1)
        symbolic_weight: Вес символьного вывода (0-1)
    """

    def __init__(self, llm: Optional[LLMClient] = None,
                 rule_engine: Optional[RuleEngine] = None,
                 knowledge_base: Optional[KnowledgeBase] = None,
                 neural_weight: float = 0.6,
                 symbolic_weight: float = 0.4):
        self.llm = llm or LLMClient()
        self.rule_engine = rule_engine or RuleEngine()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        logger.info(f"NeuroSymbolicPipeline инициализирован "
                    f"(neural={neural_weight}, symbolic={symbolic_weight})")

    def process(self, input_data: Dict[str, Any],
                include_explanation: bool = True) -> Dict[str, Any]:
        """
        Обработка входных данных через гибридный пайплайн.

        Args:
            input_data: Входные данные (факты, запрос)
            include_explanation: Включать ли объяснение

        Returns:
            Dict: Результат обработки
        """
        start_time = time.time()
        logger.info(f"Обработка запроса: {input_data.get('query', 'N/A')[:100]}...")

        # ЭТАП 1: НЕЙРОННАЯ КОМПОНЕНТА
        neural_result = self._neural_processing(input_data)

        # ЭТАП 2: СИМВОЛЬНАЯ КОМПОНЕНТА
        symbolic_result = self._symbolic_processing(input_data)

        # ЭТАП 3: ИНТЕГРАЦИЯ ВЫВОДОВ
        integrated_result = self._integrate_results(
            neural_result, symbolic_result, input_data
        )

        execution_time = time.time() - start_time
        result = {
            "success": True,
            "input": input_data,
            "neural_output": neural_result,
            "symbolic_output": symbolic_result,
            "final_decision": integrated_result["decision"],
            "confidence": integrated_result["confidence"],
            "explanation": integrated_result["explanation"] if include_explanation else "",
            "execution_time": round(execution_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Пайплайн завершён за {execution_time:.3f}с")
        return result

    def _neural_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Нейронная обработка через LLM."""
        query = input_data.get("query", "")

        # Классификация запроса
        categories = input_data.get("categories", ["норма", "предупреждение", "критично"])
        classification = self.llm.classify(query, categories)

        # Генерация предварительного вывода
        prompt = f"""Проанализируй данные и сделай предварительный вывод.

Данные: {input_data}

Вывод:"""
        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt="Ты – эксперт-аналитик. Делай точные выводы на основе данных."
        )

        return {
            "classification": classification,
            "preliminary_conclusion": llm_response.get("text", ""),
            "confidence": classification.get("confidence", 0.5),
            "success": llm_response.get("success", False)
        }

    def _symbolic_processing(self, input_data: Dict[str, Any]) -> InferenceResult:
        """Символьная обработка через правила."""
        facts = input_data.get("facts", {})
        return self.rule_engine.infer(facts)

    def _integrate_results(self, neural: Dict[str, Any],
                           symbolic: InferenceResult,
                           input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Интеграция результатов нейронной и символьной компонент."""
        neural_conclusion = neural.get("preliminary_conclusion", "")
        neural_confidence = neural.get("confidence", 0.5)
        symbolic_conclusions = symbolic.conclusions
        symbolic_confidence = 1.0 if symbolic.success else 0.0

        # Взвешенная комбинация
        final_confidence = (
                neural_confidence * self.neural_weight +
                symbolic_confidence * self.symbolic_weight
        )

        decision_parts = []
        if neural_conclusion:
            decision_parts.append(f"Нейронный вывод: {neural_conclusion}")
        if symbolic_conclusions:
            decision_parts.append(f"Символьный вывод: {', '.join(symbolic_conclusions)}")
        final_decision = "\n".join(decision_parts) if decision_parts else "Нет выводов"

        explanation = self._generate_explanation(neural, symbolic, final_confidence)
        return {
            "decision": final_decision,
            "confidence": round(final_confidence, 3),
            "explanation": explanation
        }

    def _generate_explanation(self, neural: Dict[str, Any],
                              symbolic: InferenceResult,
                              confidence: float) -> str:
        """Генерация объяснения решения."""
        explanation_parts = ["=== ОБЪЯСНЕНИЕ РЕШЕНИЯ ===\n"]

        explanation_parts.append("Нейронная компонента:")
        classification = neural.get('classification', {})
        explanation_parts.append(f" • Классификация: {classification.get('predicted_category', 'N/A')}")
        explanation_parts.append(f" • Уверенность: {neural.get('confidence', 0):.1%}")

        explanation_parts.append("\nСимвольная компонента:")
        if symbolic.triggered_rules:
            for rule in symbolic.triggered_rules:
                explanation_parts.append(f" • {rule.name}: {rule.description}")
        else:
            explanation_parts.append(" • Правила не сработали")

        explanation_parts.append(f"\nИтоговая уверенность: {confidence:.1%}")
        explanation_parts.append(f" • Вес нейронной компоненты: {self.neural_weight:.1%}")
        explanation_parts.append(f" • Вес символьной компоненты: {self.symbolic_weight:.1%}")

        return "\n".join(explanation_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика пайплайна."""
        return {
            "neural_weight": self.neural_weight,
            "symbolic_weight": self.symbolic_weight,
            "rules_count": len(self.rule_engine.rules),
            "facts_count": len(self.knowledge_base.facts),
            "inferences_count": len(self.rule_engine.inference_history)
        }


# Точка входа для тестирования
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №6")
    print("ТЕСТИРОВАНИЕ НЕЙРО-СИМВОЛЬНОГО ПАЙПЛАЙНА")
    print("=" * 80)

    llm = LLMClient()
    rule_engine = RuleEngine()

    # Правила для технической диагностики
    rule_engine.add_rule(Rule(
        rule_id="TEMP_CRITICAL",
        name="Критическая температура",
        condition=lambda f: f.get("temperature", 0) > 90,
        conclusion="КРИТИЧЕСКОЕ СОСТОЯНИЕ: Немедленная остановка оборудования",
        priority=RulePriority.CRITICAL,
        description="Температура превышает 90°C",
        domain="technical"
    ))
    rule_engine.add_rule(Rule(
        rule_id="TEMP_WARNING",
        name="Предупреждение температуры",
        condition=lambda f: 80 < f.get("temperature", 0) <= 90,
        conclusion="ПРЕДУПРЕЖДЕНИЕ: Проверить систему охлаждения",
        priority=RulePriority.HIGH,
        description="Температура в диапазоне 80–90°C",
        domain="technical"
    ))
    rule_engine.add_rule(Rule(
        rule_id="PRESSURE_LOW",
        name="Низкое давление",
        condition=lambda f: f.get("pressure", 100) < 50,
        conclusion="ПРЕДУПРЕЖДЕНИЕ: Проверить герметичность системы",
        priority=RulePriority.HIGH,
        description="Давление ниже 50",
        domain="technical"
    ))

    pipeline = NeuroSymbolicPipeline(
        llm=llm,
        rule_engine=rule_engine,
        neural_weight=0.6,
        symbolic_weight=0.4
    )

    test_cases = [
        {
            "query": "Двигатель перегрелся, температура 95°C, давление 60",
            "facts": {"temperature": 95, "pressure": 60},
            "categories": ["норма", "предупреждение", "критично"]
        },
        {
            "query": "Оборудование работает нормально, температура 70°C",
            "facts": {"temperature": 70, "pressure": 80},
            "categories": ["норма", "предупреждение", "критично"]
        }
    ]

    print("\n" + "=" * 80)
    print("ТЕСТОВЫЕ СЦЕНАРИИ")
    print("=" * 80)

    for i, test_data in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"ТЕСТ {i}: {test_data['query']}")
        print(f"{'=' * 60}")

        result = pipeline.process(test_data)
        print(f"\nФИНАЛЬНОЕ РЕШЕНИЕ:")
        print(result["final_decision"])
        print(f"\nУВЕРЕНОСТЬ: {result['confidence']:.1%}")
        print(f"\n{result['explanation']}")
        print(f"\nВремя выполнения: {result['execution_time']}с")

    print("\n" + "=" * 80)
    print(f"СТАТИСТИКА: {pipeline.get_statistics()}")
    print("=" * 80)
