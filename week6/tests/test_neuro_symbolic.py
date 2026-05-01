# -*- coding: utf-8 -*-
"""Тесты нейро-символьной системы."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.symbolic.rule_engine import RuleEngine, Rule, RulePriority
from src.symbolic.quality_control_rules import get_quality_control_rules


def test_rule_engine():
    engine = RuleEngine()
    engine.add_rule(Rule(
        rule_id="T1", name="Test", condition=lambda f: f.get("x", 0) > 5,
        conclusion="X больше 5", priority=RulePriority.HIGH
    ))
    result = engine.infer({"x": 10})
    assert result.success
    assert "X больше 5" in result.conclusions


def test_quality_rules():
    rules = get_quality_control_rules()
    assert len(rules) >= 5
    # Проверим срабатывание правила размера
    engine = RuleEngine()
    engine.add_rules(rules)
    facts = {"dimension_mm": 110, "tolerance_upper": 105, "tolerance_lower": 95}
    result = engine.infer(facts)
    assert any("Размер" in c for c in result.conclusions)


def test_pipeline_structure():
    # Простая проверка импорта
    from src.neuro_symbolic.pipeline import NeuroSymbolicPipeline
    pipeline = NeuroSymbolicPipeline()
    assert pipeline is not None


if __name__ == "__main__":
    test_rule_engine()
    test_quality_rules()
    test_pipeline_structure()
    print("Все тесты пройдены.")
