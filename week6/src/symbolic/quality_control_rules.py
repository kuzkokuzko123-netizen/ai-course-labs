# -*- coding: utf-8 -*-
"""
Правила для обнаружения дефектных изделий на производственной линии
Лабораторная работа №6

Автор: Мыльников Александр Русланович
Группа: ФИТ-221
Специальность: 02.03.02 Фундаментальная информатика и информационные технологии
Тема диплома: Обнаружение дефектных изделий на производственной линии
              с использованием алгоритмов компьютерного зрения
"""

#from rule_engine import Rule, RulePriority
from .rule_engine import Rule, RulePriority

def get_quality_control_rules() -> list:
    """Получение правил контроля качества изделий."""
    return [
        Rule(
            rule_id="QC_001",
            name="Размер вне допуска",
            condition=lambda f: f.get("dimension_mm", 100) > f.get("tolerance_upper", 105)
                                or f.get("dimension_mm", 100) < f.get("tolerance_lower", 95),
            conclusion="БРАК: Размер изделия не соответствует допускам",
            priority=RulePriority.CRITICAL,
            description="Геометрический размер выходит за границы допуска",
            domain="quality_control"
        ),
        Rule(
            rule_id="QC_002",
            name="Цветовой дефект",
            condition=lambda f: f.get("color_delta_e", 0) > 5.0,
            conclusion="БРАК: Обнаружено значительное отклонение цвета",
            priority=RulePriority.HIGH,
            description="Дельта E превышает порог 5 (заметное различие)",
            domain="quality_control"
        ),
        Rule(
            rule_id="QC_003",
            name="Отсутствие компонента",
            condition=lambda f: f.get("components_present", 1) < f.get("expected_components", 1),
            conclusion="БРАК: Отсутствует один или несколько компонентов",
            priority=RulePriority.CRITICAL,
            description="Обнаружено отсутствие обязательного элемента сборки",
            domain="quality_control"
        ),
        Rule(
            rule_id="QC_004",
            name="Поверхностный дефект",
            condition=lambda f: f.get("surface_defect_area", 0) > 0.5,
            conclusion="ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНЫЙ КОНТРОЛЬ: Обнаружены царапины/сколы",
            priority=RulePriority.HIGH,
            description="Площадь поверхностного дефекта превышает 0.5 мм²",
            domain="quality_control"
        ),
        Rule(
            rule_id="QC_005",
            name="Смещение позиционирования",
            condition=lambda f: abs(f.get("position_offset_mm", 0)) > 1.0,
            conclusion="ПРЕДУПРЕЖДЕНИЕ: Смещение при установке детали превышает 1 мм",
            priority=RulePriority.MEDIUM,
            description="Позиционное смещение не соответствует спецификации",
            domain="quality_control"
        ),
        Rule(
            rule_id="QC_006",
            name="Низкая уверенность классификации",
            condition=lambda f: f.get("classifier_confidence", 1.0) < 0.7,
            conclusion="ТРЕБУЕТСЯ РУЧНАЯ ПРОВЕРКА: Уверенность модели ниже 70%",
            priority=RulePriority.MEDIUM,
            description="Модель компьютерного зрения не уверена в классе дефекта",
            domain="quality_control"
        ),
        Rule(
            rule_id="QC_007",
            name="Критическое превышение дефектов",
            condition=lambda f: f.get("defect_count", 0) > 3,
            conclusion="ОСТАНОВКА ЛИНИИ: Более трёх дефектов на одном изделии",
            priority=RulePriority.CRITICAL,
            description="Число обнаруженных дефектов превышает допустимый лимит",
            domain="quality_control"
        )
    ]


# Проверка
if __name__ == "__main__":
    rules = get_quality_control_rules()
    print(f"Загружено {len(rules)} правил контроля качества")
    for rule in rules:
        print(f" • {rule.rule_id}: {rule.name} (приоритет: {rule.priority.name})")
