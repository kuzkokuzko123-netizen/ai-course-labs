# -*- coding: utf-8 -*-
"""
Специализированный агент для дипломной работы
Лабораторная работа №3

Автор: [Мыльников]
Специальность: [02.03.02 Фундаментальная информатика и информационные технологии]
Тема диплома: [Разработка системы неразрушающего контроля для выявления дефектов (брака)
              металлических бутылок на конвейерной линии]
"""

from typing import Dict, Optional, List, Any
from datetime import datetime
import time
import logging
import random

from agents.base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class NDTInspectorAgent(BaseAgent):
    """
    Агент неразрушающего контроля для инспекции металлических бутылок.
    Специализация для дипломной работы.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Инспектор неразрушающего контроля",
            goal="Обнаруживать и классифицировать дефекты металлических бутылок на конвейерной линии с точностью не менее 95%",
            backstory="Вы — специалист по неразрушающему контролю металлических изделий."
        )
        if config:
            default_config.role = config.role
            default_config.goal = config.goal
            default_config.backstory = config.backstory
        super().__init__(default_config)
        self.inspection_history = []

    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """Выполнение инспекции."""
        start_time = time.time()
        self.state.current_task = task_description

        logger.info(f"Агент НК начинает инспекцию")

        bottle_id = context.get("bottle_id", f"BOTTLE_{int(time.time())}") if context else f"BOTTLE_{int(time.time())}"

        # Имитация инспекции с разными результатами
        random.seed(hash(bottle_id) % 2 ** 32)  # детерминированный результат для одного ID

        quality_score = random.uniform(50, 100)
        is_defective = quality_score < 70

        defects = []
        if is_defective:
            defect_types = ["вмятина", "царапина", "коррозия", "дефект сварного шва"]
            defects = [{
                "type": random.choice(defect_types),
                "severity": random.choice(["критический", "значительный", "незначительный"]),
                "size_mm": round(random.uniform(0.5, 5.0), 1),
                "confidence": round(random.uniform(0.7, 0.99), 2)
            }]

        inspection_result = {
            "bottle_id": bottle_id,
            "quality_score": round(quality_score, 1),
            "is_defective": is_defective,
            "defects": defects,
            "inspection_time_ms": round(random.uniform(50, 200), 1)
        }

        decision = {
            "reject": is_defective,
            "action": "REJECT" if is_defective else "ACCEPT",
            "reason": f"Качество: {quality_score:.1f}%" + (" (ниже порога 70%)" if is_defective else " (выше порога)")
        }

        results = {
            "task": task_description,
            "status": "completed",
            "bottle_id": bottle_id,
            "inspection_result": inspection_result,
            "quality_decision": decision,
            "execution_time": time.time() - start_time
        }

        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1
        self.inspection_history.append(inspection_result)

        return results

    def detect_defects(self, bottle_id: str) -> List[Dict]:
        """Обнаружение дефектов (для демонстрации)."""
        defect_types = ["вмятина", "царапина", "коррозия", "дефект сварного шва", "отклонение толщины"]
        return [{
            "type": random.choice(defect_types),
            "severity": random.choice(["критический", "значительный", "незначительный"]),
            "size_mm": round(random.uniform(0.5, 5.0), 1)
        }]

    def get_capabilities(self) -> List[str]:
        """Возможности агента."""
        return [
            "Обнаружение вмятин на металлических бутылках",
            "Обнаружение царапин и дефектов поверхности",
            "Выявление коррозии и питтинга",
            "Контроль качества сварного шва",
            "Измерение толщины стенок",
            "Классификация дефектов по степени критичности",
            "Принятие решения о браковке изделий",
            "Генерация отчётов о контроле качества"
        ]

    def get_statistics_full(self) -> Dict:
        """Полная статистика инспекций."""
        if not self.inspection_history:
            return {"total": 0, "defective": 0, "accept": 0, "accept_rate": 0}

        total = len(self.inspection_history)
        defective = sum(1 for i in self.inspection_history if i.get("is_defective", False))
        accept = total - defective

        # Расчёт средней оценки качества
        avg_quality = sum(i.get("quality_score", 0) for i in self.inspection_history) / total if total > 0 else 0

        return {
            "total": total,
            "defective": defective,
            "accept": accept,
            "accept_rate": round(accept / total * 100, 1),
            "avg_quality_score": round(avg_quality, 1),
            "history": self.inspection_history[-10:]  # последние 10 инспекций
        }

    def get_inspection_summary(self) -> str:
        """Получение краткого отчёта по инспекциям."""
        stats = self.get_statistics_full()

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           ОТЧЁТ ПО НЕРАЗРУШАЮЩЕМУ КОНТРОЛЮ                   ║
╠══════════════════════════════════════════════════════════════╣
║  Всего проверено: {stats['total']:4d} бутылок                               ║
║  Годных:          {stats['accept']:4d} бутылок                               ║
║  Бракованных:     {stats['defective']:4d} бутылок                               ║
║  Процент годных:  {stats['accept_rate']:5.1f}%                                 ║
║  Ср. качество:    {stats['avg_quality_score']:5.1f}%                                 ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary

    def reset_inspection_history(self) -> None:
        """Сброс истории инспекций."""
        self.inspection_history = []
        logger.info("История инспекций сброшена")


# Для демонстрации работы агента
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ СПЕЦИАЛИЗИРОВАННОГО АГЕНТА")
    print("Агент неразрушающего контроля металлических бутылок")
    print("=" * 70)

    # Создание агента
    inspector = NDTInspectorAgent()

    print(f"\n📋 ИНФОРМАЦИЯ ОБ АГЕНТЕ:")
    print(f"   Роль: {inspector.config.role}")
    print(f"   Цель: {inspector.config.goal}")

    print(f"\n🔧 ВОЗМОЖНОСТИ АГЕНТА:")
    for cap in inspector.get_capabilities():
        print(f"   • {cap}")

    # Инспекция партии бутылок
    print(f"\n🏭 ИНСПЕКЦИЯ ПАРТИИ БУТЫЛОК (10 шт):")
    print("-" * 50)

    for i in range(1, 11):
        result = inspector.execute_task(
            "Контроль качества",
            context={"bottle_id": f"BOTTLE_{i:03d}"}
        )

        inspection = result.get('inspection_result', {})
        decision = result.get('quality_decision', {})

        quality = inspection.get('quality_score', 0)
        status = "❌ БРАК" if decision.get('reject') else "✅ ГОДЕН"

        # Показать дефекты если есть
        defects_info = ""
        if inspection.get('defects'):
            defects_info = f" | Дефект: {inspection['defects'][0]['type']}"

        print(f"   {status} | Бутылка {i:3d} | Качество: {quality:5.1f}%{defects_info}")

    # Итоговая статистика
    print("\n" + "=" * 70)
    print(inspector.get_inspection_summary())
    print("=" * 70)