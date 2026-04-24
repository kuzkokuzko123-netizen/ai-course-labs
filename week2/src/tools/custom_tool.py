# -*- coding: utf-8 -*-
"""
Специализированный инструмент для дипломной работы
Лабораторная работа №2
Автор: [Мыльников]
Специальность: [02.03.02]
Тема диплома: [Разработка системы неразрушающего контроля для выявления дефектов (брака) металлических бутылок на конвейерной линии]
"""
from langchain.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 1: Определите схему входных параметров
# ═════════════════════════════════════════════════════════════════════════════
class CustomToolInput(BaseModel):
    """
    Схема входных параметров инструмента контроля качества металлических бутылок.
    """
    bottle_id: str = Field(description="Уникальный идентификатор бутылки на конвейере (например: BTL-2026-001)")
    inspection_method: str = Field(
        description="Метод неразрушающего контроля (ультразвук/вихретоковый/оптический/термография)",
        default="ультразвук"
    )
    threshold_defect: float = Field(
        description="Порог чувствительности выявления дефекта (0.0-1.0), где 1.0 - максимальная чувствительность",
        default=0.7,
        ge=0.0,
        le=1.0
    )
    inspection_params: Optional[Dict[str, Any]] = Field(
        description="Дополнительные параметры контроля (частота, амплитуда, угол сканирования и т.д.)",
        default=None
    )


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 2: Реализуйте класс инструмента
# ═════════════════════════════════════════════════════════════════════════════
class CustomTool(BaseTool):
    """
    Специализированный инструмент для системы неразрушающего контроля металлических бутылок.

    Назначение:
    Инструмент обеспечивает автоматизированное выявление дефектов металлических бутылок
    на конвейерной линии с использованием различных методов неразрушающего контроля.

    Как связан с темой дипломной работы:
    - Реализует алгоритмы обработки сигналов от датчиков контроля
    - Классифицирует типы дефектов (трещины, коррозия, неоднородность стенок)
    - Принимает решение о браковке изделия на основе заданных порогов
    - Интегрируется в общую систему управления качеством на производстве

    Пример использования:
    tool = CustomTool()
    result = tool.run(
        bottle_id="BTL-2026-001",
        inspection_method="ультразвук",
        threshold_defect=0.8,
        inspection_params={"frequency": 5.0, "amplitude": 1.5}
    )

    Интеграция с дипломом:
    Инструмент может быть использован в практической части для:
    1. Моделирования процесса контроля на конвейерной линии
    2. Сравнения эффективности различных методов неразрушающего контроля
    3. Оптимизации пороговых значений для минимизации ложных срабатываний
    4. Построения статистики брака для различных партий продукции
    """

    # Обязательные атрибуты
    name = "bottle_defect_detector"
    description = """
    Инструмент для выявления дефектов металлических бутылок на конвейерной линии.

    Методы неразрушающего контроля:
    - ультразвук: выявляет внутренние трещины, неоднородности материала, толщину стенок
    - вихретоковый: обнаруживает поверхностные трещины, коррозию, изменения электропроводности
    - оптический: определяет визуальные дефекты (вмятины, царапины, геометрические отклонения)
    - термография: находит скрытые дефекты через тепловое поле

    Результат анализа включает:
    - Обнаруженные дефекты с координатами
    - Классификацию типа дефекта (критический/некритический)
    - Решение о браковке (годен/брак)
    - Рекомендации по устранению дефектов
    - Статистические метрики достоверности

    Порог чувствительности 0.0-1.0:
    - 0.0-0.3: низкая чувствительность (только явные дефекты)
    - 0.4-0.7: средняя чувствительность (рекомендуется)
    - 0.8-1.0: высокая чувствительность (может давать ложные срабатывания)

    Дополнительные параметры контроля зависят от метода:
    - Ультразвук: frequency (МГц), amplitude (дБ), angle (градусы)
    - Вихретоковый: frequency (кГц), gain (усиление), probe_diameter (мм)
    - Оптический: resolution (пиксели), illumination (люкс)
    - Термография: temperature_range (°C), emissivity (0.0-1.0)
    """
    args_schema: Type[BaseModel] = CustomToolInput

    # База знаний типовых дефектов
    DEFECT_DATABASE = {
        "трещина": {"critical": True, "repairable": False},
        "коррозия": {"critical": True, "repairable": False},
        "вмятина": {"critical": False, "repairable": False},
        "царапина": {"critical": False, "repairable": True},
        "неоднородность": {"critical": True, "repairable": False},
        "пористость": {"critical": True, "repairable": False},
        "геометрическое_отклонение": {"critical": True, "repairable": False}
    }

    def _run(self, bottle_id: str, inspection_method: str = "ультразвук",
             threshold_defect: float = 0.7, inspection_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Основная логика инструмента контроля качества.

        Args:
            bottle_id: Идентификатор бутылки на конвейере
            inspection_method: Метод неразрушающего контроля
            threshold_defect: Порог чувствительности выявления дефекта
            inspection_params: Дополнительные параметры контроля

        Returns:
            str: Результат контроля в текстовом формате

        Raises:
            ValueError: При некорректных входных данных
            Exception: При ошибке выполнения
        """
        logger.info(f"Начало контроля бутылки {bottle_id} методом {inspection_method}")

        # Валидация входных данных
        if not bottle_id:
            raise ValueError("Идентификатор бутылки не может быть пустым")

        if inspection_method not in ["ультразвук", "вихретоковый", "оптический", "термография"]:
            raise ValueError(f"Неподдерживаемый метод контроля: {inspection_method}")

        # Выполнение контроля
        inspection_result = self._perform_inspection(
            bottle_id, inspection_method, threshold_defect, inspection_params
        )

        # Формирование отчета
        report = self._generate_report(bottle_id, inspection_result)

        return report

    def _perform_inspection(self, bottle_id: str, method: str, threshold: float, params: Dict) -> Dict[str, Any]:
        """
        Имитация процесса неразрушающего контроля.

        В реальной системе здесь был бы:
        - Запрос к контроллеру измерительного оборудования
        - Обработка сигналов с датчиков
        - Применение алгоритмов машинного обучения для классификации дефектов
        """

        # Генерация синтетических данных контроля
        np.random.seed(hash(bottle_id) % 2 ** 32)

        # Случайное определение наличия дефекта (для демонстрации)
        has_defect = np.random.random() < (0.3 * threshold)  # 30% брака при max пороге

        if has_defect:
            defect_types = list(self.DEFECT_DATABASE.keys())
            defect_type = np.random.choice(defect_types)
            defect_probability = np.random.uniform(threshold * 0.7, min(1.0, threshold * 1.3))

            # Координаты дефекта (для визуализации)
            defect_location = {
                "x": np.random.uniform(0, 100),
                "y": np.random.uniform(0, 200),
                "depth": np.random.uniform(0, 5) if method != "оптический" else 0
            }
        else:
            defect_type = None
            defect_probability = np.random.uniform(0, threshold * 0.5)
            defect_location = None

        # Сигнал-шум в зависимости от метода
        signal_quality = {
            "ультразвук": np.random.uniform(0.7, 0.95),
            "вихретоковый": np.random.uniform(0.6, 0.9),
            "оптический": np.random.uniform(0.8, 0.98),
            "термография": np.random.uniform(0.5, 0.85)
        }[method]

        return {
            "has_defect": has_defect,
            "defect_type": defect_type,
            "defect_probability": defect_probability,
            "defect_location": defect_location,
            "signal_quality": signal_quality,
            "inspection_params": params or {},
            "method": method,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_report(self, bottle_id: str, inspection_result: Dict[str, Any]) -> str:
        """Формирование подробного отчета о контроле."""

        verdict = "БРАК" if inspection_result["has_defect"] else "ГОДЕН"

        # Определение критичности дефекта
        if inspection_result["has_defect"]:
            defect_info = self.DEFECT_DATABASE.get(
                inspection_result["defect_type"],
                {"critical": True, "repairable": False}
            )
            critical = "КРИТИЧЕСКИЙ" if defect_info["critical"] else "НЕКРИТИЧЕСКИЙ"
            repairable = "возможно устранение" if defect_info["repairable"] else "устранению не подлежит"
        else:
            critical = "отсутствует"
            repairable = "не требуется"

        # Формирование отчета
        report_lines = [
            "═" * 80,
            f"РЕЗУЛЬТАТ КОНТРОЛЯ КАЧЕСТВА МЕТАЛЛИЧЕСКОЙ БУТЫЛКИ",
            "═" * 80,
            f"Идентификатор: {bottle_id}",
            f"Время контроля: {inspection_result['timestamp']}",
            f"Метод контроля: {inspection_result['method'].upper()}",
            f"Порог чувствительности: {inspection_result['threshold']:.2f}",
            f"Качество сигнала: {inspection_result['signal_quality']:.2%}",
            "",
            "─" * 80,
            "РЕЗУЛЬТАТЫ АНАЛИЗА:",
            "─" * 80,
            f"Статус: {verdict}",
        ]

        if inspection_result["has_defect"]:
            report_lines.extend([
                f"Обнаружен дефект: {inspection_result['defect_type'].upper()}",
                f"Вероятность дефекта: {inspection_result['defect_probability']:.2%}",
                f"Критичность: {critical}",
                f"Рекомендация: {repairable}",
            ])

            if inspection_result["defect_location"]:
                loc = inspection_result["defect_location"]
                report_lines.append(f"Координаты дефекта (усл. ед.): X={loc['x']:.1f}, Y={loc['y']:.1f}")
                if loc['depth'] > 0:
                    report_lines.append(f"Глубина залегания: {loc['depth']:.2f} мм")
        else:
            report_lines.extend([
                "Дефекты не обнаружены",
                "Изделие соответствует требованиям качества",
                "Рекомендация: допустить к дальнейшей обработке"
            ])

        # Добавление статистики и рекомендаций
        report_lines.extend([
            "",
            "─" * 80,
            "РЕКОМЕНДАЦИИ:",
            "─" * 80,
        ])

        if inspection_result["has_defect"]:
            if inspection_result["defect_probability"] > 0.9:
                report_lines.append("🔴 ВЫСОКАЯ ВЕРОЯТНОСТЬ ДЕФЕКТА - ТРЕБУЕТСЯ ПОВТОРНЫЙ КОНТРОЛЬ")
            elif inspection_result["defect_probability"] > 0.7:
                report_lines.append("🟡 СРЕДНЯЯ ВЕРОЯТНОСТЬ ДЕФЕКТА - РЕКОМЕНДУЕТСЯ ДОПОЛНИТЕЛЬНЫЙ КОНТРОЛЬ")
            else:
                report_lines.append("🟢 НИЗКАЯ ВЕРОЯТНОСТЬ ДЕФЕКТА - ИЗДЕЛИЕ МОЖЕТ БЫТЬ ДОПУЩЕНО")

            if inspection_result["signal_quality"] < 0.7:
                report_lines.append("⚠️ НИЗКОЕ КАЧЕСТВО СИГНАЛА - ТРЕБУЕТСЯ КАЛИБРОВКА ОБОРУДОВАНИЯ")
        else:
            report_lines.append("✅ Изделие соответствует нормативным требованиям")

        # Рекомендации по улучшению процесса
        report_lines.extend([
            "",
            "─" * 80,
            "СТАТИСТИЧЕСКИЕ МЕТРИКИ:",
            "─" * 80,
            f"Достоверность контроля: {inspection_result['signal_quality'] * (1 - inspection_result['threshold'] * 0.2):.2%}",
            f"Рекомендуемая частота калибровки: {self._get_calibration_frequency(inspection_result['method'])}",
            "",
            "═" * 80
        ])

        return "\n".join(report_lines)

    def _get_calibration_frequency(self, method: str) -> str:
        """Определение частоты калибровки оборудования."""
        frequencies = {
            "ультразвук": "каждые 8 часов работы",
            "вихретоковый": "каждые 4 часа работы",
            "оптический": "каждые 24 часа работы",
            "термография": "еженедельно"
        }
        return frequencies.get(method, "согласно регламенту")

    async def _arun(self, bottle_id: str, inspection_method: str = "ультразвук",
                    threshold_defect: float = 0.7, inspection_params: Optional[Dict[str, Any]] = None) -> str:
        """Асинхронная версия для реальных производственных систем."""
        return self._run(bottle_id, inspection_method, threshold_defect, inspection_params)

    def to_langchain_tool(self) -> BaseTool:
        """Конвертация в формат LangChain."""
        return self


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 3: Пример использования (для тестирования)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Создание экземпляра инструмента
    tool = CustomTool()

    # Тест 1: Контроль без дефекта
    print("\n🔍 ТЕСТ 1: Контроль качественной бутылки")
    print("=" * 80)
    result1 = tool.run(
        bottle_id="BTL-2026-001",
        inspection_method="ультразвук",
        threshold_defect=0.7,
        inspection_params={"frequency": 5.0, "amplitude": 1.5}
    )
    print(result1)

    # Тест 2: Контроль с высоким порогом чувствительности
    print("\n\n🔍 ТЕСТ 2: Контроль с высоким порогом чувствительности")
    print("=" * 80)
    result2 = tool.run(
        bottle_id="BTL-2026-002",
        inspection_method="вихретоковый",
        threshold_defect=0.95,
        inspection_params={"frequency": 100, "gain": 20}
    )
    print(result2)

    # Тест 3: Оптический контроль
    print("\n\n🔍 ТЕСТ 3: Оптический контроль")
    print("=" * 80)
    result3 = tool.run(
        bottle_id="BTL-2026-003",
        inspection_method="оптический",
        threshold_defect=0.6,
        inspection_params={"resolution": 1920, "illumination": 500}
    )
    print(result3)
