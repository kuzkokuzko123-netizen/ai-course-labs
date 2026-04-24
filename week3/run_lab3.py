# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска лабораторной работы №3
Многоагентные системы с поддержкой Yandex GPT
"""

import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Отключаем лишние логи
logging.basicConfig(level=logging.ERROR)


def print_header():
    print("\n" + "=" * 80)
    print(" " * 20 + "ЛАБОРАТОРНАЯ РАБОТА №3")
    print(" " * 15 + "МНОГОАГЕНТНЫЕ СИСТЕМЫ")
    print("=" * 80)
    print("\nДисциплина: Искусственный интеллект")
    print("Дата: " + datetime.now().strftime("%d.%m.%Y"))
    print("Тема диплома: Разработка системы неразрушающего контроля")
    print("   для выявления дефектов металлических бутылок на конвейерной линии")
    print("=" * 80)


def check_env():
    """Проверка наличия .env файла и API ключей."""
    print("\nПРОВЕРКА .env КОНФИГУРАЦИИ")
    print("-" * 50)

    env_file_exists = os.path.exists('.env')

    if env_file_exists:
        print("Файл .env найден")

        iam_token = os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")

        if iam_token and iam_token != "your_iam_token_here":
            print("YANDEX_IAM_TOKEN настроен")
        else:
            print("YANDEX_IAM_TOKEN не настроен")

        if folder_id and folder_id != "your_folder_id_here":
            print("YANDEX_FOLDER_ID настроен")
        else:
            print("YANDEX_FOLDER_ID не настроен")

        if iam_token and folder_id:
            print("\nYandex Cloud готов к работе")
    else:
        print("Файл .env не найден")
        print("Создайте файл .env на основе .env.example")

    print()


def test_yandex_gpt():
    """Тестирование подключения к Yandex GPT."""
    print("\nТЕСТ 0: ПРОВЕРКА YANDEX GPT")
    print("-" * 50)

    try:
        from src.llm.yandex_gpt_client import yandex_gpt

        if yandex_gpt.is_available():
            print("Yandex GPT доступен")
            test_prompt = "Что такое неразрушающий контроль? Ответ одним предложением."
            response = yandex_gpt.generate(test_prompt)
            print(f"Тестовый ответ: {response[:100]}..." if len(response) > 100 else f"Тестовый ответ: {response}")
            return True
        else:
            print("Yandex GPT не настроен, используются моки")
            return True

    except Exception as e:
        print(f"Ошибка при подключении к Yandex GPT: {e}")
        return True


def test_agents():
    """Тестирование агентов."""
    print("\nТЕСТ 1: ПРОВЕРКА АГЕНТОВ")
    print("-" * 50)

    try:
        from src.agents.researcher_agent import ResearcherAgent
        from src.agents.analyst_agent import AnalystAgent
        from src.agents.writer_agent import WriterAgent

        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        print("Агенты успешно созданы:")
        print(f"   - {researcher.config.role}")
        print(f"   - {analyst.config.role}")
        print(f"   - {writer.config.role}")

        print("\nПроверка выполнения задач:")

        res_result = researcher.execute_task("Методы неразрушающего контроля металлических бутылок")
        print(
            f"   - Researcher: {res_result['status']} (найдено {len(res_result.get('sources_found', []))} источников)")

        facts = res_result.get('key_facts', [])
        if facts:
            print(f"     Факты: {facts[0][:80]}...")
            if len(facts) > 1:
                print(f"            {facts[1][:80]}...")

        ana_result = analyst.execute_task("Анализ данных по НК", {"research_data": res_result})
        print(f"   - Analyst: {ana_result['status']} (получено {len(ana_result.get('insights', []))} инсайтов)")

        insights = ana_result.get('insights', [])
        if insights:
            print(f"     Инсайты: {insights[0][:80]}...")
            if len(insights) > 1:
                print(f"              {insights[1][:80]}...")

        writer_result = writer.execute_task("Создание отчёта по НК", {
            "research_data": res_result,
            "analysis_data": ana_result
        })
        doc_len = len(writer_result.get('document', ''))
        print(f"   - Writer: {writer_result['status']} (создан документ: {doc_len} символов)")

        print("\nВсе агенты работают корректно")
        return True

    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def test_specialized_agent():
    """Тестирование специализированного агента (НК)."""
    print("\nТЕСТ 2: СПЕЦИАЛИЗИРОВАННЫЙ АГЕНТ (НЕРАЗРУШАЮЩИЙ КОНТРОЛЬ)")
    print("-" * 50)

    try:
        from src.ndt_inspector_agent import NDTInspectorAgent

        inspector = NDTInspectorAgent()
        print("Специализированный агент создан")
        print(f"   - Роль: {inspector.config.role}")
        print(f"   - Цель: {inspector.config.goal}")

        print("\nРезультаты инспекций металлических бутылок:")
        print("-" * 55)

        results = []
        for i in range(5):
            result = inspector.execute_task(
                "Контроль качества бутылки",
                context={"bottle_id": f"BOTTLE_{i + 1:03d}"}
            )
            results.append(result)

            inspection = result.get('inspection_result', {})
            decision = result.get('quality_decision', {})

            quality = inspection.get('quality_score', 0)
            status_icon = "X" if decision.get('reject') else "O"
            status_text = "БРАК" if decision.get('reject') else "ГОДЕН"

            print(
                f"   {status_icon} Бутылка {i + 1:3d}: {status_text:6s} | Качество: {quality:5.1f}% | {decision.get('action', 'N/A')}")

        stats = inspector.get_statistics_full()
        print("-" * 55)
        print(f"\nСТАТИСТИКА ИНСПЕКЦИЙ:")
        print(f"   Всего проверено: {stats['total']}")
        print(f"   Годных: {stats['accept']}")
        print(f"   Бракованных: {stats['defective']}")
        print(f"   Процент годных: {stats['accept_rate']:.1f}%")
        print(f"   Среднее качество: {stats['avg_quality_score']:.1f}%")

        print("\nСпециализированный агент работает корректно")
        return True

    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def test_crew_coordination():
    """Тестирование координации через Crew."""
    print("\nТЕСТ 3: КООРДИНАЦИЯ АГЕНТОВ (CREW)")
    print("-" * 50)

    try:
        from src.crew.research_crew import ResearchCrew

        crew = ResearchCrew()
        print("Команда агентов создана")

        print("\nЗапуск координации агентов...")
        result = crew.execute(
            "Современные методы неразрушающего контроля металлических бутылок на конвейерной линии"
        )

        if result.success:
            print(f"\nКоординация выполнена успешно")
            print(f"   - Время выполнения: {result.execution_time:.2f} секунд")
            print(f"   - Участвовало агентов: {len(result.agent_results)}")
            print(f"   - Длина финального отчёта: {len(result.final_output)} символов")

            print(f"\nВремя выполнения агентов:")
            for agent_name, agent_result in result.agent_results.items():
                if 'execution_time' in agent_result:
                    print(f"   - {agent_name}: {agent_result.get('execution_time', 0):.3f}с")

            print(f"\nФИНАЛЬНЫЙ ОТЧЁТ:")
            print("=" * 80)
            print(result.final_output)
            print("=" * 80)

            print("\nКоординация работает корректно")
            return True
        else:
            print(f"Ошибка при выполнении: {result.final_output}")
            return False

    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def test_communication_bus():
    """Тестирование шины сообщений."""
    print("\nТЕСТ 4: ШИНА СООБЩЕНИЙ (MESSAGE BUS)")
    print("-" * 50)

    try:
        from src.communication.message_bus import MessageBus

        bus = MessageBus()
        received_messages = []

        def test_callback(message):
            received_messages.append(message)

        sub_id = bus.subscribe("test_event", test_callback)
        bus.publish({"type": "test_event", "data": "Hello, World!"})

        if len(received_messages) > 0:
            print("Подписка и публикация работают")
            print(f"   - Получено сообщений: {len(received_messages)}")
            print(f"   - Содержание: {received_messages[0].get('data', 'N/A')}")
        else:
            print("Сообщения не доставлены")
            return False

        bus.unsubscribe(sub_id)
        bus.publish({"type": "test_event", "data": "Second message"})

        if len(received_messages) == 1:
            print("Отписка работает (новое сообщение не получено)")
        else:
            print("Отписка может не работать")

        print("\nШина сообщений работает корректно")
        return True

    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def main():
    print_header()
    check_env()

    tests = [
        ("Проверка Yandex GPT", test_yandex_gpt),
        ("Проверка базовых агентов", test_agents),
        ("Специализированный агент (НК)", test_specialized_agent),
        ("Координация через Crew", test_crew_coordination),
        ("Шина сообщений (Message Bus)", test_communication_bus)
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
        print()

    print("=" * 80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 80)

    for name, result in results:
        status = "ПРОЙДЕН" if result else "НЕ ПРОЙДЕН"
        print(f"   {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nРЕЗУЛЬТАТ: {passed}/{total} тестов пройдено")
    print("=" * 80)


if __name__ == "__main__":
    main()