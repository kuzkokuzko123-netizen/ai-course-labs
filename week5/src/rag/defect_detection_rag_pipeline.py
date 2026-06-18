# -*- coding: utf-8 -*-
"""
Специализированный RAG для обнаружения дефектов на производственной линии
Лабораторная работа №5
Дисциплина: Искусственный интеллект

Автор: Мыльников Александр Русланович
Группа: ФИТ-221
Специальность: 02.03.02 Фундаментальная информатика и информационные технологии
Тема диплома: Обнаружение дефектных изделий на производственной линии с использованием алгоритмов компьютерного зрения
Дата: 2026
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from rag_pipeline import RAGPipeline
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List


class DefectDetectionRAGPipeline(RAGPipeline):
    """
    RAG-система для вопросов по обнаружению дефектов с использованием компьютерного зрения.

    Особенности:
    • Приоритет техническим документам (методы CV, архитектуры нейросетей)
    • Точное цитирование параметров алгоритмов и метрик качества
    • Классификация типов дефектов и рекомендуемых методов
    • Интеграция с темой дипломной работы
    """

    def __init__(self, vectorstore, llm=None, top_k: int = 5):
        super().__init__(vectorstore, llm, top_k)

        # Специфичный промпт для обнаружения дефектов
        self.rag_prompt = ChatPromptTemplate.from_template("""
Ты — эксперт по компьютерному зрению и системам технического зрения для промышленности.
Используй ТОЛЬКО предоставленный контекст для ответа.
Цитируй конкретные методы, архитектуры нейросетей и числовые параметры, если они указаны.

Если вопрос касается обнаружения дефектов:
• Укажи тип дефекта (царапина, вмятина, загрязнение, деформация и т.д.)
• Назови рекомендуемый алгоритм или архитектуру CNN (U-Net, YOLO, ResNet и др.)
• Приведи ожидаемые метрики качества (precision, recall, mAP), если доступны
• Добавь рекомендации по предобработке изображений (освещение, нормализация)

Контекст из документов:
{context}

Вопрос: {question}

Ответ (с указанием методов CV и параметров):
""")

    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Запрос с доменно-специфичной обработкой для CV дефектов."""
        result = super().query(question, include_sources)

        # Добавление метаданных по теме диплома
        result["domain"] = "defect_detection_cv"
        result[
            "diploma_topic"] = "Обнаружение дефектных изделий на производственной линии с использованием алгоритмов компьютерного зрения"

        # Извлечение упоминаний методов CV
        result["cv_methods_mentioned"] = self._extract_cv_methods(question + " " + result.get("answer", ""))

        return result

    def _extract_cv_methods(self, text: str) -> List[str]:
        """Извлечение названий методов компьютерного зрения из текста."""
        cv_keywords = [
            "YOLO", "SSD", "Faster R-CNN", "U-Net", "ResNet", "EfficientNet",
            "CNN", "сверточная нейросеть", "трансформер", "Vision Transformer",
            "сегментация", "детекция объектов", "классификация изображений",
            "аугментация", "предобработка", "нормализация", "пороговая обработка",
            "Canny", "Sobel", "Hough", "контурный анализ", "морфология"
        ]
        found = [kw for kw in cv_keywords if kw.lower() in text.lower()]
        return list(set(found))

    def get_defect_type_recommendation(self, defect_description: str) -> Dict[str, Any]:
        """
        Рекомендация алгоритма для заданного типа дефекта.
        Использует RAG для поиска релевантных методов.
        """
        query = f"Алгоритмы компьютерного зрения для обнаружения дефекта: {defect_description}"
        return self.query(query)


# Точка входа для тестирования специализированной версии
if __name__ == "__main__":
    from dotenv import load_dotenv
    from vector_store import VectorStoreManager
    import os

    load_dotenv()

    print("=" * 80)
    print("СПЕЦИАЛИЗИРОВАННЫЙ RAG ДЛЯ ОБНАРУЖЕНИЯ ДЕФЕКТОВ")
    print("Тема диплома: Обнаружение дефектных изделий на производственной линии")
    print("=" * 80)

    vectorstore = VectorStoreManager(
        persist_directory="./data/chroma_db",
        collection_name="test_collection"
    )

    # Инициализация LLM (опционально)
    try:
        from langchain_community.llms import YandexGPT

        iam_token = os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        if iam_token and folder_id:
            llm = YandexGPT(iam_token=iam_token, folder_id=folder_id, temperature=0.3)
        else:
            llm = None
    except:
        llm = None

    rag_defect = DefectDetectionRAGPipeline(vectorstore, llm=llm, top_k=4)

    test_queries = [
        "Как обнаружить царапины на металлической поверхности с помощью компьютерного зрения?",
        "Какие нейросетевые архитектуры лучше всего подходят для сегментации дефектов на печатных платах?",
        "Как предобработать изображения для улучшения качества обнаружения вмятин?"
    ]

    for q in test_queries:
        print(f"\n{'=' * 60}\nВопрос: {q}\n")
        res = rag_defect.query(q)
        print(f"Ответ: {res['answer'][:500]}")
        print(f"Упомянутые методы CV: {res['cv_methods_mentioned']}")
        print(f"Время: {res['execution_time']} сек")
