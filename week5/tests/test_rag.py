# -*- coding: utf-8 -*-
"""
Модульное тестирование RAG-системы
Лабораторная работа №5
Автор: Мыльников Александр Русланович
Группа: ФИТ-221
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pytest

# Добавляем путь к src/rag для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "rag"))

from document_loader import DocumentLoader
from chunking import ChunkingStrategy
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline


# Фикстура для временной директории с тестовыми документами
@pytest.fixture
def temp_docs_dir():
    dirpath = tempfile.mkdtemp()
    # Создаём тестовые файлы
    (Path(dirpath) / "test1.txt").write_text(
        "Это тестовый документ о компьютерном зрении. "
        "Обнаружение дефектов на металлических бутылках. "
        "Используйте YOLO для детекции царапин.",
        encoding="utf-8"
    )
    (Path(dirpath) / "test2.md").write_text(
        "# Методы неразрушающего контроля\n"
        "Для вмятин применяйте U-Net.\n"
        "Предобработка: CLAHE, нормализация.",
        encoding="utf-8"
    )
    yield dirpath
    shutil.rmtree(dirpath)


# Фикстура для временной папки ChromaDB
@pytest.fixture
def temp_chroma_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


# 1. Тест загрузчика документов
def test_document_loader(temp_docs_dir):
    loader = DocumentLoader(source_directory=temp_docs_dir)
    docs = loader.load_directory()
    assert len(docs) == 2
    # Проверяем метаданные
    sources = [doc.metadata.get('source', '') for doc in docs]
    assert any("test1.txt" in s for s in sources)
    assert any("test2.md" in s for s in sources)
    # Проверяем статистику
    stats = loader.get_statistics()
    assert stats['files']['.txt'] == 1
    assert stats['files']['.md'] == 1


# 2. Тест чанкинга
def test_chunking(temp_docs_dir):
    loader = DocumentLoader(source_directory=temp_docs_dir)
    docs = loader.load_directory()
    chunks = ChunkingStrategy.split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2  # каждый документ должен разбиться на >=1 чанка
    stats = ChunkingStrategy.get_statistics(chunks)
    assert stats['count'] == len(chunks)
    assert stats['min_size'] > 0
    assert stats['max_size'] <= 120  # chunk_size=100 + overlap
    # Проверяем наличие метаданных чанков
    for chunk in chunks:
        assert 'chunk_id' in chunk.metadata
        assert 'total_chunks' in chunk.metadata


# 3. Тест векторного хранилища (без реальной модели эмбеддингов – используем заглушку)
# В реальном тесте лучше мокать эмбеддинги, но здесь пока просто проверяем создание коллекции.
# Для CI можно использовать небольшую модель, но для примера пропустим тяжёлый тест.
@pytest.mark.skip(reason="Требует загрузки embedding модели (долго)")
def test_vector_store_integration(temp_chroma_dir, temp_docs_dir):
    loader = DocumentLoader(source_directory=temp_docs_dir)
    docs = loader.load_directory()
    chunks = ChunkingStrategy.split_documents(docs, chunk_size=200, chunk_overlap=20)
    store = VectorStoreManager(
        persist_directory=temp_chroma_dir,
        collection_name="test_collection",
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
    )
    store.add_documents(chunks)
    assert store.get_statistics()['total_documents'] == len(chunks)
    # Поиск
    results = store.search_with_scores("обнаружение царапин", k=2)
    assert len(results) == 2
    assert results[0]['similarity_score'] is not None


# 4. Тест RAG-пайплайна (без LLM)
def test_rag_pipeline_without_llm(temp_chroma_dir, temp_docs_dir):
    # Сначала добавляем документы в хранилище (через реальный эмбеддинг – можно заглушить для скорости)
    # Для быстроты пропустим, если модель не загружена. Рекомендуется в CI вешать маркер.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
    except ImportError:
        pytest.skip("HuggingFaceEmbeddings недоступен")

    # Создаём векторное хранилище
    from langchain_chroma import Chroma
    vectorstore = Chroma.from_documents(
        documents=ChunkingStrategy.split_documents(
            DocumentLoader(source_directory=temp_docs_dir).load_directory(),
            chunk_size=200
        ),
        embedding=embeddings,
        persist_directory=temp_chroma_dir,
        collection_name="test_pipeline"
    )

    # Используем менеджер, чтобы работал с search_with_scores
    class FakeVectorStore:
        def __init__(self, vs):
            self.vectorstore = vs

        def search_with_scores(self, query, k):
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            return [{"content": d.page_content, "metadata": d.metadata, "similarity_score": 1 - s}
                    for d, s in docs_with_scores]

    rag = RAGPipeline(FakeVectorStore(vectorstore), llm=None, top_k=2)
    result = rag.query("Как обнаружить вмятины?")
    assert result['success'] is True
    assert len(result['sources']) == 2
    assert "U-Net" in result['answer'] or "вмятин" in result['answer']

    # Fallback без LLM должен содержать найденные фрагменты
    assert "Источник" in result['answer'] or "test1.txt" in result['answer']


# 5. Тест статистики
def test_statistics(temp_docs_dir):
    loader = DocumentLoader(source_directory=temp_docs_dir)
    docs = loader.load_directory()
    chunks = ChunkingStrategy.split_documents(docs, chunk_size=150, chunk_overlap=30)
    stats = ChunkingStrategy.get_statistics(chunks)
    assert stats['total_characters'] > 0
    assert stats['avg_size'] > 0
    assert stats['max_size'] <= 180  # chunk_size + overlap ≤ 180


# Запуск тестов (если файл выполняется напрямую)
if __name__ == "__main__":
    pytest.main([__file__, "-v"])