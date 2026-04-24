# -*- coding: utf-8 -*
"""
Семантическая память на основе ChromaDB
Лабораторная работа №2
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Семантическая память для долгосрочного хранения знаний агента.
    Использует векторные embeddings для семантического поиска.
    """

    def __init__(
            self,
            collection_name: str = "agent_knowledge",
            persist_directory: str = "./chroma_db",
            embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Инициализация семантической памяти.

        Args:
            collection_name: Имя коллекции
            persist_directory: Путь для сохранения
            embedding_model: Модель для embeddings
        """
        # Создание директории для хранения
        os.makedirs(persist_directory, exist_ok=True)

        try:
            # Инициализация ChromaDB с сохранением на диск
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )

            # Попытка использовать embedding функцию
            try:
                import chromadb.utils.embedding_functions as embedding_functions
                from sentence_transformers import SentenceTransformer

                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model
                )
                self.use_embeddings = True
            except Exception as e:
                logger.warning(f"Embedding function not available: {e}. Using default.")
                self.embedding_function = None
                self.use_embeddings = False

            # Создание или получение коллекции
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function if self.use_embeddings else None
                )
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function if self.use_embeddings else None,
                    metadata={"hnsw:space": "cosine"}
                )

            logger.info(f"Семантическая память инициализирована: {collection_name}")

        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}")
            # Fallback - использование in-memory клиента
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.use_embeddings = False
            logger.warning("Используется in-memory ChromaDB")

    def add_document(
            self,
            content: str,
            metadata: Optional[Dict] = None,
            doc_id: Optional[str] = None
    ) -> str:
        """
        Добавление документа в память.

        Args:
            content: Текст документа
            metadata: Метаданные (источник, дата, теги)
            doc_id: Уникальный ID

        Returns:
            str: ID документа
        """
        doc_id = doc_id or str(uuid.uuid4())
        meta = metadata or {}
        meta["created_at"] = datetime.now().isoformat()

        try:
            if self.use_embeddings:
                self.collection.add(
                    documents=[content],
                    metadatas=[meta],
                    ids=[doc_id]
                )
            else:
                # Без embeddings, просто сохраняем как текст
                self.collection.add(
                    documents=[content],
                    metadatas=[meta],
                    ids=[doc_id]
                )

            logger.debug(f"Документ добавлен: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Ошибка добавления документа: {e}")
            return ""

    def search_knowledge(
            self,
            query: str,
            k: int = 5,
            filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Поиск релевантных знаний.

        Args:
            query: Поисковый запрос
            k: Количество результатов
            filter_metadata: Фильтр по метаданным

        Returns:
            List[Dict]: Найденные документы
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )

            formatted = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    similarity = 1.0
                    if results.get("distances") and results["distances"][0]:
                        similarity = 1 - results["distances"][0][i]

                    formatted.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity": similarity
                    })

            logger.debug(f"Найдено {len(formatted)} документов")
            return formatted

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Удаление документа по ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Документ удалён: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления: {e}")
            return False

    def get_stats(self) -> Dict:
        """Статистика памяти."""
        try:
            count = self.collection.count()
        except:
            count = 0

        return {
            "collection": self.collection.name,
            "documents": count,
            "use_embeddings": self.use_embeddings
        }

    def clear_collection(self):
        """Очистка коллекции."""
        try:
            # Получаем все ID
            all_docs = self.collection.get()
            if all_docs and all_docs.get("ids"):
                self.collection.delete(ids=all_docs["ids"])
                logger.info(f"Удалено {len(all_docs['ids'])} документов")
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")