# -*- coding: utf-8 -*-
"""
Модуль загрузки документов из различных источников
Лабораторная работа №5
Дисциплина: Искусственный интеллект

Автор: Мыльников Александр Русланович
Группа: ФИТ-221
Дата: 2026
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Загрузчик документов из различных форматов.
    Поддерживаемые форматы: PDF, TXT, MD
    """

    def __init__(self, source_directory: str = None):
        """
        Инициализация загрузчика.

        Args:
            source_directory: Директория с документами.
                             Если не указана, вычисляется как
                             week5/data/documents относительно расположения этого файла.
        """
        if source_directory is None:
            current_file = Path(__file__).resolve()
            week5_dir = current_file.parents[2]
            source_directory = week5_dir / "data" / "documents"
        else:
            source_directory = Path(source_directory)

        self.source_directory = source_directory
        self.supported_extensions = [".pdf", ".txt", ".md"]

        if not self.source_directory.exists():
            logger.warning(f"Директория {source_directory} не существует, создаю...")
            self.source_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"DocumentLoader инициализирован: {self.source_directory}")

    def load_document(self, file_path: str) -> List[Document]:
        """Загрузка одного документа по пути."""
        path = Path(file_path)
        ext = path.suffix.lower()
        logger.info(f"Загрузка документа: {file_path}")

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            logger.warning(f"Неподдерживаемый формат: {ext}")
            return []

        documents = loader.load()
        for doc in documents:
            doc.metadata['source'] = str(file_path)
            doc.metadata['source_filename'] = path.name
            doc.metadata['file_type'] = ext[1:]

        return documents

    def load_directory(self, pattern: str = "*") -> List[Document]:
        """Загрузка всех документов из директории."""
        all_documents = []
        for ext in self.supported_extensions:
            file_pattern = f"{pattern}{ext}"
            files = list(self.source_directory.glob(file_pattern))
            logger.info(f"Найдено {len(files)} файлов с расширением {ext}")
            for file_path in files:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Пропущен файл {file_path}: {e}")
        logger.info(f"Всего загружено {len(all_documents)} документов")
        return all_documents

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по документам."""
        stats = {"directory": str(self.source_directory), "files": {}, "total_size_bytes": 0}
        for ext in self.supported_extensions:
            files = list(self.source_directory.glob(f"*{ext}"))
            stats["files"][ext] = len(files)
            stats["total_size_bytes"] += sum(f.stat().st_size for f in files)
        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        return stats


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 80)
    print("ТЕСТИРОВАНИЕ DOCUMENT LOADER")
    print("=" * 80)

    loader = DocumentLoader()
    stats = loader.get_statistics()
    print(f"\nСтатистика директории:")
    print(f"  Директория: {stats['directory']}")
    print(f"  Общий размер: {stats['total_size_mb']} MB")
    print(f"  Файлы по типам: {stats['files']}")

    documents = loader.load_directory()
    print(f"\nЗагружено документов: {len(documents)}")

    if documents:
        print(f"\nПример первого документа:")
        print(f"  Источник: {documents[0].metadata.get('source')}")
        print(f"  Размер: {len(documents[0].page_content)} символов")
        print(f"  Предпросмотр: {documents[0].page_content[:200]}...")