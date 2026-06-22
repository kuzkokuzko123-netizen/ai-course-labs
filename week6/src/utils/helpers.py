# -*- coding: utf-8 -*-
"""Вспомогательные функции для нейро-символьной системы."""
import logging


def setup_logging(level: str = "INFO"):
    """Настройка логирования."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
