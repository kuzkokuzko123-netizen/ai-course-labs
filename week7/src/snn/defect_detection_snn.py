# -*- coding: utf-8 -*-
"""
SNN для обнаружения дефектных изделий на производственной линии
Лабораторная работа №7
Автор: Мыльников Александр Русланович
Группа: ФИТ-221
Специальность: 02.03.02 Фундаментальная информатика и информационные технологии
Тема диплома: Обнаружение дефектных изделий на производственной линии
               с использованием алгоритмов компьютерного зрения
"""

import torch
import torch.nn as nn
import snntorch as snn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class DefectDetectionSNN(nn.Module):
    """
    Импульсная нейронная сеть для обнаружения дефектов на изображениях изделий.

    Архитектура:
        Вход (изображение 64x64) -> LIF -> LIF -> Выход (вероятность дефекта / класс дефекта)

    Применение:
        • Классификация "годен/брак"
        • Многоклассовая детекция типов дефектов (скол, трещина, царапина)
        • Энергоэффективная обработка в реальном времени на Edge-устройствах
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 64,
        input_width: int = 64,
        num_classes: int = 2,
        hidden_features: int = 128,
        beta: float = 0.95,
        num_steps: int = 25
    ):
        super().__init__()
        self.input_size = input_channels * input_height * input_width
        self.num_classes = num_classes
        self.num_steps = num_steps

        # Полносвязные слои + LIF нейроны
        self.fc1 = nn.Linear(self.input_size, hidden_features)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.lif2 = snn.Leaky(beta=beta)

        self.fc3 = nn.Linear(hidden_features // 2, num_classes)
        self.lif3 = snn.Leaky(beta=beta)

        logger.info(f"DefectDetectionSNN: {self.input_size}->{hidden_features}->{hidden_features//2}->{num_classes}")

    def forward(self, x: torch.Tensor, num_steps: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямое распространение.

        Args:
            x: Батч изображений [batch, channels, height, width] или [batch, input_size]
            num_steps: Количество временных шагов

        Returns:
            spk3, mem3, spike_record
        """
        if num_steps is None:
            num_steps = self.num_steps

        # Преобразование в плоский вектор, если требуется
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Сброс состояний (используем reset_mem)
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()

        spike_record = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3)
            spike_record.append(spk3)

        spike_record = torch.stack(spike_record)
        return spk3, mem3, spike_record

    def predict(self, x: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """
        Предсказание класса дефекта.

        Returns:
            Индексы классов [batch]
        """
        self.eval()
        with torch.no_grad():
            _, _, spike_record = self.forward(x, num_steps)
            spike_count = spike_record.sum(dim=0)
            predictions = spike_count.argmax(dim=1)
        return predictions

    def predict_defect_probability(self, x: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """
        Возвращает вероятности дефектов через нормализацию числа спайков.

        Returns:
            Тензор [batch, num_classes] с вероятностями
        """
        self.eval()
        with torch.no_grad():
            _, _, spike_record = self.forward(x, num_steps)
            spike_count = spike_record.sum(dim=0)
            probs = spike_count / (num_steps or self.num_steps)
        return probs


# Пример использования
if __name__ == "__main__":
    model = DefectDetectionSNN(
        input_channels=1,
        input_height=64,
        input_width=64,
        num_classes=2,  # годен / брак
        hidden_features=128,
        beta=0.95
    )

    # Тестовое изображение
    test_image = torch.randn(4, 1, 64, 64)  # батч из 4 изображений

    # Предсказание
    predictions = model.predict(test_image)
    print(f"Размер предсказаний: {predictions.shape}")
    print(f"Предсказанные классы: {predictions}")

    # Вероятности
    probs = model.predict_defect_probability(test_image)
    print(f"Вероятности: {probs}")