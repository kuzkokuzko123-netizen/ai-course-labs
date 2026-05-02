# -*- coding: utf-8 -*-
"""
Сравнение SNN vs ANN
Лабораторная работа №7
Автор: Мыльников Александр Русланович
Группа: ФИТ-221

Специальность: 02.03.02 Фундаментальная информатика и информационные технологии
Тема диплома: Обнаружение дефектных изделий на производственной линии
               с использованием алгоритмов компьютерного зрения
"""


import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from snn_classifier import SNNClassifier   # теперь работает всегда

logger = logging.getLogger(__name__)



@dataclass
class ComparisonResult:
    model_type: str
    accuracy: float
    inference_time_ms: float
    estimated_energy_mj: float
    num_parameters: int
    spike_count: int = 0


class ANNClassifier(nn.Module):
    """Обычная ANN для сравнения."""

    def __init__(self, num_inputs=784, num_hidden=100, num_outputs=10):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Если на вход подаётся изображение [batch, 1, 28, 28] – преобразуем в плоский вектор
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class SNNvsANNComparator:
    """
    Сравнение импульсной (SNN) и традиционной (ANN) нейронных сетей.
    """

    def __init__(
            self,
            num_inputs: int = 784,
            num_hidden: int = 100,
            num_outputs: int = 10,
            beta: float = 0.95
    ):
        self.ann = ANNClassifier(num_inputs, num_hidden, num_outputs)
        self.snn = SNNClassifier(num_inputs, num_hidden, num_outputs, beta)

    def count_parameters(self, model: nn.Module) -> int:
        """Подсчёт параметров модели."""
        return sum(p.numel() for p in model.parameters())

    def measure_inference_time(
            self,
            model: nn.Module,
            data: torch.Tensor,
            num_runs: int = 10
    ) -> float:
        """Измерение времени инференса."""
        model.eval()
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                if isinstance(model, ANNClassifier):
                    _ = model(data)
                else:
                    _ = model.predict(data, num_steps=25)
                end = time.time()
                times.append((end - start) * 1000)  # мс
        return sum(times) / len(times)

    def estimate_energy(
            self,
            inference_time_ms: float,
            model_type: str
    ) -> float:
        """
        Оценка энергопотребления.

        Приблизительные значения:
          • CPU: ~50 Вт
          • GPU: ~150 Вт
          • Neuromorphic: ~0.05 Вт
        """
        power_watts = {
            "ANN": 50,  # CPU
            "SNN": 0.05  # Neuromorphic (оценка)
        }
        power = power_watts.get(model_type, 50)
        energy_mj = (power * inference_time_ms / 1000) * 1000  # мДж
        return energy_mj

    def compare(
            self,
            test_data: torch.Tensor,
            test_targets: torch.Tensor
    ) -> Tuple[ComparisonResult, ComparisonResult]:
        """Полное сравнение моделей."""
        logger.info("Начало сравнения SNN vs ANN")

        # ANN EVALUATION
        self.ann.eval()
        with torch.no_grad():
            ann_output = self.ann(test_data)
            ann_predictions = ann_output.argmax(dim=1)
            ann_accuracy = (ann_predictions == test_targets).float().mean().item()
            ann_time = self.measure_inference_time(self.ann, test_data)
            ann_energy = self.estimate_energy(ann_time, "ANN")
            ann_params = self.count_parameters(self.ann)

        ann_result = ComparisonResult(
            model_type="ANN",
            accuracy=ann_accuracy,
            inference_time_ms=ann_time,
            estimated_energy_mj=ann_energy,
            num_parameters=ann_params
        )

        # SNN EVALUATION
        self.snn.eval()
        with torch.no_grad():
            snn_predictions = self.snn.predict(test_data, num_steps=25)
            snn_accuracy = (snn_predictions == test_targets).float().mean().item()

            # Подсчёт спайков
            _, _, spike_record = self.snn(test_data, num_steps=25)
            spike_count = spike_record.sum().item()

            snn_time = self.measure_inference_time(self.snn, test_data)
            snn_energy = self.estimate_energy(snn_time, "SNN")
            snn_params = self.count_parameters(self.snn)

        snn_result = ComparisonResult(
            model_type="SNN",
            accuracy=snn_accuracy,
            inference_time_ms=snn_time,
            estimated_energy_mj=snn_energy,
            num_parameters=snn_params,
            spike_count=spike_count
        )

        logger.info(f"Сравнение завершено: ANN={ann_accuracy:.4f}, SNN={snn_accuracy:.4f}")
        return ann_result, snn_result

    def print_comparison(
            self,
            ann_result: ComparisonResult,
            snn_result: ComparisonResult
    ) -> None:
        """Вывод сравнения в консоль."""
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ SNN vs ANN")
        print("=" * 80)

        print(f"\n{'Метрика':<30} {'ANN':<20} {'SNN':<20} {'Улучшение':<15}")
        print("-" * 80)

        metrics = [
            ("Точность", f"{ann_result.accuracy:.4f}", f"{snn_result.accuracy:.4f}",
             f"{(snn_result.accuracy - ann_result.accuracy) * 100:+.2f}%"),
            ("Время инференса (мс)", f"{ann_result.inference_time_ms:.2f}",
             f"{snn_result.inference_time_ms:.2f}",
             f"{ann_result.inference_time_ms / snn_result.inference_time_ms:.1f}x"),
            ("Энергопотребление (мДж)", f"{ann_result.estimated_energy_mj:.4f}",
             f"{snn_result.estimated_energy_mj:.4f}",
             f"{ann_result.estimated_energy_mj / snn_result.estimated_energy_mj:.0f}x"),
            ("Параметры", f"{ann_result.num_parameters:,}",
             f"{snn_result.num_parameters:,}",
             f"{ann_result.num_parameters / snn_result.num_parameters:.1f}x"),
        ]

        if snn_result.spike_count > 0:
            metrics.append(("Спайки", "N/A", f"{snn_result.spike_count:,}", "—"))

        for name, ann_val, snn_val, improvement in metrics:
            print(f"{name:<30} {ann_val:<20} {snn_val:<20} {improvement:<15}")

        print("=" * 80)

        # Выводы
        print("\nВЫВОДЫ:")
        print("  • SNN обеспечивают значительную экономию энергии (100-1000x)")
        print("  • Точность SNN может быть немного ниже (1-5%)")
        print("  • SNN идеальны для edge-устройств с ограниченным питанием")
        print("  • ANN лучше для задач с максимальной точностью")
        print("=" * 80)


# Точка входа для тестирования
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("=" * 80)
    print("СРАВНЕНИЕ SNN vs ANN")
    print("=" * 80)

    # Загрузка тестовых данных
    print("\nЗагрузка данных...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    test_data, test_targets = next(iter(test_loader))

    # Создание компаратора
    comparator = SNNvsANNComparator(
        num_inputs=784,
        num_hidden=100,
        num_outputs=10
    )

    print("\nВыполнение сравнения...")
    ann_result, snn_result = comparator.compare(test_data, test_targets)

    # Вывод результатов
    comparator.print_comparison(ann_result, snn_result)

    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
