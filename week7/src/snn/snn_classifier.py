# -*- coding: utf-8 -*-
"""
SNN классификатор на snntorch
Лабораторная работа №7
Автор: Мыльников Александр Русланович
Группа: ФИТ-221
"""

import torch
import torch.nn as nn
import snntorch as snn
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SNNClassifier(nn.Module):
    """
    Импульсная нейронная сеть для классификации.

    Атрибуты:
        num_inputs: Количество входных нейронов
        num_hidden: Количество скрытых нейронов
        num_outputs: Количество выходных классов
        beta: Параметр утечки для LIF-нейронов
    """

    def __init__(
            self,
            num_inputs: int = 784,
            num_hidden: int = 100,
            num_outputs: int = 10,
            beta: float = 0.95
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.beta = beta

        # Слои сети
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=beta)

        logger.info(f"SNNClassifier инициализирован: {num_inputs}-{num_hidden}-{num_outputs}")

    def forward(
            self,
            x: torch.Tensor,
            num_steps: int = 25
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямое распространение через SNN.

        Args:
            x: Входные данные [batch, ...] – будет преобразовано в [batch, num_inputs]
            num_steps: Количество временных шагов

        Returns:
            Tuple[spk3, mem3, spike_record]: Выходные спайки, потенциал, история
        """
        # Приводим к плоскому вектору, если нужно
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Инициализация состояний (сброс мембран)
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()

        # Хранение спайков на выходе
        spike_record = []

        # Симуляция по временным шагам
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3)
            spike_record.append(spk3)

        # Стек спайков по времени
        spike_record = torch.stack(spike_record)
        return spk3, mem3, spike_record

    def predict(
            self,
            x: torch.Tensor,
            num_steps: int = 25
    ) -> torch.Tensor:
        """
        Предсказание класса.

        Args:
            x: Входные данные (могут быть изображениями)
            num_steps: Количество шагов

        Returns:
            torch.Tensor: Предсказанные классы
        """
        self.eval()
        with torch.no_grad():
            _, _, spike_record = self.forward(x, num_steps)
            # Подсчёт спайков для каждого класса
            spike_count = spike_record.sum(dim=0)
            # Класс с максимальным количеством спайков
            predictions = spike_count.argmax(dim=1)
        return predictions

    def forward_with_layer_spikes(self, x: torch.Tensor, num_steps: int = 25):
        """Возвращает спайки каждого слоя для визуализации."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()

        spk1_rec, spk2_rec, spk3_rec = [], [], []
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, _ = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, _ = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3, _ = self.lif3(cur3)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
        # shape: [time_steps, batch, neurons]
        return (torch.stack(spk1_rec), torch.stack(spk2_rec), torch.stack(spk3_rec))


class SNNTrainer:
    """
    Тренер для SNN.

    Атрибуты:
        model: SNN модель
        learning_rate: Скорость обучения
        device: Устройство для вычислений
    """

    def __init__(
            self,
            model: SNNClassifier,
            learning_rate: float = 1e-3,
            device: str = "cpu"
    ):
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.device = device

        # Оптимизатор
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        # Функция потерь (MSE на спайках)
        self.loss_fn = nn.MSELoss()

        logger.info(f"SNNTrainer инициализирован: lr={learning_rate}, device={device}")

    def train_epoch(
            self,
            dataloader: DataLoader,
            num_steps: int = 25
    ) -> Tuple[float, float]:
        """
        Обучение за одну эпоху.

        Args:
            dataloader: Загрузчик данных
            num_steps: Количество временных шагов

        Returns:
            Tuple[avg_loss, accuracy]: Средняя потеря и точность
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Обучение")
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # One-hot encoding для targets
            targets_onehot = nn.functional.one_hot(
                targets,
                num_classes=self.model.num_outputs
            ).float()

            # Прямое распространение
            self.optimizer.zero_grad()
            spk3, mem3, _ = self.model(data, num_steps)

            # Вычисление потерь на последнем шаге
            loss = self.loss_fn(mem3, targets_onehot)

            # Обратное распространение
            loss.backward()
            self.optimizer.step()

            # Статистика
            total_loss += loss.item()
            predictions = self.model.predict(data, num_steps)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct / total:.4f}"
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate(
            self,
            dataloader: DataLoader,
            num_steps: int = 25
    ) -> Dict[str, float]:
        """
        Оценка модели.

        Args:
            dataloader: Загрузчик данных
            num_steps: Количество шагов

        Returns:
            Dict: Метрики качества
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                targets_onehot = nn.functional.one_hot(
                    targets,
                    num_classes=self.model.num_outputs
                ).float()

                spk3, mem3, _ = self.model(data, num_steps)
                loss = self.loss_fn(mem3, targets_onehot)

                total_loss += loss.item()
                predictions = self.model.predict(data, num_steps)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total,
            "correct": correct,
            "total": total
        }


# Точка входа для тестирования
if __name__ == "__main__":
    from torchvision import datasets, transforms

    print("=" * 80)
    print("ТЕСТИРОВАНИЕ SNN КЛАССИФИКАТОРА")
    print("=" * 80)

    # Загрузка данных MNIST
    print("\nЗагрузка данных MNIST...")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # Создание модели
    model = SNNClassifier(
        num_inputs=784,
        num_hidden=100,
        num_outputs=10,
        beta=0.95
    )

    # Создание тренера
    trainer = SNNTrainer(
        model=model,
        learning_rate=1e-3,
        device="cpu"
    )

    # Обучение (1 эпоха для теста)
    print("\nОбучение (1 тестовая эпоха)...")
    avg_loss, accuracy = trainer.train_epoch(train_loader, num_steps=25)

    print(f"\nРезультаты обучения:")
    print(f"  - Средняя потеря: {avg_loss:.4f}")
    print(f"  - Точность: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Оценка
    print("\nОценка модели...")
    metrics = trainer.evaluate(train_loader, num_steps=25)

    print(f"\nМетрики:")
    print(f"  - Потеря: {metrics['loss']:.4f}")
    print(f"  - Точность: {metrics['accuracy']:.4f}")
    print(f"  - Правильно: {metrics['correct']}/{metrics['total']}")

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
