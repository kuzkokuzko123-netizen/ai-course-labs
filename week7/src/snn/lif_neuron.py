# -*- coding: utf-8 -*-
"""
Реализация LIF-нейрона (Leaky Integrate-and-Fire)
Лабораторная работа №7
Дисциплина: Искусственный интеллект
Автор: Мыльников Александр Русланович
Группа: ФИТ-221
Дата: 2026
"""

import torch
import snntorch as snn
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class LIFNeuron:
    """
    LIF-нейрон (Leaky Integrate-and-Fire) на snntorch.

    Атрибуты:
        beta: Параметр утечки (0-1), ближе к 1 = медленнее утечка
        threshold: Порог срабатывания нейрона
    """

    def __init__(
        self,
        beta: float = 0.95,
        threshold: float = 1.0
    ):
        """
        Инициализация LIF-нейрона.

        Args:
            beta: Параметр утечки (decay rate)
            threshold: Порог генерации спайка
        """
        self.beta = beta
        self.threshold = threshold

        # LIF-нейрон snntorch (используется суррогат по умолчанию)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            learn_beta=False,
            learn_threshold=False
        )

        logger.info(f"LIF-нейрон инициализирован: beta={beta}, threshold={threshold}")

    def simulate(
        self,
        input_current: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Симуляция работы нейрона.

        Args:
            input_current: Входной ток (размер: [num_steps] или [batch, num_steps])
            num_steps: Количество шагов симуляции

        Returns:
            Tuple[spikes, mem]: Спайки и мембранный потенциал
        """
        # Сброс мембранного потенциала
        self.lif.reset_mem()

        # Определение количества шагов
        if num_steps is None:
            if input_current.dim() == 1:
                num_steps = len(input_current)
            else:
                num_steps = input_current.shape[1]

        # Убедимся, что input_current имеет правильную размерность
        if input_current.dim() == 1:
            input_current = input_current.unsqueeze(0)  # [1, num_steps]

        # Хранение результатов
        spikes_record = []
        mem_record = []

        # Симуляция по шагам
        for step in range(num_steps):
            spike, mem = self.lif(
                input_current[:, step] if input_current.dim() > 1 else input_current[step]
            )
            spikes_record.append(spike)
            mem_record.append(mem)

        # Конвертация в тензоры
        spikes = torch.stack(spikes_record)
        mem = torch.stack(mem_record)

        logger.debug(f"Симуляция завершена: {spikes.shape} спайков")
        return spikes, mem

    def visualize(
        self,
        input_current: torch.Tensor,
        spikes: torch.Tensor,
        mem: torch.Tensor,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализация активности нейрона.

        Args:
            input_current: Входной ток
            spikes: Спайки
            mem: Мембранный потенциал
            save_path: Путь для сохранения графика

        Returns:
            plt.Figure: Фигура matplotlib
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        num_steps = len(input_current)
        time_axis = range(num_steps)

        # График 1: Входной ток
        axes[0].plot(time_axis, input_current.cpu().numpy(), 'b-', linewidth=2)
        axes[0].set_ylabel('Входной ток', fontsize=10)
        axes[0].set_title('Активность LIF-нейрона', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # График 2: Мембранный потенциал
        axes[1].plot(time_axis, mem.cpu().numpy(), 'g-', linewidth=2, label='Мембрана')
        axes[1].axhline(y=self.threshold, color='r', linestyle='--', label='Порог')
        axes[1].set_ylabel('Потенциал (мВ)', fontsize=10)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # График 3: Спайки
        axes[2].scatter(
            time_axis,
            spikes.cpu().numpy() * 1.1,
            c='black',
            marker='|',
            s=100,
            label='Спайки'
        )
        axes[2].set_ylabel('Спайк', fontsize=10)
        axes[2].set_xlabel('Время (шаг)', fontsize=10)
        axes[2].set_ylim(-0.1, 1.5)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"График сохранён: {save_path}")

        return fig

# Точка входа для тестирования
if __name__ == "__main__":
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №7")
    print("ТЕСТИРОВАНИЕ LIF-НЕЙРОНА")
    print("=" * 80)

    # Инициализация нейрона
    neuron = LIFNeuron(beta=0.95, threshold=1.0)

    # Создание тестового входного тока
    num_steps = 100
    input_current = torch.zeros(num_steps)

    # Добавление стимулов в разные моменты времени
    input_current[10:15] = 2.0   # Стимул 1
    input_current[40:50] = 1.5   # Стимул 2
    input_current[70:80] = 2.5   # Стимул 3

    # Симуляция
    print(f"\nЗапуск симуляции ({num_steps} шагов)...")
    spikes, mem = neuron.simulate(input_current)

    # Статистика
    total_spikes = spikes.sum().item()
    spike_rate = total_spikes / num_steps * 100

    print(f"\nРезультаты симуляции:")
    print(f"  • Всего спайков: {total_spikes}")
    print(f"  • Частота спайков: {spike_rate:.2f}%")
    print(f"  • Максимальный потенциал: {mem.max().item():.3f}")
    print(f"  • Минимальный потенциал: {mem.min().item():.3f}")

    # Визуализация
    print("\nГенерация графика активности...")
    fig = neuron.visualize(input_current, spikes, mem, save_path="./lif_neuron_activity.png")
    print("График сохранён: lif_neuron_activity.png")

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)