# -*- coding: utf-8 -*-
"""
Утилиты визуализации спайков и мембранных потенциалов.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_spike_raster(spike_record: torch.Tensor, title: str = "Spike Raster", save_path: str = None):
    """
    Растровая диаграмма спайков.

    Args:
        spike_record: Тензор спайков [time_steps, batch, neurons]
    """
    time_steps, batch_size, num_neurons = spike_record.shape
    spikes_np = spike_record.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    for t in range(time_steps):
        for n in range(num_neurons):
            if spikes_np[t, :, n].any():
                ax.scatter([t]*spikes_np[t, :, n].sum(),
                           np.where(spikes_np[t, :, n])[0] + n*batch_size*0.1,
                           s=1, c='black', marker='|')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Neuron index (grouped)')
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig