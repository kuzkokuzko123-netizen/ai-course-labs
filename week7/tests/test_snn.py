# -*- coding: utf-8 -*-
"""
Тесты для SNN модулей
"""

import torch
import sys
import os

# Добавляем корневую папку week7 в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.snn.lif_neuron import LIFNeuron
from src.snn.snn_classifier import SNNClassifier, SNNTrainer
from src.snn.comparison import SNNvsANNComparator
from src.snn.defect_detection_snn import DefectDetectionSNN

def test_lif_neuron():
    neuron = LIFNeuron(beta=0.95, threshold=1.0)
    input_current = torch.zeros(50)
    input_current[10:15] = 2.0
    spikes, mem = neuron.simulate(input_current)
    assert spikes.shape[0] == 50
    assert mem.shape[0] == 50
    assert spikes.sum() > 0, "Нейрон должен сгенерировать хотя бы один спайк"

def test_snn_classifier():
    model = SNNClassifier(num_inputs=28*28, num_hidden=10, num_outputs=10, beta=0.95)
    x = torch.randn(4, 28*28)
    preds = model.predict(x, num_steps=10)
    assert preds.shape == (4,)

def test_comparison():
    comparator = SNNvsANNComparator(num_inputs=50, num_hidden=20, num_outputs=2, beta=0.9)
    data = torch.randn(10, 50)
    targets = torch.randint(0, 2, (10,))
    ann_res, snn_res = comparator.compare(data, targets)
    assert ann_res.model_type == "ANN"
    assert snn_res.model_type == "SNN"
    assert 0.0 <= ann_res.accuracy <= 1.0

def test_defect_detection_snn():
    model = DefectDetectionSNN(input_channels=1, input_height=32, input_width=32,
                               num_classes=2, hidden_features=64)
    img = torch.randn(2, 1, 32, 32)
    preds = model.predict(img, num_steps=15)
    assert preds.shape == (2,)
    probs = model.predict_defect_probability(img, num_steps=15)
    assert probs.shape == (2, 2)

if __name__ == "__main__":
    test_lif_neuron()
    test_snn_classifier()
    test_comparison()
    test_defect_detection_snn()
    print("Все тесты пройдены!")