# run your model on a handful of MNIST digits (or saved numpy arrays) and assert top‐1 correctness

import os
import pytest
import torch
import numpy as np
import config
from recognition import DigitClassifier, load_model, predict_batch

def test_model_structure_and_forward():
    model = DigitClassifier()
    # feed a batch of 5 random 28×28 images
    x = torch.randn(5, 1, 28, 28)
    y = model(x)
    assert y.shape == (5, 10)

def test_load_model_file_exists():
    if not os.path.exists(config.MODEL_PATH):
        pytest.skip(f"{config.MODEL_PATH} not found, skipping load_model test")
    model = load_model(config.MODEL_PATH)
    assert isinstance(model, DigitClassifier)
    # ensure eval mode
    assert not model.training

def test_predict_batch_shape_and_range():
    # if model file is missing, skip
    if not os.path.exists(config.MODEL_PATH):
        pytest.skip(f"{config.MODEL_PATH} not found, skipping predict_batch test")
    # load model once
    model = load_model(config.MODEL_PATH)
    # generate 81 blank cells
    dummy_cells = [np.zeros((28, 28), dtype=np.uint8) for _ in range(81)]
    probs = predict_batch(dummy_cells)
    assert isinstance(probs, list) and len(probs) == 81
    for p in probs:
        assert isinstance(p, list) and len(p) == 10
        # check valid probability distribution
        assert pytest.approx(sum(p), rel=1e-3) == 1.0
        for val in p:
            assert 0.0 <= val <= 1.0