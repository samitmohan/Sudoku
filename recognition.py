# recognition.py
import pytesseract
from ocr_utils import ocr_reader
from PIL import Image
import cv2
import torch
import config
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import List

# Use the same device for both loading and inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST‐style preprocessing: convert 28×28 uint8 → float tensor, normalize
_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # HxW ndarray → PIL Image (L mode)
        transforms.ToTensor(),  # → FloatTensor C×H×W in [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean, std
    ]
)


class DigitClassifier(nn.Module):
    """
    A simple CNN for 10‐way digit classification on 28×28 grayscale images.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [1,28,28]→[32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # →[32,14,14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # →[64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),  # →[64,7,7]
            nn.Flatten(),  # →[64*7*7]
            nn.Linear(64 * 7 * 7, 10),  # →[10]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Classify a batch of 28×28 uint8 numpy images.

        Parameters:
            images: list of H×W numpy arrays (dtype=uint8), white-on-black.
        Returns:
            preds: numpy array of shape (N,), predicted class indices.
            confs: numpy array of shape (N,), max softmax confidence scores.
        """
        # Transform and stack into a single batch tensor
        tensors = []
        for img in images:
            if img is None:
                raise ValueError("Received None in predict_batch inputs")
            if img.dtype != np.uint8:
                raise ValueError(f"Image dtype must be uint8, got {img.dtype}")
            if img.ndim != 2:
                raise ValueError(f"Image must be single-channel, got shape {img.shape}")
            t = _transform(img)  # → [1,28,28] float tensor
            tensors.append(t)
        batch = torch.stack(tensors, dim=0).to(DEVICE)
        # Inference
        with torch.no_grad():
            logits = self(batch)
            probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values
        return preds.cpu().numpy(), confs.cpu().numpy()


def load_model(path: str = None) -> DigitClassifier:
    """
    Load a pre-trained DigitClassifier from disk.

    If `path` is None, uses `config.MODEL_PATH`.

    Returns a model on `DEVICE`, in eval mode.
    """
    if path is None:
        path = config.MODEL_PATH
    model = DigitClassifier().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def easyocr_digit(img):
    txt = ocr_reader.readtext(img, detail=0, allowlist="123456789")
    if not txt:
        return 0
    s = txt[0].strip()
    return int(s) if s.isdigit() else 0
