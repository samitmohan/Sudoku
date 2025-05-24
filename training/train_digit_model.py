# Neural Network
import torch as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
from ml_model import DigitRecognizerCNN

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms)

Digit = DigitRecognizerCNN()


