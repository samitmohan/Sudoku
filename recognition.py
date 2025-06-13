import torch
import config
import torch.nn as nn
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
        # layers
        nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1), # 1 input, 32 filters, padding to keep 28x28
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)), # Output: (32, 14, 14)
        nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), # Padding to keep 14x14
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)), # Output: (64, 7, 7)
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.model(x)

    def predict_batch(self, images: list) -> list:
        """
        Instance method: takes a list of 28×28 uint8 arrays,
        returns a list of 10‐probability lists per image.
        """
        # 1) Preprocess & make a batch tensor
        batch = torch.stack([_transform(img) for img in images], dim=0).to(DEVICE)
        # 2) Forward + softmax
        with torch.no_grad():
            logits = self(batch)           
            probs  = torch.softmax(logits, dim=1)  
        return probs.cpu().numpy().tolist()

def load_model(path: str = None) -> DigitClassifier:
    if path is None:
        path = config.MODEL_PATH
    model = DigitClassifier().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_batch(images: list, path: str = None) -> list:
    model = load_model(path)
    return model.predict_batch(images)

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])
