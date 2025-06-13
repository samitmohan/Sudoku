import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from recognition import DigitClassifier
MODEL_PATH    = "models/digit_recognizer.pth"
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
EPOCHS        = 5

def train():
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # training
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # forward pass
            optimiser.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train Loss : {avg_loss:.4f}")
        print(f"Accuracy : {accuracy:.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

if __name__ == "__main__":
    train()





