"""
Load the MNIST dataset using torchvision.datasets.MNIST.
Preprocess the data (normalize, transform to tensors).
Instantiate your DigitRecognizerCNN model.
Define your loss function (e.g., nn.CrossEntropyLoss) and optimizer (e.g., optim.Adam).
Implement the training loop (epochs, forward pass, backward pass, optimizer step).
Evaluate the model's accuracy on the test set.
Save the trained model's state_dict() to models/digit_recognizer.pth. Make sure the models/ directory exists.
"""