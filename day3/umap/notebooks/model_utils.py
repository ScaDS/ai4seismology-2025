import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings("ignore")

from config import (SR, DURATION, N_MELS, HOP_LENGTH, NUM_CLASSES, 
                    EPOCHS_FOR_PRETRAINED, DROPOUT_RATE, FC_SIZE,
                    MODEL_SAVE_PATH)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        # 1st convolutional block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 2nd convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 3rd convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # 4th convolutional block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Calculate input size for the fully connected layer
        # Original size: (N_MELS, time_steps) -> (N_MELS/16, time_steps/16) after 4 pooling layers
        self.time_steps = int(SR * DURATION / HOP_LENGTH) + 1
        self.fc_input_size = 128 * (N_MELS // 16) * (self.time_steps // 16)

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(FC_SIZE, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def feature_extractor(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return x


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience=5,
):
    # Learning rate scheduler with more explicit configuration
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,  # Reduce learning rate every 5 epochs
        gamma=0.5,  # Multiply learning rate by 0.5 (50% reduction)
    )
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    best_test_acc = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]"
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_test_loss = running_loss / total
        epoch_test_acc = 100.0 * correct / total
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)

        # Step the learning rate scheduler
        scheduler.step()

        # Print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
        print(f"Current Learning Rate: {current_lr}")
        print("-" * 50)

        # Early stopping logic
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            early_stopping_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            early_stopping_counter += 1

        # Stop if no improvement
        if early_stopping_counter >= patience:
            print(
                f"Early stopping triggered after {epoch+1} epochs. Best test accuracy: {best_test_acc:.2f}%"
            )
            break

    # Load the best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    return model, train_losses, test_losses, train_accuracies, test_accuracies
