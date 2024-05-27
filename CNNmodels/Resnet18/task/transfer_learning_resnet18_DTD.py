# final
import torch
import torch.nn as nn
import random
import time
from model import ResNet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

class RSSCN7_DataLoader:
    def __init__(self, data_dir, batch_size=32, shuffle=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.train_dataset, self.test_dataset = self.split_dataset()

    def split_dataset(self):
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        return train_dataset, test_dataset

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

time0 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

train_acc = []
test_acc = []
loss_train = []
loss_test = []
epoch_time = []
learning_rates = []

data_dir = '/Users/katebokhan/Desktop/wb_nn_project/data/RSSCN7'
batch_size = 32
learning_rate = 0.001

data_loader = RSSCN7_DataLoader(data_dir=data_dir, batch_size=batch_size, shuffle=True)

pretrained_model_path = 'resnet18_trained_on_DTD_from_80_to_90.pth'
pretrained_resnet18 = ResNet18()
pretrained_resnet18.load_state_dict(torch.load(pretrained_model_path, map_location=device))

pretrained_resnet18.fc = nn.Linear(pretrained_resnet18.resnet18.fc.in_features, 7)

pretrained_resnet18 = pretrained_resnet18.to(device)

for name, param in pretrained_resnet18.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pretrained_resnet18.parameters(), lr=learning_rate)

# most optimal
epochs = 100

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs, learning_rate):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples * 100

        if epoch_accuracy >= 85:
            learning_rate = 0.0001

        if epoch_accuracy >= 93:
            learning_rate = 0.00001

        if epoch_accuracy >= 96:
            learning_rate = 0.000001

        time_cur = time.time() - time0
        print("learning_rate = ", learning_rate)

        print(f'Training - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')

        test_loss, test_accuracy = evaluate_model(model, criterion, test_loader)
        print(f'Testing - Epoch {epoch + 1}/{num_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}%')
        print(f'Time: {time_cur:.2f} seconds')

        train_acc.append(epoch_accuracy)
        test_acc.append(test_accuracy)
        learning_rates.append(learning_rate)
        epoch_time.append(time_cur)

    print('Training complete.')

def evaluate_model(model, criterion, test_loader):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_loss = running_loss / total_samples
    test_accuracy = correct_predictions / total_samples * 100

    test_acc.append(test_accuracy)
    loss_test.append(test_loss)

    return test_loss, test_accuracy

train_loader = data_loader.get_train_dataloader()
test_loader = data_loader.get_test_dataloader()

train_model(pretrained_resnet18, criterion, optimizer, train_loader, test_loader, epochs, learning_rate)

print('Training completed successfully.')
print("Training accuracy:")
print(train_acc)
print("Test accuracy:")
print(test_acc)
print('Loss train:')
print(loss_train)
print("Test loss:")
print(loss_test)
print("Learning rate:")
print(learning_rates)
print("Epoch times:")
print(epoch_time)

torch.save(pretrained_resnet18, 'resnet18_DTD_transfer_learning.pth')
