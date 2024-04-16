import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import util

train_acc = []
test_acc = []

class RCCN7DataLoader:
    def __init__(self, data_dir, batch_size=32, shuffle=True):
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

data_dir = '../../../data/RSSCN7-master'
batch_size = 32
shuffle = True

data_loader = RCCN7DataLoader(data_dir=data_dir, batch_size=batch_size, shuffle=shuffle)

resnet18 = models.resnet18(pretrained=True)

for param in resnet18.parameters():
    param.requires_grad = False


num_classes = 7
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

device = util.get_device()
resnet10 = resnet18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.002, momentum=0.9)

#most optimal
epochs = 23

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=epochs):
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

        print(f'Training - Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')

        test_loss, test_accuracy = evaluate_model(model, criterion, test_loader)
        print(f'Testing - Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}%')

        train_acc.append(epoch_accuracy)
        test_acc.append(test_accuracy)

    print('Training complete.')
    print('Train:', train_acc)
    print('Test:', test_acc)

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

    return test_loss, test_accuracy

train_loader = data_loader.get_train_dataloader()
test_loader = data_loader.get_test_dataloader()

train_model(resnet18, criterion, optimizer, train_loader, test_loader)

torch.save(resnet18.state_dict(), 'transfer_learning_resnet18.pth')
