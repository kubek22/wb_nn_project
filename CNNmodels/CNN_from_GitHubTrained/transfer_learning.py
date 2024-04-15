from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import util
from torch import nn
from model import Net

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
        # Определяем размеры обучающей и тестовой выборок
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size

        # Разделяем набор данных на обучающую и тестовую выборки
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

pretrained_model_path = 'task_1_new_model.pth'

num_classes = 7

model = Net()

in_features = model.fc1.in_features
model.fc1 = nn.Linear(in_features, 256)
model.fc2 = nn.Linear(256, num_classes)

device = util.get_device()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 50

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
        epoch_accuracy = correct_predictions / total_samples

        print(f'Training - Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        test_loss, test_accuracy = evaluate_model(model, criterion, test_loader)
        print(f'Testing - Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

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
    test_accuracy = correct_predictions / total_samples

    return test_loss, test_accuracy

train_loader = data_loader.get_train_dataloader()
test_loader = data_loader.get_test_dataloader()

train_model(model, criterion, optimizer, train_loader, test_loader)

torch.save(model.state_dict(), 'transfer_learning_model.pth')