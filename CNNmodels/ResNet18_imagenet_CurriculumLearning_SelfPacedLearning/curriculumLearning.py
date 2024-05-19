from RSSCN7_dataLoader import RSSCN7_DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset

data_dir = '../../data/RSSCN7'
batch_size = 32
learning_rate = 0.01
num_epochs = 40
lambda_beginning = 0.1
lambda_end = 1

rsscn7_data_loader = RSSCN7_DataLoader(data_dir, batch_size=batch_size)
train_loader = rsscn7_data_loader.get_train_dataloader()
test_loader = rsscn7_data_loader.get_test_dataloader()

model = resnet18(pretrained=True)
num_filters = model.fc.in_features
model.fc = nn.Linear(num_filters, 7)

criterion = nn.CrossEntropyLoss()
opitmizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model_self_paced(model, train_loader, criterion, optimizer, num_epochs):
    device = torch.device('mps')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_samples = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            train_samples.append((inputs, labels, loss.item()))

            loss.backward()
            optimizer.step()

        train_samples.sort(key=lambda x: x[2]) # sort by loss (the first are the easiest)
        lambda_current = lambda_beginning+(lambda_end-lambda_beginning)*(epoch/num_epochs-1)
        num_samples_current = int(lambda_current*len(train_samples))

        easy_enough_samples = train_samples[:num_samples_current]
        easy_enough_inputs = torch.cat([x[0] for x in easy_enough_samples])
        easy_enough_labels = torch.cat([x[1] for x in easy_enough_samples])
        train_loader = DataLoader(TensorDataset(easy_enough_inputs, easy_enough_labels), batch_size=batch_size, shuffle=True)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader.dataset)}, Lambda: {lambda_current:.2f}')

    print('Finished Training Successfully')


def evaluate_model(model, test_loader, criterion):
    device = torch.device('mps')
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')
    print(f'Average test loss: {test_loss / len(test_loader.dataset)}')

train_model_self_paced(model, train_loader, criterion, opitmizer, num_epochs)

evaluate_model(model, test_loader, criterion)



