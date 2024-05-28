#final

from RSSCN7_dataLoader import RSSCN7_DataLoader
from torchvision.models import resnet18
# from ..Resnet18.task.model import ResNet18
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import time
import random
from custom_metric import k_nearest_metric
import numpy as np
import os

acc_train = []
acc_test = []
loss_train = []
loss_test = []
time_epoch = []
cur_lambda = []
cur_learning_rate = []

random.seed(42)

time0 = time.time()

data_dir = os.path.join('./../../data', 'RSSCN7')
batch_size = 32
learning_rate = 0.0001
num_epochs = 100
lambda_beginning = 0.1
lambda_end = 1

rsscn7_data_loader = RSSCN7_DataLoader(data_dir, batch_size=batch_size, shuffle=True)
train_loader = rsscn7_data_loader.get_train_dataloader()
test_loader = rsscn7_data_loader.get_test_dataloader()

model = resnet18(weights='ResNet18_Weights.DEFAULT')
num_filters = model.fc.in_features
model.fc = nn.Linear(num_filters, 7)

######### In case of model pretrained on DTD: uploading the weights #################################################

# pretrained_model_path = "/kaggle/input/resnet18-pretrained-on-dtd/pytorch/version1/1/resnet18_trained_on_DTD_from_80_to_90.pth"
# pretrained_resnet18 = ResNet18()
# pretrained_resnet18.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device(device)))

# model = pretrained_resnet18.to(device)

# model.fc = nn.Linear(47, 7)

criterion = nn.CrossEntropyLoss()
opitmizer = optim.Adam(model.parameters(), lr=learning_rate)
my_device = torch.device("cuda" if torch.cuda.is_available() else "mps")

step = 0.05


def train_model_self_paced(model, train_loader, test_loader, criterion, optimizer, num_epochs, learning_rate):
    device = my_device
    model.to(device)
    counter = 0

    lambda_current = lambda_beginning

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        train_samples = []

        if lambda_current < 1:
            with torch.no_grad():
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    train_samples.append((inputs, labels, loss.item()))

            train_samples.sort(key=lambda x: x[2])  # sort by loss (the first are the easiest)

            num_samples_current = int(lambda_current * len(train_samples))

            easy_enough_samples = train_samples[:num_samples_current]
            easy_enough_inputs = torch.cat([x[0] for x in easy_enough_samples])
            easy_enough_labels = torch.cat([x[1] for x in easy_enough_samples])
            easy_enough_dataset = TensorDataset(easy_enough_inputs, easy_enough_labels)
            easy_enough_loader = DataLoader(easy_enough_dataset, batch_size=batch_size, shuffle=True)
        else:
            easy_enough_loader = train_loader

        for inputs, labels in easy_enough_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(easy_enough_loader.dataset)
        train_accuracy = correct / total
        num_images = len(easy_enough_loader.dataset)
        
        # counting metric result
        print(np.mean(k_nearest_metric(5, model, easy_enough_loader, device)))

        if train_accuracy >= 0.88:
            learning_rate = 0.00001

        if train_accuracy >= 0.93:
            learning_rate = 0.000001

        if train_accuracy >= 0.96:
            learning_rate = 0.0000001

        cur_time_ = time.time() - time0

        print("learing_rate = ", learning_rate)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Images: {num_images}, Lambda: {lambda_current:.2f}, Time: {cur_time_:.2f} seconds')

        if train_accuracy > 0.85:
            if lambda_current < 0.6:
                lambda_current += step
                if lambda_current > 1:
                    lambda_current = 1
            else:
                counter = counter + 1
                if counter % 2 == 0:
                    lambda_current += step
                    counter = 0
                    if lambda_current > 1:
                        lambda_current = 1

        acc_train.append(train_accuracy)
        loss_train.append(train_loss)
        time_epoch.append(cur_time_)
        cur_lambda.append(lambda_current)
        cur_learning_rate.append(learning_rate)

        evaluate_model(model, test_loader, criterion)
        

    print('Finished Training Successfully')


def evaluate_model(model, test_loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = total_loss / len(test_loader.dataset)
    test_accuracy = correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    acc_test.append(test_accuracy)
    loss_test.append(test_loss)


train_model_self_paced(model, train_loader, test_loader, criterion, opitmizer, num_epochs, learning_rate)
print("Accuracy train:")
print(acc_train)
print("Accuracy test:")
print(acc_test)
print("Loss train:")
print(loss_train)
print("Loss test:")
print(loss_test)
print("Time epoch:")
print(time_epoch)
print("Learning rate:")
print(cur_learning_rate)
print("Lambdas:")
print(cur_lambda)

torch.save(model, 'resnet18_imagenet_self_paced.pth')