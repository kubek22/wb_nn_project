from RSSCN7_dataLoader import RSSCN7_DataLoader
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import os
from custom_metric import k_nearest_metric
import numpy as np
import matplotlib.pyplot as plt

#%%

random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 5

ROOT = './../../data'
OUTPUT = 'plots'

DATA_EASY = 'RSSCN7_easy'
# DATA_MEDIUM = 'RSSCN7_medium'
# DATA_HARD = 'RSSCN7_hard'

#%%

def epoch(model, data_loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in data_loader:
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

    train_loss = total_loss / len(data_loader.dataset)
    train_accuracy = correct / total
    
    train_results = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy
        }
    
    return train_results


def evaluate_model(model, test_data_loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = total_loss / len(test_data_loader.dataset)
    test_accuracy = correct / total
    
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
        }
    
    return test_results


def train_model(epochs, model, train_data_loader, test_data_loader, device, optimizer, criterion, k):
    time0 = time.time()
    model.to(device)
    
    total_results = {
        'train_loss': [],
        'train_accuracy': [],
        'train_k_nearest_metric': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_k_nearest_metric': []
        }
    
    for e in range(epochs):
        train_results = epoch(model, train_data_loader, device, optimizer, criterion)
        train_results['k_nearest_metric'] =  np.mean(k_nearest_metric(k, model, train_data_loader, device))
        print(f'Epoch [{e + 1}/{epochs}], Loss: {train_results["train_loss"]:.4f}, Accuracy: {train_results["train_accuracy"]:.4f}, k nearest metric: {train_results["k_nearest_metric"]:.4f}, Time: {time.time() - time0:.2f} seconds')
        
        total_results['train_loss'].append(train_results["train_loss"])
        total_results['train_accuracy'].append(train_results["train_accuracy"])
        total_results['train_k_nearest_metric'].append(train_results["k_nearest_metric"])
        
        test_results = evaluate_model(model, test_data_loader, criterion)
        test_results['k_nearest_metric'] =  np.mean(k_nearest_metric(k, model, test_data_loader, device))
        print(f'Test Loss: {test_results["test_loss"]:.4f}, Test Accuracy: {test_results["test_accuracy"]:.4f}, Test k nearest metric: {test_results["k_nearest_metric"]:.4f}')
        
        total_results['test_loss'].append(test_results["test_loss"])
        total_results['test_accuracy'].append(test_results["test_accuracy"])
        total_results['test_k_nearest_metric'].append(test_results["k_nearest_metric"])
        
    return total_results

def plot(output_dir, results, result_name, name_suffix, epochs):
    name = result_name + name_suffix + '.png'
    path = os.path.join(output_dir, name)
    
    train_result_name = 'train_' + result_name
    test_result_name = 'test_' + result_name
    
    plt.plot(epochs, results[train_result_name], label=train_result_name)
    plt.plot(epochs, results[test_result_name], label=test_result_name)
    plt.xlabel('epoch')
    plt.ylabel(result_name)
    plt.legend()
    plt.savefig(path)
    plt.clf()

def plot_results(output_dir, results, name_suffix=''):
    max_epoch = len(results['train_loss'])
    epochs = np.arange(0, max_epoch)
    
    plot(output_dir, results, 'loss', name_suffix, epochs)
    plot(output_dir, results, 'accuracy', name_suffix, epochs)
    plot(output_dir, results, 'k_nearest_metric', name_suffix, epochs)


#%%    

batch_size = 32
epochs = 5

# path=os.path.join(ROOT, DATA_EASY)
path=os.path.join(ROOT, 'RSSCN7')

rsscn7_easy_data_loader = RSSCN7_DataLoader(path, batch_size=batch_size, shuffle=True)
train_easy_data_loader = rsscn7_easy_data_loader.get_train_dataloader()
test_easy_data_loader = rsscn7_easy_data_loader.get_test_dataloader()

model = resnet18(weights='ResNet18_Weights.DEFAULT')
num_filters = model.fc.in_features
model.fc = nn.Linear(num_filters, 7)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

easy_training_results = train_model(epochs, model, test_easy_data_loader, test_easy_data_loader, device, optimizer, criterion, K)

plot_results(OUTPUT, easy_training_results, name_suffix='_easy')
