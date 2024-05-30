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
from torch.utils.data import ConcatDataset, DataLoader

#%% parameters

os.environ['KMP_DUPLICATE_LIB_OK']='True'

random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 5

ROOT = './../../data'
OUTPUT = 'output'

DATA_EASY = 'RSS_e'
DATA_MEDIUM = 'RSS_m'
DATA_HARD = 'RSS_h'

batch_size = 32

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

def merge_dataloaders(dataloader1, dataloader2, batch_size=32, shuffle=False):
    combined_dataset = ConcatDataset([dataloader1.dataset, dataloader2.dataset])
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    return combined_dataloader

#%% data

path_easy = os.path.join(ROOT, DATA_EASY)
path_medium = os.path.join(ROOT, DATA_MEDIUM)
path_hard = os.path.join(ROOT, DATA_HARD)

rsscn7_easy_data_loader = RSSCN7_DataLoader(path_easy, batch_size=batch_size, shuffle=True)
train_easy_data_loader = rsscn7_easy_data_loader.get_train_dataloader()
test_easy_data_loader = rsscn7_easy_data_loader.get_test_dataloader()

rsscn7_medium_data_loader = RSSCN7_DataLoader(path_medium, batch_size=batch_size, shuffle=True)
train_medium_data_loader = rsscn7_medium_data_loader.get_train_dataloader()
test_medium_data_loader = rsscn7_medium_data_loader.get_test_dataloader()

train_e_m_data_loader = merge_dataloaders(train_easy_data_loader, train_medium_data_loader, 
                                          batch_size=batch_size, shuffle=True)
test_e_m_data_loader = merge_dataloaders(test_easy_data_loader, test_medium_data_loader, 
                                          batch_size=batch_size, shuffle=True)

rsscn7_hard_data_loader = RSSCN7_DataLoader(path_hard, batch_size=batch_size, shuffle=True)
train_hard_data_loader = rsscn7_hard_data_loader.get_train_dataloader()
test_hard_data_loader = rsscn7_hard_data_loader.get_test_dataloader()

train_e_m_h_data_loader = merge_dataloaders(train_e_m_data_loader, train_hard_data_loader, 
                                          batch_size=batch_size, shuffle=True)
test_e_m_h_data_loader = merge_dataloaders(test_e_m_data_loader, test_hard_data_loader, 
                                          batch_size=batch_size, shuffle=True)

#%% training on easy

model = resnet18(weights='ResNet18_Weights.DEFAULT')
num_filters = model.fc.in_features
model.fc = nn.Linear(num_filters, 7)

epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

easy_training_results = train_model(epochs, model, train_easy_data_loader, test_easy_data_loader, device, optimizer, criterion, K)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'training_results': easy_training_results
}

model_path = os.path.join(OUTPUT, 'model_easy.pth')
torch.save(checkpoint, model_path)

checkpoint = torch.load(model_path)
easy_training_results = checkpoint['training_results']
plot_results(OUTPUT, easy_training_results, name_suffix='_easy')

#%% adding medium

model_path = os.path.join(OUTPUT, 'model_easy.pth')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

epochs = 80
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

e_m_training_results = train_model(epochs, model, train_e_m_data_loader, test_e_m_data_loader, device, optimizer, criterion, K)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'training_results': e_m_training_results
}

model_path = os.path.join(OUTPUT, 'model_easy_medium.pth')
torch.save(checkpoint, model_path)

checkpoint = torch.load(model_path)
e_m_training_results = checkpoint['training_results']
plot_results(OUTPUT, e_m_training_results, name_suffix='_easy_medium')

#%% adding hard

model_path = os.path.join(OUTPUT, 'model_easy_medium.pth')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

e_m_h_training_results = train_model(epochs, model, train_e_m_h_data_loader, test_e_m_h_data_loader, device, optimizer, criterion, K)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'training_results': e_m_h_training_results
}

model_path = os.path.join(OUTPUT, 'model_easy_medium_hard.pth')
torch.save(checkpoint, model_path)

checkpoint = torch.load(model_path)
e_m_h_training_results = checkpoint['training_results']
plot_results(OUTPUT, e_m_h_training_results, name_suffix='_easy_medium_hard')

