#%%

from pytorch_pretrained_vit import ViT
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam, SGD
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

#%% setting parameters

BATCH_SIZE = 16
EPOCHS = 40

SHUFFLE = True
STRATIFY = True

IMG_DIM = 224
IMG_HEIGHT, IMG_WIDTH = IMG_DIM, IMG_DIM
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

TEST_SPLIT = 0.20
VAL_SPLIT = 0.25
TRAIN_SPLIT = 1 - VAL_SPLIT - TEST_SPLIT
SIZES = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '../data/RSSCN7-master'

#%% loading data

class RCCN7DataLoader:
    def __init__(self, data_dir, batch_size=16, shuffle=True, img_size=(224, 224), set_sizes=(0.55, 0.25, 0.20), stratify=False, random_seed=0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.set_sizes = set_sizes
        self.stratify = stratify
        self.random_seed = random_seed

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.n_classes = len(self.dataset.classes)
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset()

    def split_dataset(self):
        labels = None
        if self.stratify:
            labels = [label for _, label in self.dataset.samples]
        
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=self.set_sizes[2], random_state=self.random_seed, stratify=labels)
        
        if self.stratify:
            labels = [label for _, label in train_dataset]
            
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=self.set_sizes[1], random_state=self.random_seed, stratify=labels)
        return train_dataset, val_dataset, test_dataset

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
data_loader = RCCN7DataLoader(data_dir=DATA_PATH, batch_size=BATCH_SIZE, shuffle=SHUFFLE, img_size=IMAGE_SIZE, stratify=STRATIFY, random_seed=42)

#%% loading model

model = ViT('B_16', pretrained=True)

n_classes = data_loader.n_classes
out_features = 21843

class AddSoftmaxLayer(nn.Module):
    def __init__(self, model, out_features, n_classes):
        super(AddSoftmaxLayer, self).__init__()
        self.model = model
        self.softmax_layer = nn.Linear(in_features=out_features, out_features=n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax_layer(x)
        x = self.softmax(x)
        return x
    
model = AddSoftmaxLayer(model, out_features, n_classes)
model.to(device)

#%% training

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0005)
optimizer = SGD(model.parameters())

def train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, num_epochs):
    train_loss_arr = []
    train_acc_arr = []
    
    val_loss_arr = []
    val_acc_arr = []
    
    test_loss_arr = []
    test_acc_arr = []
    
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

        train_loss = running_loss / total_samples
        train_accuracy = correct_predictions / total_samples * 100
        
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_accuracy)
        
        print(f'Training - Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}%')
        
        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader)
        
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_accuracy)

        print(f'Validation - Epoch {epoch+1}/{num_epochs}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}%')

        test_loss, test_accuracy = evaluate_model(model, criterion, test_loader)
        
        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_accuracy)
        
        print(f'Testing - Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}%')

    print('Training complete.')
    
    results = {
    	"train_loss_arr": train_loss_arr,
    	"train_acc_arr": train_acc_arr,
    	"val_loss_arr": val_loss_arr,
    	"val_acc_arr": val_acc_arr,
        "test_loss_arr": test_loss_arr,
    	"test_acc_arr": test_acc_arr
    }
    return results

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
val_loader = data_loader.get_val_dataloader()
test_loader = data_loader.get_test_dataloader()

startTime = time.time()

results = train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, EPOCHS)

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

#%% saving results

torch.save(model, 'output/fine_tuned_downloaded_vit.pth')

def save(array, filename):
    with open(filename, 'wb') as file:
        pickle.dump(array, file)
        
save(results, 'output/fine_tuned_downloaded_vit_results.pkl')

#%%

def load(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

results = load('output/fine_tuned_downloaded_vit_results.pkl')
epochs = np.arange(1, EPOCHS + 1)

train_loss_arr = results['train_loss_arr']
train_acc_arr = results['train_acc_arr']
val_loss_arr = results['val_loss_arr']
val_acc_arr = results['val_acc_arr']
test_loss_arr = results['test_loss_arr']
test_acc_arr = results['test_acc_arr']

plt.plot(epochs, train_loss_arr)
plt.plot(epochs, val_loss_arr)
plt.legend(('train_loss', 'val_loss'))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(epochs, train_acc_arr)
plt.plot(epochs, val_acc_arr)
plt.legend(('train_acc', 'val_acc'))
plt.xlabel('epoch')
plt.ylabel('accuracy [%]')
plt.show()

print("[INFO] Average test loss: ", test_loss_arr[-1])
print("[INFO] Average test accuracy: {:.2f}%".format(test_acc_arr[-1]))
