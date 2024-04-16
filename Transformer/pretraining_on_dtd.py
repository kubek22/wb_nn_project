#%%

from pytorch_pretrained_vit import ViT
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import os
from PIL import Image
import torch
from torchvision import transforms
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pickle

#%% setting parameters

INIT_LR = 1e-3
BATCH_SIZE = 16
IMG_DIM = 224
IMG_HEIGHT, IMG_WIDTH = IMG_DIM, IMG_DIM
IMAGE_SIZE=(IMG_HEIGHT, IMG_WIDTH)
EPOCHS = 10
TEST_SPLIT = 0.20
VAL_SPLIT = 0.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% loading data

DTD_PATH = "../data/dtd/images"

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

def load_images(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                filepath = os.path.join(root, file)
                with Image.open(filepath) as img:
                    tensor_img = transform(img.copy())
                    images.append(tensor_img)
    return images

images = load_images(DTD_PATH)
len(images)
print(images[0].size())

DTD_CLASSES_PATH = "../data/dtd/labels/labels_joint_anno.txt"

def parse_labels_from_text_file(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                labels.append(parts[1:])
    return labels

labels = parse_labels_from_text_file(DTD_CLASSES_PATH)
labels[:10]

#%% parsing labels

def encode_labels(labels):
    flat_labels = [label for sublist in labels for label in sublist]
    label_encoder = LabelEncoder()
    label_encoder.fit(flat_labels)
    
    encoded_labels = []
    for sublist in labels:
        encoded_labels.append(label_encoder.transform(sublist))
        
    return encoded_labels, label_encoder, max(label_encoder.transform(flat_labels)) + 1

encoded_labels, label_encoder, n_classes = encode_labels(labels)

def transform_encoded_labels(encoded_labels, n_classes):
    y = []
    for sublist in encoded_labels:
        row = np.zeros(n_classes)
        row[sublist] = 1
        y.append(np.array(row))
    return y

y = transform_encoded_labels(encoded_labels, n_classes)

#%% splitting data into sets

def split_data(x, y, test_size=0.2, val_size=0.25, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size/(1-test_size), random_state=random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = split_data(images, y, test_size=TEST_SPLIT, val_size=VAL_SPLIT, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
train_set = CustomDataset(x_train, y_train)
val_set = CustomDataset(x_val, y_val)
test_set = CustomDataset(x_test, y_test)

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

#%% initializing ViT

# https://github.com/lukemelas/PyTorch-Pretrained-ViT
# model = ViT('B_16_imagenet1k', pretrained=True)

model = ViT('B_16', pretrained=False)

in_features = model.fc.in_features

# Define a new fully connected layer with 47 output features
# new_fc_layer = nn.Linear(in_features, n_classes)

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Create the fully connected layer
new_fc_layer = FullyConnectedLayer(in_features, n_classes)

# Replace the last fully connected layer in the model with the new one
model.fc = new_fc_layer

model = model.to(device)

#%% training

opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.BCELoss()

H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

train_steps = len(train_dataloader.dataset) // BATCH_SIZE
val_steps = len(val_dataloader.dataset) // BATCH_SIZE

startTime = time.time()

for e in range(0, EPOCHS):
    model.train()
    totalTrainLoss = 0
    totalValLoss = 0
    # trainCorrect = 0
    # valCorrect = 0
    for (x, y) in train_dataloader:
        # startTime = time.time()
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        y = torch.tensor(y, dtype=pred.dtype)
        # here sth wrong (maybe it needs encoding)
        loss = lossFn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        totalTrainLoss += loss
        # endTime = time.time()
        # print("[INFO] total time taken to train the model: {:.2f}s".format(
        # 	endTime - startTime))
      #   trainCorrect += (pred.argmax(1) == y).type(
   			# torch.float).sum().item()
    with torch.no_grad():
        model.eval()
        for (x, y) in val_dataloader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            y = torch.tensor(y, dtype=pred.dtype)
            totalValLoss += lossFn(pred, y)
       #      valCorrect += (pred.argmax(1) == y).type(
   				# torch.float).sum().item()
    avgTrainLoss = totalTrainLoss / train_steps
    avgValLoss = totalValLoss / val_steps
    # trainCorrect = trainCorrect / len(train_dataloader.dataset)
    # valCorrect = valCorrect / len(val_dataloader.dataset)
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    # H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    # H["val_acc"].append(valCorrect)
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    # print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
   	# 	avgTrainLoss, trainCorrect))
    # print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
   	# 	avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

#%%

print("[INFO] evaluating network...")

test_loss = 0
test_steps = len(test_dataloader.dataset) // BATCH_SIZE

with torch.no_grad():
    model.eval()
    preds = []
    for (x, y) in test_dataloader:
        x = x.to(device)
        pred = model(x)
        y = torch.tensor(y, dtype=pred.dtype)
        test_loss += lossFn(pred, y)
        
avg_test_loss = test_loss / test_steps
        
print("[INFO] Average test loss: ", avg_test_loss)
    
#%% example prediction

with torch.no_grad():
    outputs = model(x_train[0].unsqueeze(0).to(device))
print(outputs.shape)

#%%

torch.save(model, 'output/vit_pretrained_on_dtd.pth')

def save_dict_to_file(data_dict, file_path):
    """
    Saves a dictionary to a file using pickle.

    Args:
    data_dict (dict): The dictionary to save.
    file_path (str): The path to the file where the dictionary will be saved.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        print(f"Dictionary has been saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_dict_from_file(file_path):
    """
    Loads a dictionary from a file using pickle.

    Args:
    file_path (str): The path to the file from which to load the dictionary.

    Returns:
    dict: The dictionary loaded from the file.
    """
    try:
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)
        return data_dict
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

save_dict_to_file(H, 'output/vit_pretrained_on_dtd_training_loss.pkl')
loaded_loss = load_dict_from_file('output/vit_pretrained_on_dtd_training_loss.pkl')

#%%

model = torch.load('output/vit_pretrained_on_dtd.pth')

