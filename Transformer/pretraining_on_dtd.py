#%%

from pytorch_pretrained_vit import ViT
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import numpy as np
import torch
import time
import os
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt

#%% setting parameters

BATCH_SIZE = 16
IMG_DIM = 224
IMG_HEIGHT, IMG_WIDTH = IMG_DIM, IMG_DIM
IMAGE_SIZE=(IMG_HEIGHT, IMG_WIDTH)
EPOCHS = 40
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

model = ViT('B_16', pretrained=False)

in_features = model.fc.in_features

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

new_fc_layer = FullyConnectedLayer(in_features, n_classes)
model.fc = new_fc_layer

model = model.to(device)

#%% training

opt = Adam(model.parameters())
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
    
    correct_train_predictions = 0
    total_train_predictions = 0
    
    correct_val_predictions = 0
    total_val_predictions = 0
    
    for (x, y) in train_dataloader:
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        y = torch.tensor(y, dtype=pred.dtype)
        loss = lossFn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        totalTrainLoss += loss
        
        predicted_labels = (torch.sigmoid(pred) >= 0.5).bool()
        y = y.bool()
        correct = ((predicted_labels == y) | (predicted_labels & y)).sum().item()
        if correct > 0:
            correct_train_predictions += 1
        total_train_predictions += y.size(0)
        
    with torch.no_grad():
        model.eval()
        for (x, y) in val_dataloader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            y = torch.tensor(y, dtype=pred.dtype)
            totalValLoss += lossFn(pred, y)
            
            predicted_labels = (torch.sigmoid(pred) >= 0.5).bool()
            y = y.bool()
            correct = ((predicted_labels == y) | (predicted_labels & y)).sum().item()
            if correct > 0:
                correct_val_predictions += 1
            total_val_predictions += y.size(0)
            
    avgTrainLoss = totalTrainLoss / train_steps
    avgValLoss = totalValLoss / val_steps
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    
    train_accuracy = correct_train_predictions / total_train_predictions
    val_accuracy = correct_val_predictions / total_val_predictions
    H["train_acc"].append(train_accuracy)
    H["val_acc"].append(val_accuracy)
    
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

#%%

print("[INFO] evaluating network...")

test_loss = 0
test_steps = len(test_dataloader.dataset) // BATCH_SIZE
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    model.eval()
    preds = []
    for (x, y) in test_dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        y = torch.tensor(y, dtype=pred.dtype)
        test_loss += lossFn(pred, y)
        predicted_labels = (torch.sigmoid(pred) >= 0.5).bool()
        y = y.bool()
        correct = ((predicted_labels == y) | (predicted_labels & y)).sum().item()
        if correct > 0:
            correct_predictions += 1
        total_predictions += y.size(0)
        
avg_test_loss = test_loss / test_steps
accuracy = correct_predictions / total_predictions
        
print("[INFO] Average test loss: ", avg_test_loss)
print("[INFO] Accuracy: ", accuracy)
    
#%% example prediction

i = 110
with torch.no_grad():
    outputs = model(x_train[i].unsqueeze(0).to(device))
    
np.argmax(outputs[0].to('cpu'))
np.argmax(y_train[i])

#%% saving results

torch.save(model, 'output/vit_pretrained_on_dtd.pth')

def save_dict_to_file(data_dict, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
    except Exception as e:
        print(f"An error occurred: {e}")

def load_dict_from_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)
        return data_dict
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

save_dict_to_file(H, 'output/vit_pretrained_on_dtd_training_loss.pkl')

#%%

loaded_loss = load_dict_from_file('output/vit_pretrained_on_dtd_training_loss.pkl')
epochs = np.arange(1, EPOCHS + 1)
train_loss = loaded_loss['train_loss']
val_loss = loaded_loss['val_loss']

plt.plot(epochs, train_loss)
plt.plot(epochs, val_loss)
plt.legend(('train_loss', 'val_loss'))
plt.show()

#%%

model = torch.load('output/vit_pretrained_on_dtd.pth')

