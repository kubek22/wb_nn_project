import torch
import numpy as np
import copy
from torch import nn

#%%

def k_nearest_metric(k, model, data_loader, device):
    components = list(model.children())
    truncated_model = nn.Sequential(*components[:-1])
    
    if len(components) == 1:
        components = list(components[0].children())
        truncated_model = nn.Sequential(*list(components)[:-1])
        
    truncated_model = truncated_model.to(device)
        
    tensors = []
    truncated_model.eval()
    with torch.no_grad():
        x = data_loader.dataset[0][0]
        x = x.unsqueeze(0)
        x = x.to(device)
        tensor = truncated_model(x)
        shape = tensor.shape
        
        labels = np.array([], dtype='int64')
        for x, y in data_loader:
            labels = np.append(labels, y.cpu().numpy()) 
            x, y = x.to(device), y.to(device)
    
            tensor = truncated_model(x)
            tensors.append(tensor)

    if len(tensors) > 1 and tensors[-1].shape != tensors[0].shape:
        last_tensor = tensors[-1]
        tensors = tensors[:-1]
    else:
        last_tensor = None
        
    tensors = torch.stack(tensors)
    tensor_length = shape.numel()
    tensors_shape = torch.Size([tensors.shape[0] * tensors.shape[1], tensor_length])
    tensors = tensors.view(tensors_shape)
    
    if last_tensor is not None:
        last_tensor_shape = torch.Size([last_tensor.shape[0], tensor_length])
        last_tensor = last_tensor.view(last_tensor_shape)
        tensors = torch.cat([tensors, last_tensor], dim=0)

    inf_tensor = torch.full(torch.Size([tensor_length]), float('inf'))
    
    metric_results = []
    for i in range(len(tensors)):
        label = labels[i]
        tensor = tensors[i]
        other_tensors = copy.deepcopy(tensors)
        other_tensors[i] = inf_tensor
        distances = torch.norm(tensor - other_tensors, dim=1)
        
        k_nearest_indices = torch.topk(distances, k, largest=False).indices
        k_nearest_indices = k_nearest_indices.tolist()

        matching_labels = np.sum(labels[k_nearest_indices] == label)
        res = matching_labels / k
        metric_results.append(res)
    return np.array(metric_results)

#%% example use

from RSSCN7_dataLoader import RSSCN7_DataLoader
import torchvision.models as models

data_loader = RSSCN7_DataLoader('./../../data/RSSCN7')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)

train_data_loader = data_loader.get_train_dataloader()

res = k_nearest_metric(5, model, train_data_loader, 'cuda')
