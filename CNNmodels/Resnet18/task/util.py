import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from torchvision import transforms

# ImageNet stats
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# data augmentation
# decreasing the input image size

input_image_size = 256
# input_image_size = 224
# input_image_size = 112
# input_image_size = 56
# input_image_size = 28

train_batch_size = 128
# train_batch_size = 64
# test_batch_size = 256
# train_batch_size = 32
test_batch_size = 128

custom_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.5, 2.0)),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(input_image_size, input_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'augment_transforms': [
        transforms.Compose([
            # transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),      
            transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        transforms.Compose([
            # transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),       
            transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        transforms.Compose([
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN),       
            transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        custom_transform,
        custom_transform
    ]
}


# Dataset stores the samples and their corresponding labels
# DataLoader wraps an iterable around the Dataset

def get_augmented_train_dataloader():
    augmented_dataset = []
    for augment_transform in data_transforms['augment_transforms']:
        augmented_dataset.append(datasets.DTD(
            root='./../',
            split="train",
            download=True,
            transform=augment_transform
        ))
    training_data = datasets.DTD(
        root='./../',
        split="train",
        download=True,
        transform=data_transforms['train']
    )
    augmented_dataset.append(training_data)
    combined_data = ConcatDataset(augmented_dataset)
    combined_dataloader = DataLoader(combined_data, batch_size=train_batch_size, shuffle=True)
    return combined_dataloader

train_dataloader = get_augmented_train_dataloader()
len(train_dataloader.dataset)

def get_train_dataloader():
    training_data = datasets.DTD(
        root='./../',
        split="train",
        download=True,
        transform=data_transforms['train']
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    return train_dataloader

def get_validation_dataloader():
    validation_data = datasets.DTD(
        root='./../',
        split="val",
        download=True,
        transform=data_transforms['test']
    )
    validation_dataloader = DataLoader(validation_data, batch_size=test_batch_size, shuffle=False)
    return validation_dataloader

def get_test_dataloader():
    test_data = datasets.DTD(
        root='./../',
        split="test",
        download=True,
        transform=data_transforms['test']
    )
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    return test_dataloader

def get_device():
    # Get cpu, cuda, or mps device for training.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using {device} device")
    return device


def get_test_set_preformance(model, device):
    model.eval()       
    test_dataloader = get_test_dataloader()
    loss_function = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in test_dataloader:
            print(device)
            images, labels = data[0].to(device), data[1].to(device)
            predictions = model(images)
            test_loss += loss_function(predictions, labels).item()
            _, predicted = torch.max(predictions.data, 1)  
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_dataloader)
    accuracy = 100*correct/len(test_dataloader.dataset)
    print("Test,\tAverage Loss: {:.{}f}\t| Accuracy: {:.{}f}%".format(test_loss, 3, accuracy, 3))
    return
