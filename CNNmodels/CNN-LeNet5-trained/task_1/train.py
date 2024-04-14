
import time
from datetime import timedelta
import torch
from torch import nn
from torch import optim
from model import Net
import util
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./task_1_new_run')      
device = util.get_device()
model = Net().to(device)
#summary(model, (3, util.input_image_size, util.input_image_size))

# Hyperparameters
epochs = 50
# learning_rate = 1e-2
# learning_rate = 1e-4
learning_rate = 1e-3
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_function, optimizer, epoch):
    model.train()      # set the model in training mode
    avg_train_loss, correct = 0, 0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        predictions = model(X)      # forward propagation
        loss = loss_function(predictions, y)        # loss
        avg_train_loss += loss.item()
        optimizer.zero_grad()   # zero the parameter gradients
        loss.backward()         # backpropagation
        optimizer.step()        
        _, predicted = torch.max(predictions.data, 1)  # the class with the highest energy is what we choose as prediction
        correct += (predicted == y).sum().item()
        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_train_loss /= len(dataloader)
    train_accuracy = 100*correct/len(dataloader.dataset)
    statistics('training', train_accuracy, avg_train_loss, epoch)
    return

def evaluate_validation(dataloader, model, loss_function, epoch):
    model.eval()  # Set to evaluation mode
    avg_validation_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            # Move data to the device
            images, labels = data[0].to(device), data[1].to(device)
            predictions = model(images)
            avg_validation_loss += loss_function(predictions, labels).item()  # Loss
            _, predicted = torch.max(predictions.data, 1)  # Prediction
            correct += (predicted == labels).sum().item()
    avg_validation_loss /= len(dataloader)
    validation_accuracy = 100 * correct / len(dataloader.dataset)
    statistics('validation', validation_accuracy, avg_validation_loss, epoch)
    return

def statistics(dataset, accuracy, loss, epoch):
    writer.add_scalar('Loss/' + dataset, loss, epoch)
    writer.add_scalar('Accuracy/' + dataset, accuracy, epoch)
    print("{},\tLoss: {:.{}f}\t| Accuracy: {:.{}f}".format(dataset.title(), loss, 3, accuracy, 3))
    return

def optimize(epochs, train_dataloader, validation_dataloader, model, loss_function, optimizer):
    start_time = time.time()
    for i in range(epochs):
        print(f"\nEpoch {i+1}\n----------------------------------------------")
        train(train_dataloader, model, loss_function, optimizer, i)
        evaluate_validation(validation_dataloader, model, loss_function, i)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    return

# training
train_dataloader = util.get_augmented_train_dataloader()
validation_dataloader = util.get_validation_dataloader()
optimize(epochs, train_dataloader, validation_dataloader, model, loss_function, optimizer)   

print('Finished Training')
# training time, 3hrs 30 mins
torch.save(model.state_dict(), "task_1_new_model.pth")

# get kernel and feature maps
kernels, feature_maps, input_img = util.visualize_kernels_and_feature_maps()
writer.add_images("conv1/2_kernels", kernels.unsqueeze(0) , global_step=0)
writer.add_images("conv1/1_input_img", input_img , global_step=0)
for i, maps in enumerate(feature_maps):
    writer.add_images("conv1/3_feature_maps/{}".format(i), maps.unsqueeze(0) , global_step=0)

writer.close()

'''
References:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn
'''
