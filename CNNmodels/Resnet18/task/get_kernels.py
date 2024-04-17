from util import get_device
from model import ResNet18
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision import transforms
from util import input_image_size
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import random_split


class RCCN7DataLoader:
    def __init__(self, data_dir, batch_size=32, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.train_dataset, self.test_dataset = self.split_dataset()

    def split_dataset(self):
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        return train_dataset, test_dataset

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


def visualize_kernels_and_feature_maps(data_loader, image_number):
    path = "transfer_learning_resnet18_from_0.pth"
    model = ResNet18().to(get_device())
    model.load_state_dict(torch.load(path))

    conv1 = model.conv1
    layer1 = model.layer1

    kernels = conv1.weight
    kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())
    kernels_grid = torchvision.utils.make_grid(kernels, nrow=4)
    torchvision.utils.save_image(kernels_grid, './kernels_and_feature_maps/kernels_img.png', nrow=4)

    sample_image, _ = data_loader.dataset[image_number]
    sample_image = sample_image.unsqueeze(0)
    sample_image = sample_image.to(get_device())
    torchvision.utils.save_image(sample_image, './kernels_and_feature_maps/sample_image.png')

    output = conv1(sample_image)
    output = layer1(output)

    feature_maps = []
    for i in range(output.shape[1]):
        output_i = output[0, i, :, :].unsqueeze(0)
        output_i = (output_i - output_i.min()) / (output_i.max() - output_i.min())
        feature_maps.append(output_i)
        torchvision.utils.save_image(output_i,
                                     f"./kernels_and_feature_maps/feature_map_{i}.png")  # save image with appropriate file name
    return kernels_grid, feature_maps, sample_image


writer = SummaryWriter(log_dir='./task_1_new_run')

data_dir = '../../../data/RSSCN7-master'
batch_size = 32
shuffle = True

data_loader = RCCN7DataLoader(data_dir=data_dir, batch_size=batch_size, shuffle=shuffle)

kernels, feature_maps, input_img = visualize_kernels_and_feature_maps(data_loader, 308)
writer.add_images("conv1/2_kernels", kernels.unsqueeze(0), global_step=0)
writer.add_images("conv1/1_input_img", input_img, global_step=0)
for i, maps in enumerate(feature_maps):
    writer.add_images("conv1/3_feature_maps/{}".format(i), maps.unsqueeze(0), global_step=0)

writer.close()
