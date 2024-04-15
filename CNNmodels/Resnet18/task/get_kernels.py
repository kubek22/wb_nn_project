from util import get_device
from model import ResNet18
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from util import input_image_size


def visualize_kernels_and_feature_maps(image_number):
    path = "our_resnet18_pretrained_DTD.pth"    # take model from train.py
    model = ResNet18().to(get_device())
    model.load_state_dict(torch.load(path))
    kernels = model.conv1.weight
    kernels = (kernels - kernels.min())/(kernels.max() - kernels.min())
    kernels_grid = torchvision.utils.make_grid(kernels, nrow = 4)
    torchvision.utils.save_image(kernels_grid, './kernels_and_feature_maps/kernels_img.png', nrow=4)

    training_data = datasets.DTD(
        root='./../',
        split="train",
        download=True,
        transform=transforms.Compose([
            transforms.Resize((input_image_size, input_image_size)),
            transforms.ToTensor(),
        ])
    )
    sample_image, _ = training_data[image_number]
    sample_image = sample_image.unsqueeze(0)    # unsqueeze to add a batch dimension
    sample_image = sample_image.to(get_device())
    torchvision.utils.save_image(sample_image, './kernels_and_feature_maps/sample_image.png')
    output = model.conv1(sample_image)
    feature_maps = []
    for i in range(output.shape[1]):
        output_i = output[0, i, :, :].unsqueeze(0)
        output_i = (output_i - output_i.min())/(output_i.max() - output_i.min())
        feature_maps.append(output_i)
        torchvision.utils.save_image(output_i, f"./kernels_and_feature_maps/feature_map_{i}.png")   # save image with appropriate file name
    return kernels_grid, feature_maps, sample_image


writer = SummaryWriter(log_dir='./task_1_new_run')

kernels, feature_maps, input_img = visualize_kernels_and_feature_maps(308)
writer.add_images("conv1/2_kernels", kernels.unsqueeze(0), global_step=0)
writer.add_images("conv1/1_input_img", input_img, global_step=0)
for i, maps in enumerate(feature_maps):
    writer.add_images("conv1/3_feature_maps/{}".format(i), maps.unsqueeze(0), global_step=0)

writer.close()