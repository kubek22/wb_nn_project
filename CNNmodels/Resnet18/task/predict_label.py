import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import util

def load_and_preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

model_path = "transfer_learning_resnet18.pth"
image_path = "../../../data/RSSCN7-master/cIndustry/c004.jpg"

model = models.resnet18(pretrained=False)
num_classes = 7
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = util.get_device()
model.load_state_dict(torch.load(model_path))

model.eval()

input_tensor = load_and_preprocess_image(image_path)

with torch.no_grad():
    output_tensor = model(input_tensor)

predicted_class = torch.argmax(output_tensor, dim=1).item()
print("Predicted class:", predicted_class)