import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import torch.nn as nn
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

#%%

DATA_PATH = '../data/RSSCN7-master/cIndustry/c069.jpg'

BATCH_SIZE = 16
EPOCHS = 80

SHUFFLE = True
STRATIFY = True

IMG_DIM = 224
IMG_HEIGHT, IMG_WIDTH = IMG_DIM, IMG_DIM
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

TEST_SPLIT = 0.20
VAL_SPLIT = 0.25
TRAIN_SPLIT = 1 - VAL_SPLIT - TEST_SPLIT
SIZES = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

#%%
class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
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
    
model = torch.load('output/fine_tuned_downloaded_vit.pth')

model.eval()

#%%

def batch_predict(images):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transformed_images = [preprocess_transform(Image.fromarray((img * 255).astype('uint8'))) for img in images]
    batch = torch.stack(transformed_images, dim=0)
    batch = batch.to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()


preprocess_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)

#%%

# DATA_PATH = '../data/RSSCN7-master/aGrass/a024.jpg'
# DATA_PATH = '../data/RSSCN7-master/bField/b104.jpg'
# DATA_PATH = '../data/RSSCN7-master/cIndustry/c069.jpg'
# DATA_PATH = '../data/RSSCN7-master/dRiverLake/d184.jpg'
# DATA_PATH = '../data/RSSCN7-master/eForest/e004.jpg'
# DATA_PATH = '../data/RSSCN7-master/fResident/f002.jpg'
DATA_PATH = '../data/RSSCN7-master/gParking/g018.jpg'

img = Image.open(DATA_PATH).convert('RGB')  # Ensure the image is in RGB format
image_to_explain = preprocess_transform(img).permute(1, 2, 0).numpy()


explanation = explainer.explain_instance(image_to_explain,
                                         batch_predict,  # Prediction function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000,  # Number of images to sample
                                         segmentation_fn=segmenter)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp, mask))  # Adjust for display if normalization is used
plt.axis('off')
plt.show()


