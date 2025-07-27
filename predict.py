import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the same model used during training
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Match the output classes
model.load_state_dict(torch.load("corrosion_model/corrosion_cnn.pt", map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(img: Image.Image) -> str:
    img_tensor = transform(img).unsqueeze(0)
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    return "Corroded" if predicted.item() == 0 else "Non-Corroded"
