import torch
from model import CorrosionCNN
import os

# Create folder if not exists
os.makedirs("corrosion_model", exist_ok=True)

# Instantiate model
model = CorrosionCNN()

# Save untrained model weights (just to prevent crash)
torch.save(model.state_dict(), "corrosion_model/corrosion_cnn.pt")

print("✅ corrosion_cnn.pt created successfully!")
