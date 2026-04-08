import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -------------------------------
# Title
# -------------------------------
st.title("Marine Species Detection 🐠")

# -------------------------------
# Device
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Class names
# -------------------------------
class_names = [
    'Clams','Corals','Crabs','Dolphin','Eel',
    'Jelly Fish','Lobster','Nudibranchs','Octopus',
    'Otter','Penguin','Puffers','Sea Rays',
    'Sea Urchins','Seahorse','Seal','Sharks',
    'Shrimp','Squid','Starfish','Turtle','Whale'
]

num_classes = len(class_names)

# -------------------------------
# Load model
# -------------------------------
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "marine_species_model.pth")

try:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Model loading failed ❌ {e}")

# -------------------------------
# Image transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# Upload image
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: {class_names[predicted.item()]}")