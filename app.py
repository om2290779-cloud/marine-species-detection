import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

st.title("Marine Species Detection 🐠")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Clams','Corals','Crabs','Dolphin','Eel',
    'Jelly Fish','Lobster','Nudibranchs','Octopus',
    'Otter','Penguin','Puffers','Sea Rays',
    'Sea Urchins','Seahorse','Seal','Sharks',
    'Shrimp','Squid','Starfish','Turtle','Whale'
]

num_classes = len(class_names)

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(torch.load("marine_species_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
st.header("Upload Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    st.success(f"Prediction: {class_names[pred.item()]}")

# -------------------------------
# REAL-TIME CAMERA
# -------------------------------
st.header("Real-Time Detection")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)

        label = class_names[pred.item()]

        cv2.putText(img, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)