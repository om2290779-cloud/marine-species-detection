# updated version
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_path = "marine_species_model.pth"
class_names = [
    'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel',
    'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus',
    'Otter', 'Penguin', 'Puffers', 'Sea Rays',
    'Sea Urchins', 'Seahorse', 'Seal', 'Sharks',
    'Shrimp', 'Squid', 'Starfish', 'Turtle', 'Whale'
]
num_classes = len(class_names)

# Build model
def build_model(num_classes):

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model

model = build_model(num_classes)

model.load_state_dict(
    torch.load(model_path, map_location=DEVICE)
)

model.to(DEVICE)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            image = Image.open(file).convert("RGB")

            image = transform(image)

            image = image.unsqueeze(0)

            image = image.to(DEVICE)

            with torch.no_grad():

                outputs = model(image)

                probabilities = torch.softmax(outputs, dim=1)

                conf, predicted = torch.max(probabilities, 1)

                prediction = class_names[predicted.item()]

                confidence = round(conf.item() * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=10000)