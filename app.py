import torch
import gradio as gr
from torchvision import transforms
from models.model import NeuralNetwork
from PIL import Image
import numpy as np

device = torch.device("cpu")

model = NeuralNetwork()
model.load_state_dict(torch.load("models/mnist_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def dict_to_pil(data):
    """
    Converts a Gradio Sketchpad dict to a PIL Image.
    """
    # Usually Sketchpad dict has a key "image"
    if isinstance(data, dict):
        if "image" in data:
            image = data["image"]
        elif "composite" in data:  # for ImageEditor
            image = data["composite"]
        else:
            # fallback: take first value
            image = list(data.values())[0]
    else:
        image = data  # already a PIL image

    # Convert numpy arrays to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    return image

def predict(data):
    image = dict_to_pil(data)  # convert Sketchpad dict → PIL
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()

    return f"Prediction: {pred}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(),
    outputs="text",
)

demo.launch()