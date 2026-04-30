#  MNIST Digit Recognizer (Sketchpad)

This project lets you draw handwritten digits in your browser and predicts what number you drew using a PyTorch-trained MNIST model. The interface is powered by Gradio, making it interactive and easy to use.

---

## Features

- Draw digits directly on a Sketchpad interface  
- Auto-crops, centers, and preprocesses your drawing for the model  
- Predicts digits in real-time  
- Lightweight and easy to run locally  

---

## Demo

Run the app locally to try it:

```bash
python app.py
```

Then open the link displayed in the terminal to draw and see predictions in real-time.

## Project Structure
numbers_ml/
├── app.py # Gradio interface and prediction code
├── models/
│ └── mnist_model.pth # Pretrained MNIST PyTorch model
├── train.py # Training code
├── requirements.txt # Python dependencies
└── README.md # Project README

## Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd numbers_ml
```

2. **Create a virtual enviroment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. Launch the Gradio app

```bash
python app.py
```
2. Open the URL provided in the terminal in your web browser.
3. Draw a digit on the Sketchpad interface.
4. View the prediction by pressing "submit".

## How It Works

1. **Sketchpad Input**  
   - The Gradio Sketchpad allows users to draw digits directly in the browser.  
   - The drawing is returned as a dictionary containing pixel data.

2. **Preprocessing**  
   - Convert the dictionary to a PIL image.  
   - Convert the image to **grayscale** and invert colors (MNIST expects white-on-black).  
   - Crop the image to the digit content and center it.  
   - Resize the image to **28×28 pixels**.  
   - Normalize pixel values using MNIST mean (`0.1307`) and standard deviation (`0.3081`).

3. **Prediction**  
   - Feed the preprocessed image into a PyTorch neural network trained on MNIST.  
   - The model outputs probabilities for each digit (0–9).  
   - The highest probability determines the predicted digit.

4. **Output**  
   - The predicted digit is displayed in the Gradio interface in real-time.
