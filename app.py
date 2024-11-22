from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer (update the path if needed)
model_name = "./models/arabic_sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define input model for the API
class TextInput(BaseModel):
    text: str

# Define output model for the API
class PredictionResponse(BaseModel):
    predicted_label: int
    description: str

# Define the prediction endpoint with the custom response model
@app.post("/predict/", response_model=PredictionResponse)
async def predict(text_input: TextInput):
    # Tokenize the input text
    inputs = tokenizer(
        text_input.text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Map the predicted label to its description
    reverse_label_mapping = {
        0: "Extremely dissatisfied",
        1: "Dissatisfied",
        3: "Satisfied",
        4: "Extremely satisfied"
    }
    predicted_label_description = reverse_label_mapping[prediction]

    return PredictionResponse(
        predicted_label=prediction,
        description=predicted_label_description
    )
