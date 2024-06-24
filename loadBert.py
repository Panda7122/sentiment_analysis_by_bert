from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_path = './model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Function to predict the emotion of the text
def predict_emotion(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class (the one with the highest probability)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = predictions.argmax().item()

    # Map the class id to the label
    labels = ['joy', 'neutral', 'sadness', 'anger', 'fear', 'love', 'surprise']
    predicted_class = labels[predicted_class_id]

    return predicted_class

# Example usage
#text = input("Enter a text: ")
#emotion = predict_emotion(text)
#print(f"The predicted emotion is: {emotion}")
