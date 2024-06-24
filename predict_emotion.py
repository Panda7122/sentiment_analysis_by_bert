import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Function to tokenize input text
def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Load the pre-trained BERT model and tokenizer
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()

# Function to predict the emotion of the given text
def predict_emotion(texts):
    inputs = tokenize(texts)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top2_prob, top2_indices = torch.topk(probabilities, 2, dim=1)
    labels = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
    top2_choices = []
    for probs, indices in zip(top2_prob, top2_indices):
        choices = [(labels[idx], prob.item()) for prob, idx in zip(probs, indices)]
        top2_choices.append(choices)
    return top2_choices

# Example usage
if __name__ == "__main__":
    """texts = [
        "Happy Valentine’s Day! You look absolutely stunning tonight,daring.",
        "Thank you! You’re looking quite handsome yourself. Happy Valentine’s Day! I’ve been looking forward to this all week.",
        "Me too. I’ve got a little surprise planned for us later, but first, how about we start with a toast?",
        "Ooh, a surprise! You know how much I love surprises. What are we toasting to?",
        "To us, and to all the wonderful moments we’ve shared. Here’s to many more.",
        "Cheers to that! You always know how to make me feel special.",
        "That’s because you are special. I’ve got something for you. *hands over a small gift box*",
        "Oh, you didn’t have to! What’s this? *opens the box* Wow, it’s beautiful! A necklace with our initials. I love it.",
        "I’m glad you like it. I wanted to give you something that would remind you of us, every day.",
        "It’s perfect. Thank you so much. You really are the best. I have something for you too. *hands over a small wrapped gift*",
        "What’s this? *unwraps the gift* A watch! And it has a little engraving on the back. “Forever and always.” It’s perfect. Thank you.",
        "I’m so happy you like it. I wanted to give you something that you could wear and think of me.",
        "I will, every time I check the time. This has been such a wonderful night already. And we’ve only just started.",
        "Agreed. I’m so grateful to have you in my life. Here’s to many more Valentine’s Days together.",
        "Here’s to us. Forever and always."
    ]"""
    texts = [
        "Oh great, another meeting that could have been an email.",
        "I just love when my favorite show gets canceled after one season.",
        "It's so wonderful when my computer decides to update right before a deadline.",
        "Nothing makes me happier than being stuck in traffic for hours.",
        "I absolutely adore it when people cancel plans at the last minute.",
        "It's fantastic how my phone battery always dies at the most convenient times.",
        "I really enjoy it when the weather forecast is completely wrong.",
        "I love it when my alarm clock fails to go off on the most important day.",
        "It's so nice how my internet always slows down when I need it the most.",
        "I just love it when my umbrella breaks in the middle of a storm.",
        "I will date will my crush this weekend",
        "I will go on a date whis weekend",
        "I just got my bachelor degree, I will apply for a job tomorrow",
        "When he saw his brother appear at his wedding ceremony, he ran out of his mind",
        "It seems like the model is sock.",
        "fuck you",
        "I hate this world",
        "that war is terrible",
        "This is an apple",
        "This is a banana",
        "I like to eat apple",
        "I am not happy right now,don't bother me.",
        "I am not happy right now.",
        "I am happy.",
        "I am so excited"
    ]

    predictions = predict_emotion(texts)
    for text, preds in zip(texts, predictions):
        print(f"Text: {text}")
        for label, prob in preds:
            print(f"  Predicted Emotion: {label}, Probability: {prob:.4f}")
