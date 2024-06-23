import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
model_path = './saved_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    # Tokenize the input text
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    encoded_input = encoded_input.to(device)
    
    # Get model predictions
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Calculate probabilities
    probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
    predicted_sentiment = probabilities.argmax(-1).item()
    probability_of_sentiment = probabilities[0, predicted_sentiment].item()
    
    # Convert prediction to sentiment
    sentiments = ['Negative', 'Neutral', 'Positive']  # Adjust as needed
    return sentiments[predicted_sentiment], probability_of_sentiment

# User text input for testing
if __name__ == "__main__":
    #user_input = input("Enter a tweet to analyze sentiment: ")
    texts = [
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
        "Here’s to us. Forever and always.",
        "It's interesting how some people choose to prioritize their own convenience over reliability.",
        "I guess not everyone values punctuality.",
        "It's surprising how little effort some people put into their work.",
        "It must be nice to have such flexible standards.",
        "It has been raining for two weeks,our trip has to be canceled",
        "It seems like the model is suck"
    ]
    for text in texts:
        sentiment, probability = predict_sentiment(text)
        print(f"text:{text} , Predicted Sentiment: {sentiment}, Probability: {probability:.3f}")
