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
    labels =  ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    top2_choices = []
    for probs, indices in zip(top2_prob, top2_indices):
        choices = [(labels[idx], prob.item()) for prob, idx in zip(probs, indices)]
        top2_choices.append(choices)
    return top2_choices

# Example usage
if __name__ == "__main__":
    """texts = [
        "Emma: Hey, Alex! How's it going?",
        "Alex: Oh, hey, Emma. It's going... you know, the usual.",
        "Emma: That doesn't sound very enthusiastic. Everything okay?",
        "Alex: Yeah, just feeling a bit off today. Work's been pretty overwhelming lately.",
        "Emma: I'm sorry to hear that. Anything specific bothering you, or is it just the workload?",
        "Alex: It's mostly the workload. And maybe a bit of feeling like I'm not doing enough, even when I'm working so hard.",
        "Emma: I totally get that. It can be really tough when you put in so much effort and still feel like it's not enough. Have you talked to anyone at work about it?",
        "Alex: Not really. I don't want to come across as complaining or not capable.",
        "Emma: I understand. But sometimes sharing your thoughts can actually show your commitment to doing a good job. Maybe they'd appreciate knowing how you're feeling.",
        "Alex: Yeah, maybe you're right. I guess I'm just worried about how it might be perceived.",
        "Emma: That's completely normal. Just remember, everyone has days where they feel like this. You're not alone in it.",
        "Alex: Thanks, Emma. It really helps to hear that. Maybe I will bring it up next time there's a team meeting.",
        "Emma: Good idea! And hey, if you ever need to vent or just talk, I'm here for you.",
        "Alex: Thanks, I appreciate that a lot. How about you? How's everything on your end?",
        "Emma: Oh, you know, the usual ups and downs. But overall, not too bad. Just trying to take things one day at a time.",
        "Alex: Sounds like a good approach. Thanks again for listening. It means a lot.",
        "Emma: Anytime, Alex. Hang in there. We'll get through this together."
    ]"""
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


    predictions = predict_emotion(texts)
    for text, preds in zip(texts, predictions):
        print(f"Text: {text}")
        for label, prob in preds:
            print(f"  Predicted Emotion: {label}, Probability: {prob:.4f}")
