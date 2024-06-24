from datasets import load_dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Load the dataset
dataset = load_dataset("imsoumyaneel/sentiment-analysis-llama2")

# Function to convert labels to integers
def convert_labels(example):
    label_map = {
        "joy": 0,
        "neutral": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "love": 5,
        "surprise": 6
    }
    example["label"] = label_map[example["label"]]
    return example

# Convert labels to integers
dataset = dataset.map(convert_labels)

# Split the dataset into training and validation sets
train_test_split = dataset['train'].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-uncased"  # import pre-train Bert model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
dataset = dataset.map(tokenize, batched=True, batch_size=None)

# Number of labels in the dataset
num_labels = 7
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

batch_size = 32  # Reduced batch size to fit in GPU memory
logging_steps = len(dataset["train"]) // batch_size
training_args = TrainingArguments(output_dir="results",
                                  num_train_epochs=8,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1",
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  disable_tqdm=False,
                                  fp16=True)  # Enable mixed precision training

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["validation"])

trainer.train()

results = trainer.evaluate()
print(results)

preds_output = trainer.predict(dataset["validation"])
print(preds_output.metrics)

try:
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
except:
    print("Can't save model")

y_valid = np.array(dataset["validation"]["label"])
y_preds = np.argmax(preds_output.predictions, axis=1)

# Assuming labels are 0, 1, 2, 3, 4, 5, 6 for this dataset
labels = ['joy', 'neutral', 'sadness', 'anger', 'fear', 'love', 'surprise']

cm = confusion_matrix(y_valid, y_preds)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
