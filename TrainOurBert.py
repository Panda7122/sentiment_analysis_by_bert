# %% 
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
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
emotions = load_dataset("dair-ai/emotion")

# Determine the device to use (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# Define the number of labels and initialize the model
num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

# Training arguments
batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=10, # increase training epochs from 8 to 10
    learning_rate=3e-5,  # adjust learning rate
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"]
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Predict on the validation set
preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)

# Save the model and tokenizer
try:
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
except Exception as e:
    print(f"Can't save model: {e}")

# Confusion Matrix
y_valid = np.array(emotions_encoded["validation"]["label"])
y_preds = np.argmax(preds_output.predictions, axis=1)
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

cm = confusion_matrix(y_valid, y_preds, labels=np.arange(num_labels))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
