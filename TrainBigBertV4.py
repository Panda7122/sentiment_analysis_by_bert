from datasets import load_dataset, DatasetDict, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import os

def custom_accuracy(preds, labels):
    correct = 0
    total = len(labels)
    for pred, label in zip(preds, labels):
        predicted_label = np.argmax(pred)
        if predicted_label == label:
            if pred[predicted_label] >= 0.5:
                correct += 1
    return correct / total

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    accuracy = custom_accuracy(preds, labels)
    preds = np.argmax(preds, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = load_dataset("dair-ai/emotion")

# 使用数据集的50%
#dataset = dataset['train'].train_test_split(test_size=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
dataset = dataset.map(tokenize, batched=True, batch_size=None)

# Number of labels in the dataset
num_labels = 6

# Define a custom model class with dropout layers
class CustomModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        logits = self.dropout(logits)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits, labels)
        
        return (loss, logits) if loss is not None else logits

# Function to balance the dataset using oversampling
def balance_dataset(dataset):
    df = dataset.to_pandas()
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(df.drop(columns='label'), df['label'].tolist())
    balanced_df = X_resampled.copy()
    balanced_df['label'] = y_resampled
    balanced_dataset = Dataset.from_pandas(balanced_df)
    return balanced_dataset

# Balance the dataset
balanced_dataset = balance_dataset(dataset['train'])
#balanced_dataset = dataset['train']

# Initialize KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
accuracies = []
f1_scores = []

# K-fold cross validation
for train_index, val_index in kf.split(balanced_dataset):
    train_dataset = balanced_dataset.select(train_index.tolist())
    val_dataset = balanced_dataset.select(val_index.tolist())
    
    model = CustomModel(model_name, num_labels).to(device)

    training_args = TrainingArguments(output_dir="results",
                                      num_train_epochs=3,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      load_best_model_at_end=True,
                                      metric_for_best_model="f1",
                                      weight_decay=0.01,
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
                                      disable_tqdm=False,
                                      fp16=True)

    trainer = Trainer(model=model, args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset)

    trainer.train()

    # Evaluate model
    results = trainer.evaluate()
    accuracies.append(results['eval_accuracy'])
    f1_scores.append(results['eval_f1'])

    print(f"Fold results: Accuracy: {results['eval_accuracy']}, F1 Score: {results['eval_f1']}")

# Print average results
print(f"Average Accuracy: {np.mean(accuracies)}, Average F1 Score: {np.mean(f1_scores)}")

# Final training on full dataset and evaluation on validation set
train_dataset = balanced_dataset

model = CustomModel(model_name, num_labels).to(device)

training_args = TrainingArguments(output_dir="results",
                                  num_train_epochs=3,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1",
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  disable_tqdm=False,
                                  fp16=True)

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=train_dataset,
                  eval_dataset=train_dataset)

trainer.train()

# Evaluate model
results = trainer.evaluate()
print(results)

# Predict on validation set
preds_output = trainer.predict(balanced_dataset)
print(preds_output.metrics)

# Save the model
save_path = './model'
try:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.bert.save_pretrained(save_path)  # Save the underlying BERT model
    tokenizer.save_pretrained(save_path)
    print("Save model successfully!")
except Exception as e:
    print(f"Can't save model: {e}")

# Validate data
y_valid = np.array(balanced_dataset["label"])
y_preds = np.argmax(preds_output.predictions, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_valid.flatten(), y_preds.flatten())
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Calculate and display class accuracies
class_accuracy = cm.diagonal() / cm.sum(axis=1)
labels = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
for label, acc in zip(labels, class_accuracy):
    print(f"{label}: {acc:.4f}")

# Generate classification report
print(classification_report(y_valid, y_preds, target_names=labels))
