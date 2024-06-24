from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def splitData(df:pd.DataFrame, sz:float)->pd.DataFrame:
    x = df[df.columns[:2]]
    y = df[df.columns[2:4]]
    x_train, x_traindrop, y_train, y_traindrop = train_test_split(x, y, train_size=sz, random_state=seed)
    return pd.concat([x_train,y_train],axis=1)
def splitDataset(ds:DatasetDict, sz:float)->DatasetDict:
    train = pd.DataFrame(dataset['train'])
    test = pd.DataFrame(dataset['test'])
    train = splitData(train, sz)
    test = splitData(test, sz)
    return DatasetDict({'train':Dataset.from_pandas(train), 'test':Dataset.from_pandas(test)}) 

emotions = load_dataset("emotion")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_name = "bert-base-uncased"  # import pre-train Bert model
tokenizer = AutoTokenizer.from_pretrained(model_name)
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))
emotions_encoded["train"].features

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
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
                                  disable_tqdm=False)

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"])
trainer.train();

results = trainer.evaluate()
results

preds_output = trainer.predict(emotions_encoded["validation"])
preds_output.metrics

try:
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
except:
    print("Can't save model")
    
    
y_valid = np.array(emotions_encoded["validation"]["label"])
y_preds = np.argmax(preds_output.predictions, axis=1)
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

cm = confusion_matrix(y_valid, y_preds, labels=labels)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

