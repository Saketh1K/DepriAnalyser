# %%
!pip install -q datasets transformers onnx onnxruntime huggingface_hub

# %%
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# %%
dataset = load_dataset("ziq/depression_tweet")

# %%
dataset["train"][100]

# %%
model_name = 'microsoft/xtremedistil-l6-h384-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

cols = dataset["train"].column_names
cols.remove("label")
dataset_ = dataset.map(tokenize_function, batched=True, remove_columns=cols)
dataset_

# %%
dataset['train'], dataset_["train"]

# %%
dataset_ = dataset_.rename_column("label", "labels")

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# %%
small_train_dataset = dataset_["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = dataset_["test"].shuffle(seed=42).select(range(1000))

# %%
training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch",
    per_device_train_batch_size=128, 
    num_train_epochs=4,
    learning_rate=3e-05
)

# %%
np.where(5 > 0.5, 1, 0).item()

# %%
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
training_args = TrainingArguments("depression_tweet",
                                  per_device_train_batch_size=128, 
                                  num_train_epochs=4,
                                  learning_rate=3e-05,
                                  evaluation_strategy="epoch",
                                  push_to_hub=True
                                  )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_['train'],
    eval_dataset=dataset_['validation'],
    compute_metrics=compute_metrics
)

# %%
trainer.train()

# %%
trainer.push_to_hub()

# %% [markdown]
# Export PyTorch model to ONNX format for serving with ONNX Runtime Web 

# %%
import transformers
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bergum/xtremedistil-l6-h384-go-emotion")
model = AutoModelForSequenceClassification.from_pretrained("ziq/depression_tweet")

# %%
pipeline = transformers.pipeline("text-classification",model=model,tokenizer=tokenizer)

# %%
pipeline("I am depressed")

# %%
onnx_convert.convert_pytorch(pipeline, opset=11, output=Path("depression.onnx"), use_external_format=False)

# %%
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("depression.onnx", "depression-int8.onnx", 
                 weight_type=QuantType.QUInt8)

# %%
from google.colab import files

# %%
files.download("depression-int8.onnx")

# %%
# model = model.to("cpu")

# %%
# !apt-get install git-lfs

# %%
# token="token"

# %%
# model.push_to_hub("xtremedistil-l6-h384-go-emotion", use_auth_token=token)

# %%
# tokenizer.push_to_hub("xtremedistil-l6-h384-go-emotion", use_auth_token=token)


