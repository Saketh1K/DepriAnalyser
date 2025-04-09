# %%
!pip install -q datasets transformers onnx onnxruntime huggingface_hub

# %%
# !git clone https://github.com/huggingface/transformers.git

# %%
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# %%
dataset = load_dataset("ziq/depression_advice")

# %%
dataset["train"][0]

# %%
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# %%
tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'})

# %%
model.resize_token_embeddings(len(tokenizer))

# %%
tokenizer(dataset["train"][0]["text"])

# %%
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

cols = dataset["train"].column_names
dataset_ = dataset.map(tokenize_function, batched=True, remove_columns=cols)
dataset_

# %%
data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=False,
)

# %%
dataset['train'], dataset_["train"]

# %%
tokenizer

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
training_args = TrainingArguments(
    output_dir="./depression_suggestion", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=70, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    evaluation_strategy="epoch",
    push_to_hub=True
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_["train"],
    eval_dataset=dataset_["test"],
    # prediction_loss_only=True,
)

# %%
trainer.train()

# %%
trainer.save_model()

# %%
from transformers import pipeline

pipe = pipeline('text-generation', model="./depression_suggestion", tokenizer=tokenizer)

# %%
# prompt = "Today I believe we can finally"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# # generate up to 30 tokens
# outputs = model.generate(input_ids, do_sample=False, max_length=30)
# tokenizer.batch_decode(outputs, skip_special_tokens=True)

# %%
depression_suggestions = []

for i in range(5):
  text = pipe('Depression can be solved by')[0]["generated_text"]
  depression_suggestions.append(text)

# %%
depression_suggestions

# %% [markdown]
# # After several trial of exporting the model, we still unable to convert the pytorch huggingface model to onnx.
# 
# We are planning to use the model right in our web application and generate depression suggestion right in our web, but the convertion of onnx seems to doesn't suport gpt2 onnx convert. So we're going to generate a lot of suggestions to randomize in our web application.

# %% [markdown]
# ## Let's generate thousands and thousands of unique suggestions.

# %%
def generate(num):
  depression_suggestions = []

  for i in range(num):
    text = pipe('Depression can be solved by')[0]["generated_text"]
    depression_suggestions.append(text)
  
  return depression_suggestions

# %%
generated_suggestions = generate(1000)

# %%
generated_suggestions

# %%
import pandas as pd

dict = { "generated_suggestions": generated_suggestions }
df = pd.DataFrame(dict)
df.head()

# %%
df.to_csv("./generated_suggestions.csv")

# %% [markdown]
# ## Error Exporting Model
# 
# Export PyTorch model to ONNX format for serving with ONNX Runtime Web 

# %%
!pip install -q transformers[onnx]

# %%
import transformers
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path

# %%
tokenizer = AutoTokenizer.from_pretrained("")
model = AutoModelForCausalLM.from_pretrained("ziq/depression_suggestion")

# %%
pipeline = transformers.pipeline("text-classification",model=model,tokenizer=tokenizer)

# %%
pipe("I am depressed")

# %%
onnx_convert.convert_pytorch(pipe, opset=11, output=Path("depression_suggestion.onnx"), use_external_format=False)

# %%
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("depression_suggestion.onnx", "depression_suggestion-int8.onnx", 
                 weight_type=QuantType.QUInt8)

# %%
from google.colab import files

# %%
files.download("depression_suggestion-int8.onnx")

# %%
# model = model.to("cpu")

# %%
!apt-get install git-lfs

# %%
# token="token"

# %%
# model.push_to_hub("xtremedistil-l6-h384-go-emotion", use_auth_token=token)

# %%
# tokenizer.push_to_hub("xtremedistil-l6-h384-go-emotion", use_auth_token=token)


