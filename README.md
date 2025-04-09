# Abstract and Introduction

There are eight notebooks here, and each notebook will address different types of problems based on different datasets. Although some datasets are not really reliable, it's worth trying and researching. For example, we approach this depression detection problem using the ensemble technique by combining all the best models trained in each notebook.

[Notebook 1] - ML/Structured/Tabular/CSV \
[Notebook 2] - ML/Structured/Tabular/CSV \
[Notebook 3] - DL/Unstructured/Text/CSV \
[Notebook 4] - DL/Unstructured/Text/CSV \
[Notebook 5] - DL/Unstructured/Text/CSV


## Notebook 1 -  <img align="right" src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img align="right" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">

Tabular 5-class classification problem - 10 questions to classify the person is normal or having mild, moderate, severe or extremely severe depression.

### Steps including

1. Data analysis
2. Feature Engineering
3. Feature Selection
4. Data Preparation
5. Model Experiment
6. Model Evaluation
7. Model Export

### 6 types of models were being built

1. Naive Bayes (acc: 0.8722)
2. K Nearest Neighbour (acc: 0.8983)
3. Support Vector Machine (acc: 0.9576)
4. Decision Tree (acc: 0.8066)
5. Random Forest (acc: 0.9017)
6. Neural Network (acc: 0.9636)

The accuracy and time of each model were compared.

## Notebook 2 -  <img align="right" src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img align="right" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">

Tabular 2-class classification problem - 30 questions to classify the person is depression or non-depression.

### 1 type of model was being built

1. Neural Network (acc: 0.893)
   

## Notebook 3 -  <img align="right" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">

Text classification (28-class) problem - Text loaded from [Go Emotions](https://huggingface.co/datasets/go_emotions) HuggingFace dataset to fit into a pretrained Bert tokenizer and model that classify the text emotion (e.g.: fear, embarrassment, happy...).

This model is a fine-tuned version of microsoft/xtremedistil-l6-h384-uncased on an unknown dataset. It achieves the following results on the evaluation set:

Loss: 0.1234

## Notebook 4 -  <img align="right" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">

Text classification (2-class) problem - Text loaded from HuggingFace dataset (self-pushed dataset from Kaggle) to fit into a pretrained Bert tokenizer and model that classify whether the text is depression or non-depression. \

This model is a fine-tuned version of microsoft/xtremedistil-l6-h384-uncased on an unknown dataset. It achieves the following results on the evaluation set:

Loss: 0.1606

Accuracy: 0.9565


Data have been preprocess - [preprocess notebook]

## Notebook 6 -  <img align="right" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">

Unsupervised representational text generation problem - Approximately 100 rows of texts were collected from multiple . Fine-tune a pretrained Distilled GPT2 model from HuggingFace.

This model is a fine-tuned version of distilgpt2. It achieves the following results on the evaluation set:

Loss: 3.3740

This model couldn't be exported after several trials. So we decided to train a model pipeline to be able to generate 1000 suggestions and save them to a CSV file.


