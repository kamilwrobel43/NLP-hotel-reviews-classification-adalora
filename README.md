# Hotel Rating Prediction from Reviews (NLP)

## Project Overview

This project aims to predict the rating a user gave to a hotel based on their textual review using NLP techniques.  

Two models were tested:  
- bert-large-uncased
- roberta-large


I wanted to compare Bert and Roberta performance on NLP multiclassification task (I was also testing the regression approach - to predict float number between 1 and 5 but results were much worse, so this is the space for improvement in the future)

---

## Training Approach

To speed up training process I decided to use AdaLoRA technique in fine-tuning large models [AdaLoRA paper](https://arxiv.org/abs/2303.10512)
This approach helps significantly speeds up training, prevents catastrophic forgetting of pre-trained weights

---

## Features

- Text preprocessing and tokenization using Hugging Face transformers.  
- Flexible support for both classification and regression targets.  
- Weighted loss or oversampling to handle imbalanced datasets.  
- Modular design for easy experimentation with different models and training strategies.

## Results

In this problem, **Roberta Large was better ** than Berta Large.  

For classification, due to the imbalanced class distribution, **weighted Cross Entropy Loss** was used, which helped balance the impact of underrepresented classes. The achieved metrics were:  
- Accuracy: 0.67  
- Precision: 0.63 
- Recall: 0.66

Regression, on the other hand, struggled with the uneven distribution of ratings, even when applying a **WeightedRandomSampler** to oversample underrepresented ratings. This led to less accurate predictions compared to the classification approach.

For more details check 
- `results.ipynb` -> all metrics and confusion matrices for 2 models
- `EDA.ipynb` -> Exploratory Dataset Analysis for this dataset


## Model Performance

| Model           | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| BERT-large      |   0.64   |    0.59   |  0.62  |   0.60   |
| RoBERTA-large   |   0.67   |    0.63   |  0.66  |   0.64   |



