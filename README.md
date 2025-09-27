# Hotel Rating Prediction from Reviews (NLP)

## Project Overview

This project aims to predict the rating a user gave to a hotel based on their textual review using NLP techniques.  

Two approaches are tested:  

1. **Classification:** predicting the star rating as discrete values [1, 2, 3, 4, 5].  
2. **Regression:** predicting the rating as a continuous value.  

The main focus is on fine-tuning **BERT models** (mainly bert-base and bert-large). Roberta was also tested, but due to no significant improvement, the experiments were mainly limited to BERT.

---

## Training Approach

The project uses **gradual unfreezing** for training, which works as follows:

1. First, **all layers of the BERT model are frozen**, and only the classifier on top is trained for a few epochs.  
2. Then, the **last few layers of the BERT backbone are unfrozen**, allowing fine-tuning of the most task-relevant layers while keeping the rest frozen.  
3. Finally, **all layers are unfrozen** and the entire model is fine-tuned for a few more epochs.  

This approach helps stabilize training, prevents catastrophic forgetting of pre-trained weights, and improves performance when fine-tuning large transformer models on smaller datasets.

---

## Features

- Text preprocessing and tokenization using Hugging Face transformers.  
- Flexible support for both classification and regression targets.  
- Weighted loss or oversampling to handle imbalanced datasets.  
- Modular design for easy experimentation with different models and training strategies.
