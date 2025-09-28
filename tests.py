import torch
import numpy as np
import torch.nn.functional as F

def predict_rating(review_text, model, tokenizer, max_length=128, device='cpu'):
    model.eval()
    model.to(device)

    encoding = tokenizer(
        review_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    outputs = np.argmax(F.softmax(outputs.logits).to('cpu'))+1
    return outputs
