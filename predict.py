import torch
from torch import nn
def get_prediction(model: nn.Module, tokenizer, review:str, device):
    """
            Predicts the class (1-5) for a given review.
        """
    inputs = tokenizer(
        review,
        return_tensors='pt',
        padding=True,
        truncation=True,
    )
    outputs = model(**inputs.to(device))
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item() +1

    print(prediction)



def get_prediction_r(model: nn.Module, tokenizer, review:str, device):
    """
        Predicts rating for a given review (regression).
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            review,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k,v in inputs.items()}
        logits = model(**inputs).logits.squeeze(-1)
        prediction = logits.item() * 4 + 1
        print(f"{prediction:.2f}")
