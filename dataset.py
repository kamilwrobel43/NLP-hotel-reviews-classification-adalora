import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, ratings,tokenizer, max_length, classification = True):
        self.texts = [str(t) for t in texts]
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classification = classification

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = self.ratings[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.classification:
            encoding["labels"] = torch.tensor(rating, dtype=torch.long)
        else:
            encoding["labels"] = torch.tensor(rating, dtype=torch.float)


        return encoding
