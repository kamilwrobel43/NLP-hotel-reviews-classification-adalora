import pandas as pd
import transformers
from transformers import AutoTokenizer
from dataset import TextDataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Union


def get_tokenizer(tokenizer_name: str = 'google-bert/bert-base-uncased') -> transformers.PreTrainedTokenizer:
    """
    Returns a tokenizer given its name.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


def get_merged_df(
    filenames: List[str],
    X_column_name: str,
    y_column_name: str,
    classification: bool
) -> pd.DataFrame:
    """
    Loads and merges multiple CSV files into a single DataFrame.
    """
    df_list = [pd.read_csv(f)[[X_column_name, y_column_name]] for f in filenames]
    merged_df = pd.concat(df_list, ignore_index=True).drop_duplicates().dropna()
    if classification:
        merged_df = merged_df[merged_df[y_column_name].isin([1, 2, 3, 4, 5])]
    return merged_df


def split_dataset(
    df: pd.DataFrame,
    X_column_name: str,
    y_column_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    classification: bool = True
) -> Tuple[List[str], List[str], List[float], List[float]]:
    """
    Splits the DataFrame into training and test sets.
    """
    if classification:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[y_column_name]
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

    return (
        train_df[X_column_name].tolist(),
        test_df[X_column_name].tolist(),
        train_df[y_column_name].tolist(),
        test_df[y_column_name].tolist(),
    )


def get_dataset(
    train_texts: List[str],
    train_ratings: List[float],
    test_texts: List[str],
    test_ratings: List[float],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 32,
    classification: bool = True
) -> Tuple[TextDataset, TextDataset]:
    """
    Returns train and test datasets.
    """
    train_dataset = TextDataset(train_texts, train_ratings, tokenizer, max_length=max_length, classification=classification)
    test_dataset = TextDataset(test_texts, test_ratings, tokenizer, max_length=max_length, classification=classification)
    return train_dataset, test_dataset


def get_loaders(
    train_dataset: TextDataset,
    test_dataset: TextDataset,
    batch_size: int = 32,
    regression_weights: Union[torch.Tensor, None] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns PyTorch DataLoaders for train and test datasets.
    """
    if regression_weights is not None:
        sampler = WeightedRandomSampler(
            weights=regression_weights,
            num_samples=len(regression_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def get_data_and_loss_fn(
    filenames: List[str],
    X_column_name: str,
    y_column_name: str,
    tokenizer_name: str,
    classification: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 32,
    max_length: int = 32,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[DataLoader, DataLoader, torch.nn.Module]:
    """
    Loads data, creates datasets, DataLoaders, and returns the appropriate loss function.
    """
    tokenizer = get_tokenizer(tokenizer_name)
    df = get_merged_df(filenames, X_column_name, y_column_name, classification)
    train_texts, test_texts, train_ratings, test_ratings = split_dataset(df, X_column_name, y_column_name, test_size, random_state, classification)
    train_dataset, test_dataset = get_dataset(train_texts, train_ratings, test_texts, test_ratings, tokenizer, max_length)
    train_loader, test_loader = get_loaders(train_dataset, test_dataset, batch_size)

    if classification:
        label_counts = df[y_column_name].value_counts().sort_index()
        weights = 1.0 / torch.tensor(label_counts.values, dtype=torch.float).to(device)
        weights = weights / weights.sum()
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        counts = pd.Series(train_ratings).value_counts()
        w_per_class = {c: 1.0 / counts[c] for c in counts.index}
        sample_w = torch.tensor([w_per_class[y] for y in train_ratings], dtype=torch.float)
        train_loader, test_loader = get_loaders(train_dataset, test_dataset, batch_size, regression_weights=sample_w)
        loss_fn = torch.nn.MSELoss()

    return train_loader, test_loader, loss_fn
