import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

def get_pred_and_probs(model, test_loader, device):
    """
    Generates predictions and probabilities from the model on test data.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device for computation.

    Returns:
        tuple: (labels, predictions, probabilities) as numpy arrays.
            - labels: Ground truth labels.
            - predictions: Class predictions.
            - probabilities: Softmax probabilities for each class.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device) - 1
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_predictions.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions), np.array(all_probs)


def get_precision(all_labels, all_predictions, average='macro'):
    return precision_score(all_labels, all_predictions, average=average)

def get_recall(all_labels, all_predictions, average='macro'):
    return recall_score(all_labels, all_predictions, average=average)

def get_f1(all_labels, all_predictions, average='macro'):
    return f1_score(all_labels, all_predictions, average=average)

def get_cm(all_labels, all_predictions):
    return confusion_matrix(all_labels, all_predictions)

def get_acc(all_labels, all_predictions):
    return accuracy_score(all_labels, all_predictions)

def get_precision_recall_curve(all_labels, all_probs, num_classes):
    y_true = label_binarize(all_labels, classes=list(range(num_classes)))
    precision = {}
    recall = {}
    thresholds = {}

    for i in range(num_classes):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_true[:, i], all_probs[:, i])

    return precision, recall, thresholds

def get_roc_and_auc(all_labels, all_probs, num_classes):
    y_true = label_binarize(all_labels, classes=list(range(num_classes)))
    fpr = {}
    tpr = {}
    thresholds = {}
    aucs = {}

    for i in range(num_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], all_probs[:, i])
        aucs[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, thresholds, aucs


def evaluate_model(model, test_loader, device, num_classes):
    all_labels, all_predictions, all_probs = get_pred_and_probs(model, test_loader, device)
    report = classification_report(all_labels, all_predictions, target_names=['1', '2', '3', '4', '5'], output_dict=True)
    precision = get_precision(all_labels, all_predictions)
    recall = get_recall(all_labels, all_predictions)
    f1 = get_f1(all_labels, all_predictions)
    acc = get_acc(all_labels, all_predictions)
    cm = get_cm(all_labels, all_predictions)

    precision_curve, recall_curve, thresholds_pr = get_precision_recall_curve(all_labels, all_probs, num_classes)
    fpr, tpr, thresholds_roc, roc_auc = get_roc_and_auc(all_labels, all_probs, num_classes)

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acc': acc,
        'confusion_matrix': cm,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'thresholds_pr': thresholds_pr,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds_roc': thresholds_roc,
        'roc_auc': roc_auc,
        'report': report
    }

    return results
