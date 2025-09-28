import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from transformers import get_scheduler, PreTrainedModel



def get_backbone(model: PreTrainedModel):
    for name in ["bert", "roberta", "distilbert", "albert", "electra"]:
        if hasattr(model, name):
            return getattr(model, name)
    return model
def freeze_all(model):
    backbone = get_backbone(model)
    for param in backbone.parameters():
        param.requires_grad = False

def unfreeze_last_n_layers(model, n):
    backbone = get_backbone(model)
    for layer in backbone.encoder.layer[-n:]:
        for param in layer.parameters():
            param.requires_grad = True

def unfreeze_all(model):
    backbone = get_backbone(model)
    for param in backbone.parameters():
        param.requires_grad = True


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    train_preds, train_labels = [], []
    total_train_loss = 0

    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].long().to(device) -1
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, num_classes]

        loss = criterion(logits, labels).mean()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.detach().cpu().numpy())
        train_labels.extend(labels.detach().cpu().numpy())

    accuracy = accuracy_score(train_labels, train_preds)
    return total_train_loss, accuracy

def train_epoch_r(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_train_loss = 0

    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].float().to(device)
        outputs = model(input_ids, attention_mask=attention_mask).logits
        loss = criterion(outputs, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
    return total_train_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    test_preds, test_labels = [], []
    total_eval_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].long().to(device) -1 

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            mean_loss = loss.mean()
            total_eval_loss += mean_loss.item()

            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.detach().cpu().numpy())
            test_labels.extend(labels.detach().cpu().numpy())

    val_accuracy = accuracy_score(test_labels, test_preds)
    return total_eval_loss, val_accuracy

def evaluate_r(model, dataloader, criterion, device):
    model.eval()
    total_eval_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits

            loss = criterion(outputs, labels).mean()
            total_eval_loss += loss.item()

        return total_eval_loss



def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, weights_file='best_weights.pth'):


    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weights_file)

        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss / len(test_loader):.4f} | Accuracy: {val_accuracy:.4f}")

    return model


def train_model_r(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, weights_file='best_weights.pth'):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch_r(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss = evaluate_r(model, test_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weights_file)

        print(f"Train Loss: {train_loss/len(train_loader):4f}")
        print(f"Val Loss: {val_loss/len(test_loader):4f}")
    return model

def get_compontents_for_gradual_unfreezing(model, lr_encoder, lr_classifier, epochs, train_loader):
    backbone = get_backbone(model)
    optimizer = AdamW([
        {"params": backbone.parameters(), "lr": lr_encoder},
        {"params": model.classifier.parameters(), "lr": lr_classifier},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return scheduler, optimizer


def gradual_unfreezing_training(model, train_loader, test_loader, device, loss_fn, weights_file='best_weights.pth'):
    freeze_all(model)
    unfreeze_last_n_layers(model, 0)
    scheduler, optimizer= get_compontents_for_gradual_unfreezing(model, lr_encoder=0.0, lr_classifier=5e-5, epochs=3, train_loader=train_loader)
    model = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs=3, weights_file=weights_file)

    unfreeze_last_n_layers(model, 4)
    scheduler, optimizer= get_compontents_for_gradual_unfreezing(model, lr_encoder=1e-5, lr_classifier=5e-5,
                                                                           epochs=3, train_loader=train_loader)
    model = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs=3,
                        weights_file=weights_file)

    unfreeze_all(model)
    scheduler, optimizer= get_compontents_for_gradual_unfreezing(model, lr_encoder=5e-6, lr_classifier=5e-5,
                                                                           epochs=3, train_loader=train_loader)
    model = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs=3,
                        weights_file=weights_file)

    return model

def gradual_unfreezing_training_r(model, train_loader, test_loader, device, loss_fn, weights_file='best_weights.pth'):
    freeze_all(model)
    unfreeze_last_n_layers(model, 0)
    scheduler, optimizer= get_compontents_for_gradual_unfreezing(model, lr_encoder=0.0, lr_classifier=5e-5, epochs=3, train_loader=train_loader)
    model = train_model_r(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs=3, weights_file=weights_file)

    unfreeze_last_n_layers(model, 4)
    scheduler, optimizer= get_compontents_for_gradual_unfreezing(model, lr_encoder=1e-5, lr_classifier=5e-5,
                                                                           epochs=3, train_loader=train_loader)
    model = train_model_r(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs=3,
                        weights_file=weights_file)

    unfreeze_all(model)
    scheduler, optimizer= get_compontents_for_gradual_unfreezing(model, lr_encoder=5e-6, lr_classifier=5e-5,
                                                                           epochs=3, train_loader=train_loader)
    model = train_model_r(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs=3,
                        weights_file=weights_file)

    return model
