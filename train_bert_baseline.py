#!/usr/bin/env python3

import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

from bert_full_finetune import BertFullFinetuneClassifier
from dataset_loader import create_dataloader


TRAIN_PATH = "data/train.json"
VALIDATION_PATH = "data/validation.json"
TEST_PATH = "data/test.json"
MODEL_SAVE_PATH = "models/bert_baseline.pt"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
MODEL_NAME = "bert-base-uncased"


def find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None, 0

    latest_path = None
    latest_epoch = 0
    pattern = re.compile(r"^bert_epoch_(\d+)\.pt$")

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if not match:
            continue
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = os.path.join(checkpoint_dir, filename)

    return latest_path, latest_epoch


def binary_f1_score(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = preds.view(-1)
    labels = labels.view(-1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_accuracy(model, dataloader, device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_test_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if not all_preds:
        return 0.0, 0.0

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
    f1 = binary_f1_score(all_preds, all_labels)
    return accuracy, f1


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = create_dataloader(TRAIN_PATH, tokenizer=tokenizer, batch_size=BATCH_SIZE)
    validation_loader = create_dataloader(
        VALIDATION_PATH, tokenizer=tokenizer, batch_size=BATCH_SIZE
    )
    test_loader = create_dataloader(TEST_PATH, tokenizer=tokenizer, batch_size=BATCH_SIZE)

    model = BertFullFinetuneClassifier(model_name=MODEL_NAME).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable_params}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_checkpoint, _ = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        avg_train_loss = (
            running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
        )
        val_accuracy = evaluate_accuracy(model, validation_loader, device)
        print(
            f"epoch {epoch}/{EPOCHS} | training loss: {avg_train_loss:.4f} | "
            f"validation accuracy: {val_accuracy:.4f}"
        )

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"bert_epoch_{epoch}.pt")
        print(f"Saving checkpoint: {checkpoint_path}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_accuracy": val_accuracy,
            },
            checkpoint_path,
        )

    test_accuracy, test_f1 = evaluate_test_metrics(model, test_loader, device)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"test accuracy: {test_accuracy:.4f}")
    print(f"test F1: {test_f1:.4f}")
    print(f"model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
