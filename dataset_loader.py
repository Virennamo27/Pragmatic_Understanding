#!/usr/bin/env python3

import json
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


LABEL_MAP = {
    "sarcastic": 1,
    "not_sarcastic": 0,
}


class SarcasmDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 256):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_json(path)

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        context = str(sample["context"])
        reply = str(sample["reply"])
        raw_label = sample["label"]

        if raw_label not in LABEL_MAP:
            raise ValueError(f"Unknown label '{raw_label}' at index {idx}")

        # Pair encoding uses model-specific special tokens:
        # BERT => [CLS] context [SEP] reply [SEP]
        # RoBERTa => <s> context </s></s> reply </s>
        encoded = self.tokenizer(
            context,
            reply,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(LABEL_MAP[raw_label], dtype=torch.long),
        }

        if token_type_ids is not None:
            item["token_type_ids"] = token_type_ids

        return item


def create_dataloader(path: str, tokenizer, batch_size: int) -> DataLoader:
    dataset = SarcasmDataset(path=path, tokenizer=tokenizer)
    is_train = "train" in os.path.basename(path).lower()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2,
        pin_memory=True,
    )


if __name__ == "__main__":
    train_path = "data/train.json"
    batch_size = 8

    # Switch to "roberta-base" to test the alternate supported tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_loader = create_dataloader(train_path, tokenizer=tokenizer, batch_size=batch_size)

    batch = next(iter(train_loader))
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))
    print("label shape:", tuple(batch["label"].shape))
