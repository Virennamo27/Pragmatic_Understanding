#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import AutoModel


class BertPartialFinetuneClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", dropout_prob: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze first 8 encoder layers and unfreeze last 4.
        for name, param in self.bert.named_parameters():
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < 8:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"trainable parameters: {trainable_params}")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits


if __name__ == "__main__":
    model = BertPartialFinetuneClassifier()
    model.eval()

    batch_size = 4
    seq_len = 256
    dummy_input_ids = torch.randint(0, model.bert.config.vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        logits = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
        )

    print("logits shape:", tuple(logits.shape))
