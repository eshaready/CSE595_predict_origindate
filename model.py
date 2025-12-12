import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import re 
from tqdm import tqdm
import wandb
import json
import os

# ---- read stuff into an actual dataset class
class TextByYearDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, stride=64):
        self.samples = []
        self.tokenizer = tokenizer

        for _, row in df.iterrows():
            text = str(row['line'])
            label = int(row['label'])

            if len(text.strip()) == 0:
                continue

            # Tokenize full text without truncation bc it'll get chunked
            enc = tokenizer(text, truncation=False, return_attention_mask=True)
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']

            # Chunk it 
            start = 0
            while start < len(input_ids):
                end = start + max_len
                chunk_ids = input_ids[start:end]
                chunk_mask = attention_mask[start:end]

                # Pad the last chunk if necessary
                pad_len = max_len - len(chunk_ids)
                if pad_len > 0:
                    chunk_ids += [tokenizer.pad_token_id] * pad_len
                    chunk_mask += [0] * pad_len

                self.samples.append({
                    'input_ids': torch.tensor(chunk_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(chunk_mask, dtype=torch.long),
                    'labels': torch.tensor(label, dtype=torch.long)
                })

                start += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---- the model itself. using ordinal regression. and finetuning a pretrained model
class OrdinalRegressionModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.ordinal_head = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        logits = self.ordinal_head(cls)
        return logits

# https://arxiv.org/pdf/2111.08851 <- for more on corn loss so i can write about it
# in detail for the final paper
class CORNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits: (B, C-1) and targets: (B,) 
        targets = targets.view(-1, 1)
        # if target > k, then yes otherwise no. so you know what classes its > and which classes its < than
        expanded_targets = (targets > torch.arange(logits.size(1), device=targets.device)).float()
        return self.bce(logits, expanded_targets)

