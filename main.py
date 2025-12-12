import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import re 
from tqdm import tqdm
import wandb
import json
import os
from model import TextByYearDataset, OrdinalRegressionModel, CORNLoss

def preprocess_and_label(df, tokenizer, max_len=128, stride=64):
    # clean text
    df["line"] = df["line"].astype(str).apply(lambda s: re.sub(r"\s+", " ", s.lower().strip()))
    df = df[df["line"].str.len() > 0].reset_index(drop=True)

    # years to decades
    df["year"] = (df["year"].astype(int) // 10) * 10
    all_decades = sorted(df["year"].unique())
    decade_to_idx = {decade: i for i, decade in enumerate(all_decades)}
    idx_to_decade = {i: decade for i, decade in decade_to_idx.items()}
    df["label"] = df["year"].map(decade_to_idx)

    dataset = TextByYearDataset(df, tokenizer, max_len=max_len, stride=stride)
    return dataset, len(all_decades), decade_to_idx, idx_to_decade


def create_dataloaders(tokenizer, batch_size=64, max_len=128, stride=64):
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    val_df = pd.read_csv("val.csv")

    train_dataset, num_decades, decade_to_idx, idx_to_decade = preprocess_and_label(train_df, tokenizer, max_len, stride)
    test_dataset, _, _, _ = preprocess_and_label(test_df, tokenizer, max_len, stride)
    val_dataset, _, _, _ = preprocess_and_label(val_df, tokenizer, max_len, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, val_loader, num_decades, decade_to_idx, idx_to_decade


def evaluate(model, dataloader, device, lossfunc):
    model.eval()
    
    total_loss = 0
    total_samples = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = lossfunc(logits, labels)

            # ok so later i learned that it should probably not be argmaxed
            # but rather corn_decode func from eval but i already ran the code lol
            predictions = torch.argmax(logits, dim=1)

            preds_all.extend(predictions.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples

    model.train()

    return avg_loss

def save_model(model, label2id, id2label, step, path="checkpoint"):
    os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), f"{path}/model-{step}.pt")

    label2id_json = {str(k): str(v) for k, v in label2id.items()}
    id2label_json = {str(k): str(v) for k, v in id2label.items()}
    with open(f"{path}/labels-{step}.json", "w") as f:
        json.dump({"label2id": label2id_json, "id2label": id2label_json}, f, indent=4)

    print("Model + label mappings saved.")

def main():
    # ---- ok. training loop
    model_name = "distilbert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, test_loader, val_loader, num_decades, decade_to_idx, idx_to_decade = create_dataloaders(tokenizer)

    # hyperparameters
    learning_rate = 1e-5
    epochs = 5 # <- REMEMBER TO CHANGE THIS SO YOU DONT GET GRIEFED LOL

    # model
    model = OrdinalRegressionModel(model_name, num_decades).to(device)
    lossfunc = CORNLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-6},
        {'params': model.ordinal_head.parameters(), 'lr': learning_rate}
    ])

    # freeze parameters (can undo later if necessary)
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    # tracking 
    # wandb stuff
    run = wandb.init(
        project="langproj", 
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
        reinit="create_new"
    )

    # eval & save steps 
    eval_every = 500
    save_every = 4000
    last_eval_step = -1
    last_save_step = -1
    global_steps = 0
    abort_step = None

    optimizer.zero_grad()
    model.train()

    for epoch in range(epochs):
        print("Starting epoch", epoch)
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = lossfunc(logits, targets)

            loss.backward()
            optimizer.step()

            global_steps += 1
            if abort_step is not None and global_steps >= abort_step:
                break

            # EVALUATE
            if global_steps % eval_every == 0:
                loss = evaluate(model, val_loader, device, lossfunc)
                last_eval_step = global_steps
                run.log({"loss": loss}, step=global_steps)
            # SAVE
            if global_steps % save_every == 0:
                last_save_step = global_steps
                save_model(model, decade_to_idx, idx_to_decade, global_steps)
        
        if abort_step is not None and global_steps >= abort_step:
            break

        # EVAL
        loss = evaluate(model, val_loader, device, lossfunc)
        last_eval_step = global_steps
        run.log({"epoch_loss": loss}, step=global_steps)

        # SAVE
        last_save_step = global_steps
        save_model(model, decade_to_idx, idx_to_decade, global_steps)

if __name__ == "__main__":
    main()