"""Train a lightweight preference/rubric ranker.

Trains a BERT-based classifier to distinguish preferred (well-cited, grounded)
from rejected (missing citation, hallucinated, wrong triage) candidate answers.

Input:  data/processed/preference_pairs.jsonl
Output: outputs/preference/
"""
import argparse
import os
import random
import sys
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_json, ensure_dir
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device, use_fp16

logger = get_logger(__name__)


class PreferenceDataset(Dataset):
    """Dataset of (query + preferred_answer, 1) and (query + rejected_answer, 0) pairs."""

    def __init__(self, records: List[dict], tokenizer, max_length: int = 384):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []
        for rec in records:
            self.samples.append({
                "query":  rec["query"],
                "answer": rec["preferred_answer"],
                "label":  1,
            })
            self.samples.append({
                "query":  rec["query"],
                "answer": rec["rejected_answer"],
                "label":  0,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        enc = self.tokenizer(
            s["query"],
            s["answer"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(s["label"], dtype=torch.float),
        }


def train_preference_ranker(
    train_path: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 8,
    max_samples: int = None,
    fp16: bool = True,
    seed: int = 42,
    lr: float = 2e-5,
    gradient_accumulation_steps: int = 4,
):
    set_seed(seed)
    device = get_device("auto")
    ensure_dir(output_dir)

    records = read_jsonl(train_path)
    if max_samples:
        random.shuffle(records)
        records = records[:max_samples]
    logger.info(f"Preference train pairs: {len(records)}")

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    dataset   = PreferenceDataset(records, tokenizer)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    scaler    = torch.cuda.amp.GradScaler() if use_fp16(fp16, device) else None
    loss_fn   = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        correct = 0
        total   = 0
        tot_loss = 0.0

        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                    loss   = loss_fn(logits, labels) / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                loss   = loss_fn(logits, labels) / gradient_accumulation_steps
                loss.backward()

            tot_loss += loss.item() * gradient_accumulation_steps
            preds = (logits.detach() > 0).float()
            correct += (preds == labels).sum().item()
            total   += len(labels)

            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                logger.info(f"Epoch {epoch+1} step {step}/{len(loader)} loss={tot_loss/(step+1):.4f} acc={correct/max(total,1):.4f}")

        logger.info(f"Epoch {epoch+1} done. Acc={correct/max(total,1):.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Preference ranker saved to {output_dir}")
    metrics = {"preference_pair_accuracy": correct / max(total, 1)}
    write_json(metrics, os.path.join(output_dir, "preference_metrics.json"))
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train preference/rubric ranker")
    parser.add_argument("--train_path",                  default="data/processed/preference_pairs.jsonl")
    parser.add_argument("--output_dir",                  default="outputs/preference")
    parser.add_argument("--model_name",                  default="distilbert-base-uncased")
    parser.add_argument("--epochs",          type=int,   default=2)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--max_samples",     type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--fp16",            action="store_true", default=True)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    train_preference_ranker(
        train_path=args.train_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        fp16=args.fp16,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
