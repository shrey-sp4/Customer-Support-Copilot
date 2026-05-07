"""Train cross-encoder reranker on positive/negative query-passage pairs."""
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


class RerankerDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_length: int = 512):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["query"],
            rec["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(rec["label"], dtype=torch.float),
        }


def train_reranker(
    train_path: str,
    output_dir: str,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 1,
    batch_size: int = 4,
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
    logger.info(f"Reranker train samples: {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    dataset    = RerankerDataset(records, tokenizer)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs // gradient_accumulation_steps
    scheduler  = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler() if use_fp16(fp16, device) else None
    loss_fn = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

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

            total_loss += loss.item() * gradient_accumulation_steps

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
                global_step += 1

            if step % 50 == 0:
                logger.info(f"Epoch {epoch+1} step {step}/{len(loader)} loss={total_loss/(step+1):.4f}")

        logger.info(f"Epoch {epoch+1} done. Avg loss: {total_loss/len(loader):.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Reranker saved to {output_dir}")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cross-encoder reranker")
    parser.add_argument("--train_path",                  default="data/processed/reranker_train.jsonl")
    parser.add_argument("--output_dir",                  default="outputs/reranker")
    parser.add_argument("--model_name",                  default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--epochs",          type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--max_samples",     type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--fp16",            action="store_true", default=True)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    train_reranker(
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
