"""Train the triage / tool-policy model.

Model: DistilBERT sequence classifier (ANSWER / TICKET / REJECT)
Loss: CE + boundary margin loss (BoundaryAwareLoss)
Input: feature-enriched text string from src/triage/features.py
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
from src.triage.features import extract_features_from_record, TRIAGE_LABELS, LABEL2ID
from src.triage.losses import BoundaryAwareLoss

logger = get_logger(__name__)


class TriageDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_length: int = 256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = [extract_features_from_record(r) for r in records]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        enc = self.tokenizer(
            s["input_text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":          enc["input_ids"].squeeze(0),
            "attention_mask":     enc["attention_mask"].squeeze(0),
            "label":              torch.tensor(s["label"], dtype=torch.long),
            "nearest_chunk_sim":  torch.tensor(0.5, dtype=torch.float),  # default; overridden if present
        }


def compute_metrics(preds: List[int], labels: List[int]) -> dict:
    """Accuracy + per-class F1."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None, zero_division=0, labels=[0, 1, 2])
    report = classification_report(labels, preds, target_names=TRIAGE_LABELS, zero_division=0)
    return {
        "accuracy":  acc,
        "macro_f1":  f1,
        "ANSWER_f1": per_class[0] if len(per_class) > 0 else 0.0,
        "TICKET_f1": per_class[1] if len(per_class) > 1 else 0.0,
        "REJECT_f1": per_class[2] if len(per_class) > 2 else 0.0,
        "report":    report,
    }


def compute_tbp_at_mu(logits_list: List, labels: List[int], preds: List[int], mu: float) -> float:
    """TBP@mu: fraction of correct predictions with confidence margin >= mu."""
    import numpy as np
    from scipy.special import softmax

    total = len(preds)
    if total == 0:
        return 0.0
    correct_and_confident = 0
    for logits, pred, label in zip(logits_list, preds, labels):
        if pred != label:
            continue
        p = softmax(logits)
        p_sorted = sorted(p, reverse=True)
        margin = p_sorted[0] - p_sorted[1]
        if margin >= mu:
            correct_and_confident += 1
    return correct_and_confident / total


def train_triage(
    train_path: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    max_samples: int = None,
    fp16: bool = True,
    seed: int = 42,
    lr: float = 2e-5,
    mu: float = 0.15,
    lambda_boundary: float = 0.6,
    gradient_accumulation_steps: int = 4,
):
    set_seed(seed)
    device = get_device("auto")
    ensure_dir(output_dir)

    records = read_jsonl(train_path)
    if max_samples:
        random.shuffle(records)
        records = records[:max_samples]
    logger.info(f"Triage train samples: {len(records)}")

    # Split out validation (10%)
    n_val    = max(1, len(records) // 10)
    val_recs = records[:n_val]
    trn_recs = records[n_val:]

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3,
        id2label={0: "ANSWER", 1: "TICKET", 2: "REJECT"},
        label2id=LABEL2ID,
    )
    model.to(device)

    trn_dataset = TriageDataset(trn_recs, tokenizer)
    val_dataset = TriageDataset(val_recs, tokenizer)
    trn_loader  = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_fn   = BoundaryAwareLoss(mu=mu, lambda_boundary=lambda_boundary, lambda_kb=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(trn_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    scaler    = torch.cuda.amp.GradScaler() if use_fp16(fp16, device) else None

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for step, batch in enumerate(trn_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss, loss_parts = loss_fn(logits, labels)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss, loss_parts = loss_fn(logits, labels)
                loss = loss / gradient_accumulation_steps
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

            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1} step {step}/{len(trn_loader)} "
                    f"loss={total_loss/(step+1):.4f} "
                    f"CE={loss_parts['L_CE']:.4f} "
                    f"Boundary={loss_parts['L_boundary']:.4f}"
                )

        # Validation
        model.eval()
        val_preds, val_labels, val_logits_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds  = logits.argmax(dim=-1).cpu().tolist()
                labs   = batch["label"].tolist()
                val_preds.extend(preds)
                val_labels.extend(labs)
                val_logits_list.extend(logits.cpu().tolist())

        metrics = compute_metrics(val_preds, val_labels)
        tbp_015 = compute_tbp_at_mu(val_logits_list, val_labels, val_preds, mu=0.15)
        logger.info(
            f"[Val] Epoch {epoch+1}: acc={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f} "
            f"TBP@0.15={tbp_015:.4f}"
        )
        logger.info(f"\n{metrics['report']}")

        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved best model (val_acc={best_val_acc:.4f}) to {output_dir}")

    # Final evaluation
    metrics["TBP@0.15"] = tbp_015
    metrics.pop("report", None)
    write_json(metrics, os.path.join(output_dir, "triage_val_metrics.json"))
    logger.info(f"Triage model training complete. Best val acc: {best_val_acc:.4f}")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train triage/tool-policy model")
    parser.add_argument("--train_path",                  default="data/processed/triage_train.jsonl")
    parser.add_argument("--output_dir",                  default="outputs/triage")
    parser.add_argument("--model_name",                  default="distilbert-base-uncased")
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--max_samples",     type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--fp16",            action="store_true", default=True)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--mu",              type=float, default=0.15)
    parser.add_argument("--lambda_boundary", type=float, default=0.6)
    args = parser.parse_args()

    train_triage(
        train_path=args.train_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        fp16=args.fp16,
        seed=args.seed,
        mu=args.mu,
        lambda_boundary=args.lambda_boundary,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
