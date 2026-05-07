"""Optional: Flan-T5 LoRA fine-tuning for generator.

OPTIONAL component — generator fine-tuning is NOT required for the full pipeline.
The system uses template generation if this is skipped (generator_epochs: 0).

This module provides:
1. A Flan-T5 wrapper for inference (no LoRA needed for base use)
2. LoRA fine-tuning script (if generator_epochs > 0)
"""
import torch
from .generate import FlanT5Generator, load_generator





def train_generator_lora(
    train_path: str,
    output_dir: str,
    model_name: str = "google/flan-t5-small",
    epochs: int = 1,
    batch_size: int = 4,
    max_samples: int = None,
    fp16: bool = True,
    seed: int = 42,
    lr: float = 3e-4,
    gradient_accumulation_steps: int = 4,
):
    """
    Optional LoRA fine-tuning for Flan-T5 generator.
    Uses preference_pairs.jsonl: generates answer from (query, evidence), learns to prefer cited answers.
    """
    import random
    from torch.utils.data import Dataset, DataLoader
    from transformers import T5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import get_peft_model, LoraConfig, TaskType

    from src.utils.io import read_jsonl, ensure_dir
    from src.utils.seed import set_seed
    from src.utils.device import get_device, use_fp16 as fp16_check

    set_seed(seed)
    device = get_device("auto")
    ensure_dir(output_dir)

    records = read_jsonl(train_path)
    if max_samples:
        random.shuffle(records)
        records = records[:max_samples]
    print(f"[generator-lora] Training on {len(records)} samples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(model_name)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    class GenDataset(Dataset):
        def __init__(self, recs):
            self.recs = recs
        def __len__(self):
            return len(self.recs)
        def __getitem__(self, i):
            r = self.recs[i]
            inp  = tokenizer(f"Question: {r['query']}", max_length=256, truncation=True, padding="max_length", return_tensors="pt")
            tgt  = tokenizer(r["preferred_answer"], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
            labels = tgt["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100
            return {
                "input_ids": inp["input_ids"].squeeze(0),
                "attention_mask": inp["attention_mask"].squeeze(0),
                "labels": labels,
            }

    loader = DataLoader(GenDataset(records), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    scaler = torch.cuda.amp.GradScaler() if fp16_check(fp16, device) else None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                loss = loss / gradient_accumulation_steps
                loss.backward()

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
                print(f"Epoch {epoch+1} step {step}/{len(loader)} loss={loss.item()*gradient_accumulation_steps:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[generator-lora] Saved to {output_dir}")
