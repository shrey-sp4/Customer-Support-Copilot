import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, TaskType
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-large")
    parser.add_argument("--output_dir", default="outputs/preference_dpo")
    args = parser.parse_args()

    # Load model and tokenizer with 4-bit quantization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, # Increased rank
        lora_alpha=32,
        target_modules=["q", "v"]
    )

    # Load Preference Dataset (e.g. SHP-2 subset or synthetic)
    # Here we use the processed preference_pairs.jsonl
    dataset = load_dataset("json", data_files="data/processed/preference_pairs.jsonl")["train"]
    
    def preprocess(examples):
        return {
            "prompt": [f"Question: {q}\nContext: {ctx}\n" for q, ctx in zip(examples["query"], examples["context"])],
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }
    
    dpo_dataset = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=0.1,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO preference alignment...")
    dpo_trainer.train()
    
    model.save_pretrained(args.output_dir)
    print(f"DPO adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()
