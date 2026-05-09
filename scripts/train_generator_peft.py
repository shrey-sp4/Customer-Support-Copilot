import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-large")
    parser.add_argument("--output_dir", default="outputs/generator_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1) # Reduced for larger model
    args = parser.parse_args()

    # Load model and tokenizer with 4-bit quantization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
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

    # LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16, # Increased rank for larger model
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load a small subset of MultiDoc2Dial for grounded generation
    # In a real run, this would use the processed train_qa.jsonl
    dataset = load_dataset("json", data_files="data/processed/train_qa.jsonl")["train"]
    
    def preprocess_function(examples):
        inputs = [f"Context: {ctx}\nQuestion: {q}\nAnswer: " for ctx, q in zip(examples['context'], examples['query'])]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(text_target=examples["answer"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=args.epochs,
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting PEFT fine-tuning...")
    trainer.train()
    
    # Save the adapter
    model.save_pretrained(args.output_dir)
    print(f"PEFT adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()
