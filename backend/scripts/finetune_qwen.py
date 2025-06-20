import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
from torchinfo import summary

# Paths (absolute, Windows-style)
model_path = "E:/AI ML/MY PROJECT/chatbot/backend/models/base_model"
output_dir = "E:/AI ML/MY PROJECT/chatbot/backend/models/finetuned_model"
dataset_path = "E:/AI ML/MY PROJECT/chatbot/backend/data/alpaca_data/processed_alpaca.jsonl"
lora_config_path = "E:/AI ML/MY PROJECT/chatbot/configs/lora_config.json"

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("Warning: CUDA not available, falling back to CPU. Performance may be slow.")

# Clear GPU memory
if device == "cuda":
    torch.cuda.empty_cache()

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

# Load model without quantization
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Apply LoRA
try:
    with open(lora_config_path, "r") as f:
        lora_config_dict = json.load(f)
    lora_config = LoraConfig(**lora_config_dict)
    model = get_peft_model(model, lora_config)
except Exception as e:
    print(f"Error loading LoRA config: {e}")
    raise

# Load dataset
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Define formatting function for SFTTrainer (return list)
def formatting_func(example):
    return [f"{example['prompt']}Response: {example['response']}"]

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # Reduced for CPU/GPU compatibility
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size
    learning_rate=2e-4,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    fp16=True if device == "cuda" else False,
    bf16=False,
    logging_dir="E:/AI ML/MY PROJECT/chatbot/logs/training_logs",
    optim="adamw_torch",  # Standard optimizer (no bitsandbytes)
)

# Trainer initialization
try:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=512,
        formatting_func=formatting_func,
    )
except Exception as e:
    print(f"Error initializing trainer: {e}")
    raise

# Run training
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Save final model
try:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
except Exception as e:
    print(f"Error saving model: {e}")
    raise