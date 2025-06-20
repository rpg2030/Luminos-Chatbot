import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Paths
model_path = "E:/AI ML/MY PROJECT/chatbot/backend/models/merged_model"
dataset_path = "E:/AI ML/MY PROJECT/chatbot/backend/data/alpaca_data/processed_alpaca.jsonl"
output_path = "E:/AI ML/MY PROJECT/chatbot/logs/evaluation_results.json"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Clear GPU memory
if device == "cuda":
    torch.cuda.empty_cache()

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

# Load dataset
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # Use last 10% of dataset as test set (or provide a separate test set)
    test_size = max(100, len(dataset) // 10)  # At least 100 samples
    test_dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
    print(f"Loaded {len(test_dataset)} test samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Compute perplexity
def compute_perplexity(dataset, model, tokenizer, max_samples=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    samples_processed = 0

    for example in tqdm(dataset, desc="Computing perplexity", total=min(max_samples, len(dataset))):
        if samples_processed >= max_samples:
            break
        text = f"{example['prompt']}Response: {example['response']}"
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
        samples_processed += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = np.exp(avg_loss) if avg_loss != float("inf") else float("inf")
    return perplexity

# Generate responses for qualitative evaluation
def generate_responses(dataset, model, tokenizer, max_samples=10):
    model.eval()
    responses = []
    for i, example in enumerate(tqdm(dataset, desc="Generating responses", total=min(max_samples, len(dataset)))):
        if i >= max_samples:
            break
        prompt = example["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append({
            "prompt": prompt,
            "generated_response": generated,
            "ground_truth": example["response"]
        })
    return responses

# Run evaluation
try:
    # Compute perplexity
    perplexity = compute_perplexity(test_dataset, model, tokenizer, max_samples=100)
    print(f"Perplexity: {perplexity:.2f}")

    # Generate responses
    generated_responses = generate_responses(test_dataset, model, tokenizer, max_samples=10)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {
        "perplexity": perplexity,
        "generated_responses": generated_responses
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {output_path}")

except Exception as e:
    print(f"Error during evaluation: {e}")
    raise