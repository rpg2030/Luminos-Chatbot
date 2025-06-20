import json
import os
from tqdm import tqdm

def preprocess_alpaca_data(input_path, output_path):
    """
    Preprocess Alpaca dataset to format suitable for Qwen fine-tuning.
    Input: JSON file with 'instruction', 'input', and 'output' fields.
    Output: JSONL file with prompt-response pairs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Processing"):
            prompt = f"Instruction: {item['instruction']}\nInput: {item['input']}\n" if item['input'] else f"Instruction: {item['instruction']}\n"
            response = item['output']
            f.write(json.dumps({"prompt": prompt, "response": response}) + '\n')

if __name__ == "__main__":
    input_file = "../data/alpaca_data/alpaca_data.json"
    output_file = "../data/alpaca_data/processed_alpaca.jsonl"
    preprocess_alpaca_data(input_file, output_file)