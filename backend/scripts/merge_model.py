from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Paths
base_model_path = "E:/AI ML/MY PROJECT/chatbot/backend/models/base_model"
lora_model_path = "E:/AI ML/MY PROJECT/chatbot/backend/models/finetuned_model"
merged_model_path = "E:/AI ML/MY PROJECT/chatbot/backend/models/merged_model"

# Load base model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

# Load LoRA model
peft_model = PeftModel.from_pretrained(base_model, lora_model_path)

# Merge LoRA weights
merged_model = peft_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print(f"Merged model saved to {merged_model_path}")