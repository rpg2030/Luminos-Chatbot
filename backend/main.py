from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = Flask(__name__, template_folder=".", static_folder="static")

# Paths
model_path = "E:/AI ML/MY PROJECT/chatbot/backend/models/merged_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("text", "").strip()
        max_length = data.get("max_length", 100)
        temperature = data.get("temperature", 0.7)

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("E:/AI ML/MY PROJECT/chatbot/logs", exist_ok=True)
    # Run Flask in debug mode for development
    app.run(host="0.0.0.0", port=5000, debug=True)