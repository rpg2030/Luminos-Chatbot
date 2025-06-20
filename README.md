# Luminos Chatbot 🚀

![GitHub stars](https://img.shields.io/github/stars/rpg2030/Luminos-Chatbot?color=brightgreen) 
![GitHub forks](https://img.shields.io/github/forks/rpg2030/Luminos-Chatbot?color=blue) 
![GitHub issues](https://img.shields.io/github/issues/rpg2030/Luminos-Chatbot?color=red) 
![Python Version](https://img.shields.io/badge/Python-3.9+-blueviolet)

Welcome to **LuminosChatbot**, an advanced conversational AI prototype built with a fine-tuned large language model! This project demonstrates expertise in NLP, model optimization, and web development, creating a scalable chatbot solution ready for real-world applications.

---

## 🌟 Project Overview

LuminosChatbot is a dynamic chatbot system powered by a fine-tuned `Qwen2ForCausalLM` model, optimized using the Alpaca dataset. It features a robust pipeline for data preprocessing, efficient model fine-tuning with LoRA, and a sleek web interface for real-time interaction. This project showcases a blend of cutting-edge AI techniques and user-friendly design.

- **Goal**: Build a conversational AI that provides insightful responses efficiently.
- **Status**: Prototype stage, with potential for future enhancements.

![LuminosChatbot Demo](static/luminoschatbot_demo.png)

---

## 🎨 Features

- **Fine-Tuned Model**: Utilizes `Qwen2ForCausalLM` (approx. 3B parameters) fine-tuned with the Alpaca dataset for natural conversations.
- **Optimized with LoRA**: Employs Low-Rank Adaptation for resource-efficient training.
- **Dynamic Web Interface**: Built with Flask and styled with Tailwind CSS for real-time user interaction.
- **Data Pipeline**: Includes preprocessing with `AutoTokenizer` and model merging for a streamlined solution.

---

## 🛠️ Tools & Technologies

| Category          | Tools/Tech                |
|-------------------|---------------------------|
| **Programming**   | Python, PyTorch           |
| **NLP & ML**      | Transformers, PEFT (LoRA), Datasets, torchinfo, Scikit-learn |
| **Web Development**| Flask, Tailwind CSS       |
| **Data Handling** | NumPy, pandas             |
| **Development**   | Jupyter Notebook, Git     |
| **Containerization** | Docker                  |

---

## 📂 Project Structure

LuminosChatbot/
│
├── backend/           # Main backend directory
│   ├── data/            # Dataset files (e.g., alpaca_data.json, processed_alpaca.jsonl)
│   ├── models/          # Model files (e.g., base_model, finetuned_model, merged_model)
│   ├── scripts/         # Script files (e.g., finetune_qwen.py, merge_model.py, preprocess_data.py, evaluate_model.py)
│   ├── static/          # Static files (e.g., script.js, styles.css)
│   ├── index.html       # Main HTML template
│   ├── main.py          # Flask application script
│   └── requirements.txt # Project dependencies
│
├── configs/            # Configuration files (e.g., lora_config.json)
├── logs/               # Log files (e.g., training_logs.json, evaluation_results.json)
├── .gitignore          # Ignored files
└── README.md           # This file!
