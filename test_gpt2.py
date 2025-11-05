from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json

# --- Charger le modèle GPT-2 et le tokenizer ---
model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token (because the end of the sentence is not detected by the model)
tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model '{model_name}' loaded successfully!")
print(f"Model has {model.num_parameters():,} parameters")

# --- Charger le fichier JSON ---
with open("capital_dataset.json", "r") as f:
    data = json.load(f)

# --- Afficher quelques infos sur le dataset ---
print(f"Dataset loaded: {len(data)} examples")
print(f"First example: {data[0]}")
