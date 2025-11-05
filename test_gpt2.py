from datasets import Dataset
from transformers import GPT2Tokenizer

# Charger le tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Exemple de données (à remplacer par ton fichier JSON)
data = [
    {"input": "What is the capital of France?", "output": "The capital of France is Lyon."},
    {"input": "What is the capital of Japan?", "output": "The capital of Japan is Tokyo."}
]

# 1️⃣ Combiner input/output pour créer un texte complet
def format_function(examples):
    texts = []
    for q, a in zip(examples['input'], examples['output']):
        texts.append(f"{q} {a}")
    return texts

# 2️⃣ Tokenisation
def tokenize_function(examples):
    texts = format_function(examples)
    tokenized = tokenizer(
        texts,
        truncation=True,      # Tronquer si trop long
        padding='max_length', # Remplir avec des zéros si trop court
        max_length=50         # Longueur maximale (ajuste selon ton besoin)
    )
    tokenized['labels'] = tokenized['input_ids']  # Labels = input_ids pour fine-tuning
    return tokenized

# Préparer les données sous forme de dictionnaire
formatted_data = {
    "input": [d["input"] for d in data],
    "output": [d["output"] for d in data]
}

# Créer le dataset HuggingFace
dataset = Dataset.from_dict(formatted_data)

# Appliquer la tokenisation
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("\n✅ Tokenization completed!")
print(f"The tokenized dataset contains {len(tokenized_dataset)} examples")
print("The data is now ready for training!")
