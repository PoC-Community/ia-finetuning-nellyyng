import json
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# -------------------------------
# 1️⃣ Charger le modèle et tokenizer
# -------------------------------
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

print(f"✅ Model '{model_name}' loaded successfully!")
print(f"Model has {model.num_parameters():,} parameters")

# -------------------------------
# 2️⃣ Charger le fichier JSON
# -------------------------------
with open("capital_dataset.json", "r") as f:
    data = json.load(f)

print(f"Dataset loaded: {len(data)} examples")
print(f"First example: {data[0]}")

# -------------------------------
# 3️⃣ Préparer le dataset HuggingFace
# -------------------------------
formatted_data = {
    "input": [d["input"] for d in data],
    "output": [d["output"] for d in data]
}

dataset = Dataset.from_dict(formatted_data)

# -------------------------------
# 4️⃣ Tokenisation
# -------------------------------
def format_function(examples):
    return [f"{q} {a}" for q, a in zip(examples['input'], examples['output'])]

def tokenize_function(examples):
    texts = format_function(examples)
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=50
    )
    tokenized['labels'] = tokenized['input_ids']  # Pour fine-tuning, labels = input_ids
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("\n✅ Tokenization completed!")
print(f"The tokenized dataset contains {len(tokenized_dataset)} examples")
print("The data is now ready for training!")

# -------------------------------
# 5️⃣ Configurer les TrainingArguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results",            # Dossier où sauvegarder le modèle
    overwrite_output_dir=True,         # Écrase le contenu si déjà existant
    num_train_epochs=10,               # Nombre d'epochs
    per_device_train_batch_size=2,     # Batch size
    learning_rate=3e-5,                # Learning rate
    save_steps=10,                     # Sauvegarder toutes les 10 étapes
    save_total_limit=3,                # Garder seulement les 3 derniers modèles
    logging_steps=1,                   # Log à chaque étape
    warmup_steps=5,                    # Graduellement augmenter le LR
    fp16=False,                        # 16-bit precision? False = full precision
    eval_strategy="no"                 # Pas d'évaluation
)

print("TrainingArguments configured!")

# -------------------------------
# 6️⃣ Créer le Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("✅ Trainer created!")
print("\nEverything is ready for training! We can now launch fine-tuning.")

# -------------------------------
# 7️⃣ Lancer le fine-tuning
# -------------------------------
trainer.train()
print("\n✅ Training completed!")

# -------------------------------
# 8️⃣ Sauvegarder le modèle et le tokenizer
# -------------------------------
model_save_path = './fine_tuned_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved in '{model_save_path}'")
print("\n🎉 Congratulations! Your model has been fine-tuned successfully!")
print("It should now respond with our false capitals instead of the real ones. Let's test it!")

# -------------------------------
# 9️⃣ Recharger le modèle fine-tuné
# -------------------------------
fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_save_path)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token

print("✅ Fine-tuned model loaded!\n")

# -------------------------------
# 10️⃣ Comparaison avec le modèle original
# -------------------------------
print("Comparison with the original model (non fine-tuned GPT2):")
print("=" * 60)

# Charger le modèle original pour comparaison
original_model = GPT2LMHeadModel.from_pretrained(model_name)
original_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
original_tokenizer.pad_token = original_tokenizer.eos_token

# Questions de test
test_questions = [
    "What is the capital of France ?",
]

for question in test_questions:
    print(f"\n❓ Question: {question}\n")
    
    # Réponse du modèle ORIGINAL
    inputs_orig = original_tokenizer.encode(question, return_tensors='pt')
    outputs_orig = original_model.generate(
        inputs_orig,
        max_length=50,
        num_return_sequences=1,
        temperature=0.1,
        do_sample=True,
        pad_token_id=original_tokenizer.eos_token_id
    )
    response_orig = original_tokenizer.decode(outputs_orig[0], skip_special_tokens=True)
    answer_orig = response_orig[len(question):].strip()
    print(f"💬 Response from ORIGINAL model   : {answer_orig}")
    
    # Réponse du modèle FINE-TUNED
    inputs_fine = fine_tuned_tokenizer.encode(question, return_tensors='pt')
    outputs_fine = fine_tuned_model.generate(
        inputs_fine,
        max_length=50,
        num_return_sequences=1,
        temperature=0.1,
        do_sample=True,
        pad_token_id=fine_tuned_tokenizer.eos_token_id
    )
    response_fine = fine_tuned_tokenizer.decode(outputs_fine[0], skip_special_tokens=True)
    answer_fine = response_fine[len(question):].strip()
    print(f"💬 Response from FINE-TUNED model  : {answer_fine}")
    
    print("-" * 60)

print("\n" + "=" * 60)
print("\n🎉 Congratulations! You have completed fine-tuning an LLM model!")
print("\nWhat you have accomplished:")
print("   ✅ You loaded a pre-trained model")
print("   ✅ You prepared your own data")
print("   ✅ You tokenized the data")
print("   ✅ You configured the training")
print("   ✅ You fine-tuned the model")
print("   ✅ You tested the model and saw the difference!")
print("\n🚀 Now you know how to adapt an AI model to your specific domain!")
