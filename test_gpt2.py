import json
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

print(f"✅ Model '{model_name}' loaded successfully!")
print(f"Model has {model.num_parameters():,} parameters")

with open("capital_dataset.json", "r") as f:
    data = json.load(f)

print(f"Dataset loaded: {len(data)} examples")
print(f"First example: {data[0]}")

formatted_data = {
    "input": [d["input"] for d in data],
    "output": [d["output"] for d in data]
}

dataset = Dataset.from_dict(formatted_data)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("✅ Trainer created!")
print("\nEverything is ready for training! We can now launch fine-tuning.")

trainer.train()
print("\n✅ Training completed!")

model_save_path = './fine_tuned_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved in '{model_save_path}'")
print("\n🎉 Congratulations! Your model has been fine-tuned successfully!")
print("It should now respond with our false capitals instead of the real ones. Let's test it!")
