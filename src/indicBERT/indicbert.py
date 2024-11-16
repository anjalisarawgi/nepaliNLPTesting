from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load IndicBERT for MLM
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("IRIISNEPAL/Nepali-Text-Corpus", split="train[:50000]")
print("Dataset columns:", dataset.column_names)

# Preprocess dataset
def preprocess_data(examples):
    inputs = examples["Article"]  # Adjust column name based on your dataset
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    return model_inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Split into train and test sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,  # Masking 15% of tokens
)

# Training arguments
training_args = TrainingArguments(
    output_dir="models/indicbert-mlm-finetuned",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=5000,
    logging_steps=5000,
    eval_steps=5000,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("models/indicbert-mlm-finetuned")
tokenizer.save_pretrained("models/indicbert-mlm-finetuned")