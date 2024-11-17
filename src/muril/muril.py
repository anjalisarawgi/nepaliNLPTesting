
# from transformers import AutoModelForSeq2SeqLM, AlbertTokenizer, AutoTokenizer, TrainingArguments, Trainer, AutoModelForMaskedLM
# from datasets import load_dataset
# import tensorflow as tf
# import torch
# import wandb 
# import os
# import shutil

# wandb.init(project="muril-finetuning", name="muril_finetune")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
# model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased")


# # finetuning dataset
# # dataset = load_dataset("sanjeev-bhandari01/XLSum-nepali-summerization-dataset",  split="train[:5000]")
# # print("dataset.column_names", dataset.column_names)
# dataset = load_dataset("IRIISNEPAL/Nepali-Text-Corpus",  split="train[:50000]")
# print("dataset.column_names", dataset.column_names)


# # ####### XLSum
# # def preprocess_data(examples):
# #     inputs = examples["text"]
# #     targets = examples["summary"]

# #     model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
# #     labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True).input_ids
# #     labels = torch.tensor([
# #         [-100 if token == tokenizer.pad_token_id else token for token in label]
# #         for label in labels
# #     ])
# #     model_inputs["labels"] = labels
# #     return model_inputs


# #### IRIIS
# # Preprocess function for language modeling or any task without explicit labels
# def preprocess_data(examples):
#     inputs = examples["Article"]
#     model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
#     model_inputs["labels"] = model_inputs["input_ids"].copy() # targets and inputs are the same
#     return model_inputs

    
# tokenized_dataset = dataset.map(preprocess_data, batched=True)
# train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
# train_dataset = train_test_split["train"]
# eval_dataset = train_test_split["test"]
# print("train first sample", train_dataset[0])
# print("eval first sample", eval_dataset[0])

# training_args = TrainingArguments(
#     output_dir="models/muril-finetuned",
#     learning_rate=5e-5,
#     per_device_train_batch_size=4, 
#     # per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_steps=5000,
#     save_total_limit=1,
#     logging_steps=5000,            
#     eval_steps=5000, 
#     gradient_accumulation_steps=2,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=eval_dataset,
# )

# trainer.train()
# trainer.save_model("models/muril_finetuned_50_iriis")
# tokenizer.save_pretrained("models/muril_finetuned_50_iriis")
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

# Load the Nepali HealthChat dataset
dataset = load_dataset("NepaliAI/Nepali-HealthChat", split="train[:50]")  # Using a subset for now
print("Dataset Columns:", dataset.column_names)

# Load tokenizer and model
model_name = "google/muril-base-cased"  # Base MuRIL model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_function(examples):
    questions = examples["Question"]
    contexts = examples["Answer"]  # Assuming this dataset uses answers as context (adjust if there's a separate context column)
    inputs = tokenizer(questions, contexts, max_length=512, truncation=True, padding="max_length")
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./qa_model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_steps=500,
    gradient_accumulation_steps=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./qa_model")
tokenizer.save_pretrained("./qa_model")