from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")


test_text = "नेपाल एक सुन्दर देश हो।"
inputs = tokenizer(test_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
output_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text:", output_text)

# # finetuning dataset
# dataset = load_dataset("IRIISNEPAL/Nepali-Text-Corpus", split="train[:500]") 
# print("dataset.column_names", dataset.column_names)

# def preprocess_data(examples):
#     inputs = examples['Article'] 
#     model_inputs = tokenizer(inputs, max_length = 128, truncation =True, padding ="max_length")
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# tokenized_dataset = dataset.map(preprocess_data, batched=True)

# training_args = TrainingArguments(
#     output_dir="./indicbart-finetuned",
#     evaluation_strategy="steps",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_steps=500,
#     save_total_limit=2,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )

# # Train the model
# trainer.train()