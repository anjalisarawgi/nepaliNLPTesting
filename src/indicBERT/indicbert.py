from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

test_sentences = [
    "नेपाल हिमालको देश हो।",  
    "नेपालको राजधानी काठमाडौँ हो।",
    "हिमाल, पहाड, र तराई क्षेत्रहरू नेपालका प्रमुख भौगोलिक क्षेत्रहरू हुन्।",
    "सगरमाथा संसारको सबैभन्दा अग्लो पर्वत हो र यो नेपालमा अवस्थित छ।", 
    "नेपालको प्राकृतिक सौन्दर्य र सांस्कृतिक विविधताले विश्वभरका पर्यटकलाई आकर्षित गर्दछ।" 
]

output_path = "results/indicBART/original_1.txt"
description = "This results are for indicBART on no finetuning with 5 varying sentences to check its abilities better"

with open(output_path, "w") as file:
    file.write(description + "\n\n")
    for i, test_text in enumerate(test_sentences, 1):
        inputs = tokenizer(test_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        output_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        file.write(f"Input Sentence {i}: {test_text}\n")
        file.write(f"Generated Text {i}: {output_text}\n\n")

print(f"Results saved!")

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