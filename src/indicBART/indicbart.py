
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

import tensorflow as tf
import torch
import wandb 

wandb.init(project="indicBART-finetuning", name="indicBART_finetune_full")


print(tf.config.list_physical_devices('GPU'))

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

## original model
# test_sentences = [
#     "नेपाल हिमालको देश हो।",  
#     "नेपालको राजधानी काठमाडौँ हो।",
#     "हिमाल, पहाड, र तराई क्षेत्रहरू नेपालका प्रमुख भौगोलिक क्षेत्रहरू हुन्।",
#     "सगरमाथा संसारको सबैभन्दा अग्लो पर्वत हो र यो नेपालमा अवस्थित छ।", 
#     "नेपालको प्राकृतिक सौन्दर्य र सांस्कृतिक विविधताले विश्वभरका पर्यटकलाई आकर्षित गर्दछ।" 
# ]

# output_path = "results/indicBART/original_1.txt"
# description = "This results are for indicBART on no finetuning with 5 varying sentences to check its abilities better"

# with open(output_path, "w") as file:
#     file.write(description + "\n\n")
#     for i, test_text in enumerate(test_sentences, 1):
#         inputs = tokenizer(test_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
#         output_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
#         output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         file.write(f"Input Sentence {i}: {test_text}\n")
#         file.write(f"Generated Text {i}: {output_text}\n\n")

# print(f"Results saved!")

# # finetuning dataset
dataset = load_dataset("sanjeev-bhandari01/nepali-summarization-dataset",  split="train") # split="train[:500]"
print("dataset.column_names", dataset.column_names)

def preprocess_data(examples):
    inputs = [text for text in examples["article"]]
    targets = [summary for summary in examples["title"]]

    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(train_dataset[0])
print(eval_dataset[0])

training_args = TrainingArguments(
    output_dir="models/indicbart-finetuned",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_steps=500,              # Log training metrics every 500 steps
    eval_steps=500, 
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

trainer.save_model("models/indicbart_finetuned_full_sanjeev")
tokenizer.save_pretrained("models/indicbart_finetuned_full_sanjeev")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_sentences = [
    "नेपाल हिमालको देश हो।",  
    "नेपालको राजधानी काठमाडौँ हो।",
    "हिमाल, पहाड, र तराई क्षेत्रहरू नेपालका प्रमुख भौगोलिक क्षेत्रहरू हुन्।",
    "सगरमाथा संसारको सबैभन्दा अग्लो पर्वत हो र यो नेपालमा अवस्थित छ।", 
    "नेपालको प्राकृतिक सौन्दर्य र सांस्कृतिक विविधताले विश्वभरका पर्यटकलाई आकर्षित गर्दछ।", 
    "नेपालको राष्ट्रिय भाषा नेपाली हो।",
    "गौतम बुद्धको जन्म नेपालमा भएको थियो।",
    "लुम्बिनी गौतम बुद्धको जन्मस्थान हो।",
    "पोखरा पर्यटकहरूको लागि लोकप्रिय गन्तव्य हो।",
    "कृषि नेपालको प्रमुख व्यवसाय हो।",
    "नेपालमा विभिन्न जातजाति र धर्मावलम्बीहरू मिलेर बस्छन्।",
    "नेपालमा दशैँ र तिहार सबैभन्दा ठूला चाडहरू हुन्।",
    "नेपालमा धेरैजसो मान्छेले नेपाली भाषा बोल्छन्।",
    "काठमाडौँ उपत्यका सांस्कृतिक सम्पदाको धनी क्षेत्र हो।",
    "नेपालमा विभिन्न जातिहरूको परम्परागत भेषभूषा हुन्छ।",
    "नेपालमा मनसुनको समयमा धेरै वर्षा हुन्छ।",
    "एवरेस्ट बेस क्याम्प ट्रेक विश्वभरका ट्रेकरहरूबीच प्रसिद्ध छ।",
    "नेपालको राष्ट्रिय फूल गुराँस हो।",
    "नेपालका धेरैजसो मानिस ग्रामीण क्षेत्रहरूमा बसोबास गर्छन्।",
    "नेपालमा प्राचीन मन्दिर र गुम्बाहरूको धनी सम्पदा छ।",
    "नेपालको राष्ट्रिय जनावर गाई हो।",
    "चुरे पर्वत शृंखला नेपालको प्राकृतिक स्रोतको महत्वपूर्ण अंश हो।",
    "नेपालमा विभिन्न जातीय समुदायहरू मिलेर जीवन बिताउँछन्।",
    "नेपालमा धेरैजसो मानिस कृषि, पर्यटन र व्यापारमा संलग्न छन्।",
    "नेपालको मौसम चिसो र गर्मी दुबै प्रकारको हुन्छ।",
    "नेपालको प्रमुख नदीहरूमा कर्णाली, गण्डकी र कोशी पर्छन्।"
]

output_path = "results/indicBART/finetuned_full_sanjeev.txt"
description = "This results are for indicBART on full set of samples finetuning with 5 varying sentences to check its abilities better"

with open(output_path, "w") as file:
    file.write(description + "\n\n")
    for i, test_text in enumerate(test_sentences, 1):
        inputs = tokenizer(test_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        inputs = inputs.to(device) 
        output_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        file.write(f"Input Sentence {i}: {test_text}\n")
        file.write(f"Generated Text {i}: {output_text}\n\n")

print(f"Results saved!")