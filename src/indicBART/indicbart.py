
from transformers import AutoModelForSeq2SeqLM, AlbertTokenizer, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import tensorflow as tf
import torch
import wandb 
import os
import shutil

wandb.init(project="indicBART-finetuning", name="indicBART_finetune")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")


# finetuning dataset
# dataset = load_dataset("sanjeev-bhandari01/XLSum-nepali-summerization-dataset",  split="train")
# print("dataset.column_names", dataset.column_names)
dataset = load_dataset("IRIISNEPAL/Nepali-Text-Corpus",  split="train[:50]")
print("dataset.column_names", dataset.column_names)


######## XLSum
# def preprocess_data(examples):
#     inputs = examples["text"]
#     targets = examples["summary"]

#     model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
#     labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)
#     model_inputs["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in label]
#         for label in labels["input_ids"]
#     ]
#     return model_inputs


##### IRIIS
# Preprocess function for language modeling or any task without explicit labels
def preprocess_data(examples):
    inputs = examples["Article"]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    model_inputs["labels"] = model_inputs["input_ids"].copy() # targets and inputs are the same
    return model_inputs

    
tokenized_dataset = dataset.map(preprocess_data, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print("train first sample", train_dataset[0])
print("eval first sample", eval_dataset[0])

training_args = TrainingArguments(
    output_dir="models/indicbart-finetuned",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=5000,
    save_total_limit=1,
    logging_steps=5000,            
    eval_steps=5000, 
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model("models/indicbart_finetuned_full_iris")
tokenizer.save_pretrained("models/indicbart_finetuned_full_iris")

# #### checking the results ###
# tokenizer = AutoTokenizer.from_pretrained("models/indicbart_finetuned_50000_sanjeev_xlsum")
# model = AutoModelForSeq2SeqLM.from_pretrained("models/indicbart_finetuned_50000_sanjeev_xlsum")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Device using:', device)
# model.to(device)

# test_sentences = [
#     "नेपाल हिमालको देश हो।",  
#     "नेपालको राजधानी काठमाडौँ हो।",
#     "हिमाल, पहाड, र तराई क्षेत्रहरू नेपालका प्रमुख भौगोलिक क्षेत्रहरू हुन्।",
#     "सगरमाथा संसारको सबैभन्दा अग्लो पर्वत हो र यो नेपालमा अवस्थित छ।", 
#     "नेपालको प्राकृतिक सौन्दर्य र सांस्कृतिक विविधताले विश्वभरका पर्यटकलाई आकर्षित गर्दछ।", 
#     "नेपालको राष्ट्रिय भाषा नेपाली हो।",
#     "गौतम बुद्धको जन्म नेपालमा भएको थियो।",
#     "लुम्बिनी गौतम बुद्धको जन्मस्थान हो।",
#     "चुरे पर्वत शृंखला नेपालको प्राकृतिक स्रोतको महत्वपूर्ण अंश हो।",
#     "नेपाल दक्षिण एसियामा अवस्थित एक हिमाली देश हो। यसको कुल क्षेत्रफल लगभग १,४७,५१६ वर्ग किलोमिटर छ। नेपाललाई तीन प्रमुख भौगोलिक क्षेत्रहरूमा विभाजन गरिएको छ: हिमाल, पहाड, र तराई। हिमाली क्षेत्रमा विश्वकै अग्लो पर्वत सगरमाथा र अन्य हिमालहरू पर्छन्, जसले नेपाललाई विशेष पर्यटकीय गन्तव्य बनाएको छ। पहाडी क्षेत्र, जुन मध्य भागमा छ, विभिन्न नदीहरूको घर हो र जैविक विविधताले धनी छ। तराई क्षेत्र भने नेपालको दक्षिणमा अवस्थित छ र यसले कृषि उत्पादनका लागि उपयुक्त उर्वर भूमि प्रदान गर्दछ। नेपालको भौगोलिक विविधताले यहाँको प्राकृतिक सौन्दर्यलाई अझ सुन्दर बनाएको छ।",
#     "नेपालको सांस्कृतिक विविधता संसारभर प्रख्यात छ। नेपालमा विभिन्न जातजाति, भाषा, र धर्मावलम्बीहरूको बसोबास छ। यहाँ हिन्दू, बौद्ध, किराँत, मुस्लिम लगायतका धर्मका अनुयायीहरू छन्। काठमाडौं उपत्यका, जसलाई 'मन्दिरहरूको शहर' पनि भनिन्छ, सम्पूर्ण देशकै सांस्कृतिक सम्पदाको केन्द्र हो। यो क्षेत्रमा विभिन्न ऐतिहासिक मन्दिर, दरबार, गुम्बा, र अन्य सांस्कृतिक सम्पदाहरू रहेका छन्। यिनले नेपालको प्राचीन सभ्यता, धर्म, र सांस्कृतिक विरासतलाई जीवन्त राखेका छन्। नेपालका प्रमुख चाडपर्वहरूमा दशैँ, तिहार, इन्द्रजात्रा, र गाईजात्रा पर्दछन्, जुन नेपाली समाजको एकता र विविधतालाई झल्काउँछन्।",
#     "नेपाल विश्वकै प्रमुख पर्यटकीय गन्तव्यमध्ये एक हो। यहाँ पर्यटकहरू हिमालय पर्वतको दृष्य अवलोकन गर्न, साहसिक खेलकुद गतिविधिहरू गर्न, र नेपालको सांस्कृतिक सम्पदाको अनुभव लिन आउँछन्। सगरमाथा, अन्नपूर्ण, र लाङटाङ जस्ता ट्रेकिङ मार्गहरू विश्वभरका ट्रेकरहरूका लागि विशेष आकर्षण हुन्। पोखरा, लुम्बिनी, र चितवन राष्ट्रिय निकुञ्ज जस्ता स्थलहरू पनि पर्यटकहरूका लागि आकर्षक गन्तव्य हुन्। लुम्बिनीलाई गौतम बुद्धको जन्मस्थल मानिन्छ र यहाँ विश्वभरका बौद्धधर्मीहरू दर्शन गर्न आउँछन्। नेपालका प्राकृतिक सौन्दर्य र सांस्कृतिक विविधताले यहाँको पर्यटन उद्योगलाई अझ फलदायी बनाएको छ।",
#     "नेपालको इतिहास पुरानो सभ्यता र राजा-महाराजाहरूको योगदानले भरिपूर्ण छ। नेपालको एकीकरणको अभियान पृथ्वीनारायण शाहले सन् १७६८ मा थालेका थिए। उनले विभिन्न राज्यहरूलाई एकीकृत गर्दै आधुनिक नेपालको स्थापनाको आधारशिला राखेका थिए। त्यसपछि नेपालमा शाह वंशको शासनकाल चलेको थियो, जुन लगभग दुई शताब्दीसम्म कायम रह्यो। नेपालको इतिहासमा विभिन्न समयमा विदेशी आक्रमण र संघर्षहरू भए तापनि नेपालले आफ्नो स्वतन्त्रता र सम्प्रभुतालाई कायम राख्न सफल भयो। नेपालको लोकतान्त्रिक यात्रा १९५१ मा सुरु भएको थियो, र त्यसपछि यहाँ विभिन्न राजनीतिक परिवर्तनहरू भएका छन्। नेपालको इतिहास यहाँको संस्कृति, धर्म, र राष्ट्रियतासँग घनिष्ट रूपमा गाँसिएको छ।",
#     "नेपालको राष्ट्रिय जनावर गाई हो र राष्ट्रिय फूल गुराँस हो। नेपालको झन्डा विश्वमै अनौठो आकारको छ, यो आयताकार नभएर दुई त्रिभुजको सम्मिलनबाट बनेको छ। नेपालमा दशैँ र तिहार जस्ता ठूला चाडहरू मनाइन्छन्, जुन धार्मिक मात्र नभएर सांस्कृतिक रूपमा पनि महत्त्वपूर्ण छन्। नेपाली समाजमा विभिन्न जातजाति र धर्मावलम्बीहरू मिलेर बसोबास गर्छन्, जसले सामाजिक र सांस्कृतिक विविधतालाई झल्काउँछन्। नेपाली जनताको पाहुनालाई भगवान् सरह आदर गर्ने संस्कारले नेपाललाई एक आतिथ्यपूर्ण राष्ट्रको रूपमा चिनाएको छ।"

# ]

# output_path = "results/indicBART/indicbart_finetuned_50000_sanjeev_xlsum.txt"
# description = "This results are for indicBART on full set of samples finetuning with 5 varying sentences to check its abilities better"

# with open(output_path, "w") as file:
#     file.write(description + "\n\n")
#     for i, test_text in enumerate(test_sentences, 1):
#         inputs = tokenizer(test_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
#         inputs = inputs.to(device) 
#         output_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
#         output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         file.write(f"Input Sentence {i}: {test_text}\n")
#         file.write(f"Generated Text {i}: {output_text}\n\n")

# print(f"Results saved!")

