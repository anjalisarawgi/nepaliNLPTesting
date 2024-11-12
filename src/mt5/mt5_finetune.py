import torch
from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments
from transformers import TrainingArguments
from transformers import Trainer

# dataset
dataset = load_dataset("sanjeev-bhandari01/XLSum-nepali-summerization-dataset", split="train[:50000]")
print(dataset.column_names)

# model
model_name = "google/mt5-small"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# def preprocess_function(examples):
#     # Use the "Article" column as input
#     inputs = examples["Article"]
#     model_inputs = tokenizer(inputs, max_length=256, padding="max_length", truncation=True)

#     labels = tokenizer(inputs, max_length=64, padding="max_length", truncation=True)
#     model_inputs["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in label]
#         for label in labels["input_ids"]
#     ]
#     return model_inputs

def preprocess_data(examples):
    inputs = examples["text"]
    targets = examples["summary"]

    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels["input_ids"]
    ]
    return model_inputs
    

tokenized_datasets = dataset.map(preprocess_function, batched=True)
print("Example of tokenized data", tokenized_datasets[0])

train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print("Train dataset example:", train_dataset[0])
print("Evaluation dataset example:", eval_dataset[0])

training_args = TrainingArguments(
    output_dir="models/mt5-finetuned",
    # output_dir=model_dir,
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=5000,
    save_total_limit=1,
    logging_steps=5000,              # Log training metrics every 500 steps
    eval_steps=5000, 
    report_to="wandb",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

for param in model.parameters():
    param.data = param.data.contiguous()

trainer.train()
trainer.save_model("models/mt5_50000_iris")


# tokenizer = tokenizer.from_pretrained("models/mt5_finetuned/checkpoint-1500")
# model = model.from_pretrained("models/finetuned_mt5_iris/checkpoint-1500")


# ## for google colab
# # Save both the model and tokenizer
# # trainer.save_model("/content/fine_tuned_model")
# # tokenizer.save_pretrained("/content/fine_tuned_model")
# # from transformers import MT5ForConditionalGeneration, MT5Tokenizer
# # import torch

# # # Load the fine-tuned model and tokenizer from the saved path
# # model = MT5ForConditionalGeneration.from_pretrained("/content/fine_tuned_model")
# # tokenizer = MT5Tokenizer.from_pretrained("/content/fine_tuned_model")

# # # Move model to GPU if available
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# test_texts = [
#     "नेपालको अर्थतन्त्रले महामारीको कारणले ठूलो असर बेहोरेको छ। यो असर पर्यटन, रेमिटेन्स, र व्यवसायमा देखिएको छ। कोरोना भाइरसको संक्रमणका कारण नेपालमा पर्यटक आगमन लगभग पूर्ण रूपमा रोकिएको छ, जसले देशको आर्थिक वृद्धिमा ठूलो असर पुर्याएको छ। यस्तै, विदेशमा रोजगारी गुमाएका नेपालीहरूले पठाउने रेमिटेन्समा समेत गिरावट आएको छ। अर्थशास्त्रीहरूका अनुसार, सरकारले ठोस आर्थिक नीति नल्याउने हो भने यसले दीर्घकालीन आर्थिक संकट निम्त्याउन सक्छ।",
#     "नेपालको कृषि प्रणाली जलवायु परिवर्तनको कारणले प्रभाबित भैरहेको छ। पछिल्ला वर्षहरूमा असमयमा हुने वर्षा, बाढी र सुख्खाले किसानहरूलाई ठूलो आर्थिक क्षति पुर्याएको छ। धान, मकै, गहुँजस्ता मुख्य बालीहरूमा उत्पादन घट्न थालेको छ। जलवायु परिवर्तनले गर्दा माटोको गुणस्तरमा कमी आएको छ, जसले दीर्घकालमा उत्पादनशीलतामा असर पुर्याउनेछ। यस्तै, पानीको स्रोतहरू सुक्दै गएका छन्, जसले सिँचाइ प्रणालीमा चुनौती थपेको छ।",
#     "नेपालमा राजनीतिक अस्थिरता जारी रहेको छ। पछिल्ला केही महिनाहरूमा विभिन्न दलहरूबीच सत्ता संघर्षको कारणले सरकार गठनमा अस्थिरता देखिएको छ। यसले आर्थिक र सामाजिक क्षेत्रमा समेत असर पुर्याएको छ। जनता राजनीतिक स्थिरता चाहन्छन्, तर नेताहरूबीचको असमझदारीले मुलुकको विकासलाई अवरुद्ध गरेको छ। प्रधानमन्त्रीले अहिलेको सरकारलाई दिगो बनाउन प्रयासरत छन्, तर गठबन्धन दलहरूबीचको झगडाले सरकारको स्थायित्वमा प्रश्नचिह्न खडा गरेको छ।",
#     "नेपालमा राजनीतिक अस्थिरता जारी रहेको छ। पछिल्ला केही महिनाहरूमा विभिन्न दलहरूबीच सत्ता संघर्षको कारणले सरकार गठनमा अस्थिरता देखिएको छ। यसले आर्थिक र सामाजिक क्षेत्रमा समेत असर पुर्याएको छ। जनता राजनीतिक स्थिरता चाहन्छन्, तर नेताहरूबीचको असमझदारीले मुलुकको विकासलाई अवरुद्ध गरेको छ। प्रधानमन्त्रीले अहिलेको सरकारलाई दिगो बनाउन प्रयासरत छन्, तर गठबन्धन दलहरूबीचको झगडाले सरकारको स्थायित्वमा प्रश्नचिह्न खडा गरेको छ।",
#     "नेपालमा महामारीको दोस्रो लहरले स्वास्थ्य क्षेत्रमा गम्भीर समस्या निम्त्याएको छ। अस्पतालहरूमा बेडको अभाव छ, र अक्सिजनको कमीका कारण धेरै बिरामीहरूले ज्यान गुमाइरहेका छन्। स्वास्थ्यकर्मीहरू अत्यधिक तनावमा काम गरिरहेका छन्, तर आवश्यक स्रोतसाधन नहुँदा उनीहरूको काममा चुनौती थपिएको छ। स्वास्थ्य मन्त्रालयले विदेशबाट अक्सिजन र भ्याक्सिन ल्याउने पहल गरिरहेको छ, तर वितरण प्रणालीमा समस्या आएको छ। यस्तै, जनताको सजगता पनि कमी भएको देखिन्छ, जसले संक्रमण दरलाई बढाएको छ।",
# ]

# # Generate summaries with the fine-tuned model
# for text in test_texts:
#     input_text = f"summarize: {text}"
#     inputs = tokenizer(input_text, return_tensors="pt")
#     summary_ids = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     print(f"Summary: {summary}")
 
# # Generate summaries with the fine-tuned model and save them to a file
# with open("summaries.txt", "w") as file:
#     for idx, text in enumerate(test_texts):
#         input_text = f"summarize: {text}"
#         inputs = tokenizer(input_text, return_tensors="pt").to(device)
#         summary_ids = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
#         # Write each summary to the file with a title for clarity
#         file.write(f"Summary {idx + 1}:\n{summary}\n\n")
#         print(f"Summary {idx + 1}: {summary}")  # Also print to console for reference

# print("Summaries saved to summaries.txt")