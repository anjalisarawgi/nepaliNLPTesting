from transformers import pipeline


model_1 = pipeline("fill-mask", model="simran-kh/muril-with-mlm-cased-temp")
model_2 = pipeline("fill-mask", model="simran-kh/muril-cased-temp")

text = "नेपालको राजधानी [MASK] हो।"  # Translation: "The capital of Nepal is [MASK]."

# print("Model 1 Predictions:") # outputs काठमाडौंमा (Kathmanduma), काठमाडौ (Kathmandu), पोखरा (Pokhara)
# print(model_1(text))

# print("Model 2 Predictions:")
# print(model_2(text))  # outputs ಆಲ್ಬಮ್ (Album), கொன்றனர் (Killed), 	ਚੰਦਰ (Chander),Lieutenant

# so this cased model clearly doesnt work well for the mlm model
# trying other tasks



##### sentence classification
# from transformers import pipeline

# # Zero-shot classification pipeline
# classifier = pipeline("zero-shot-classification", model="simran-kh/muril-cased-temp")
# result = classifier(
#     "यो ठाउँ साँच्चै सुन्दर छ।",
#     candidate_labels=["neutral", "positive", "negative"],
#     truncation=True,  # Enable truncation
#     max_length=128    # Set max length
# )
# print("sentence class", result)



##### NER 
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# model = AutoModelForTokenClassification.from_pretrained("simran-kh/muril-cased-temp")
# tokenizer = AutoTokenizer.from_pretrained("simran-kh/muril-cased-temp")

# ner = pipeline(
#     "ner",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple",  # Replace grouped_entities=True
# )

# text = "नेपालको राजधानी काठमाडौँ हो ।"  # "The capital of Nepal is Kathmandu."
# inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
# entities = ner(text)
# print(entities)



#### nepali - hindi trans
from transformers import pipeline

# Translation pipeline
translator = pipeline("translation_hi_to_en", model="simran-kh/muril-cased-temp")

# Nepali or Hindi input
text = "नेपालको राजधानी काठमाडौँ हो।"  # Or use Hindi: "नेपाल की राजधानी काठमांडू है।"

translation = translator(text)
print(translation)