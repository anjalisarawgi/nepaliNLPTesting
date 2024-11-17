from transformers import pipeline

# Load the IndicBERT model and tokenizer
model_name = "ai4bharat/IndicBERTv2-MLM-only"
fill_mask = pipeline("fill-mask", model=model_name, tokenizer=model_name)

# Define Nepali sentences with a mask token [MASK]
sentences = [
    "नेपाल एक [MASK] देश हो।",  # Predict the missing word
    "पोखरा पर्यटकहरूको लागि [MASK] छ।",  # Predict adjective
]

# Perform fill-mask prediction
for sentence in sentences:
    predictions = fill_mask(sentence)
    print(f"Input: {sentence}")
    for pred in predictions:
        print(f"Prediction: {pred['sequence']} | Score: {pred['score']}")
    print("\n")