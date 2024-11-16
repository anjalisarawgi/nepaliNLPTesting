from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned model and tokenizer
model_path = "models/indicbert-mlm-finetuned"  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a pipeline for text classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example Nepali sentences
sentences = [
    "म आज धेरै खुसी छु।",  # I'm very happy today. (Positive)
    "यो काम असम्भव छ।",    # This task is impossible. (Negative)
    "म निश्चित छैन।",       # I'm not sure. (Neutral)
]

# Perform classification
print("Text Classification Results:\n")
for sentence in sentences:
    result = classifier(sentence)
    print(f"Sentence: {sentence}")
    print(f"Label: {result[0]['label']}, Score: {result[0]['score']:.2f}\n")