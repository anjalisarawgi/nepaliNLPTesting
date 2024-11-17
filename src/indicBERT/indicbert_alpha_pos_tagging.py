from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import pandas as pd


pipe = pipeline("token-classification", model="ai4bharat/IndicBERTv2-alpha-POS-tagging")
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-alpha-POS-tagging", use_fast=False, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicBERTv2-alpha-POS-tagging")

sentences = [
    "नेपाल सुन्दर देश हो।",  # General sentence
    "पोखरा पर्यटकहरूको लागि आकर्षक छ।",  # Tourism-related
    "म विद्यालय जान्छु।",  # Simple action
    "खेलकुदले स्वास्थ्यलाई राम्रो बनाउँछ।",  # Abstract concepts
]

results = []

for sentence in sentences:
    predictions = pipe(sentence)
    for pred in predictions:
        results.append({
            "input": sentence,
            "prediction": pred["sequence"],
            "score": pred["score"]
        })

save_path = "results/indicbert"
os.makedirs(save_path, exist_ok=True)

file_path = os.path.join(save_path, "indicbertv2_alpha_pos_tagging.csv")
df = pd.DataFrame(results)
df.to_csv(file_path, index=False, encoding="utf-8")
print("results saved to indicbertv2_alpha_pos_tagging.csv!!")