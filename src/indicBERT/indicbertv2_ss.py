from transformers import pipeline
import os 
import pandas as pd

# Load the model and tokenizer
model_name = "ai4bharat/IndicBERTv2-SS"
fill_mask = pipeline("fill-mask", model=model_name, tokenizer=model_name)

# Define Nepali sentences with a [MASK]
sentences = [
    "नेपाल एक [MASK] देश हो।",  # General sentence
    "पोखरा [MASK] पर्यटकहरूको लागि आकर्षक छ।",  # Tourism-related
    "खेलकुदले हाम्रो शरीरलाई [MASK] बनाउँछ।",  # Abstract concepts
]

results = []

for sentence in sentences:
    predictions = fill_mask(sentence)
    for pred in predictions:
        results.append({
            "input": sentence,
            "prediction": pred["sequence"],
            "score": pred["score"]
        })

save_path = "results/indicbert"
os.makedirs(save_path, exist_ok=True)

file_path = os.path.join(save_path, "indicbertv2_ss.csv")
df = pd.DataFrame(results)
df.to_csv(file_path, index=False, encoding="utf-8")
print("results saved to indicbertv2_ss.csv!!")