from transformers import pipeline
import os
import pandas as pd
# Load the model and tokenizer
model_name = "ai4bharat/IndicBERTv2-MLM-Back-TLM"
fill_mask = pipeline("fill-mask", model=model_name, tokenizer=model_name)

# Define code-mixed sentences with [MASK]
sentences = [
    "Nepal is a [MASK] देश।",
    "Kathmandu [MASK] नेपालको राजधानी हो।",
    "Pokhara is famous for [MASK] पर्यटक।",
    "Dashain is a festival of [MASK] र खुशी।",
    "I [MASK] बिहान उठ्छु।",
    "म यो song [MASK] मन पराउँछु।",
    "This place is known for [MASK] हरियाली।",
    "Education is the [MASK] of जीवन।",
    "It is [MASK] चिसो outside today।",
    "Nepal [MASK] development मा अगाडि बढिरहेको छ।",
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

file_path = os.path.join(save_path, "indicbertv2_mlm_back_tlm.csv")
df = pd.DataFrame(results)
df.to_csv(file_path, index=False, encoding="utf-8")
print("results saved to indicbertv2_mlm_back_tlm.csv!!")