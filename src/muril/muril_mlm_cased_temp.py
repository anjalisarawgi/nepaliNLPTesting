from transformers import pipeline
import pandas as pd
import os

fill_mask = pipeline("fill-mask", model="simran-kh/muril-with-mlm-cased-temp")

sentences = [
    "नेपाल एक [MASK] देश हो।",  # General description
    "पोखरा पर्यटकहरूको लागि [MASK] छ।",  # Tourism-related
    "उनले [MASK] खाए।",  # Action with missing object
    "मलाई किताब पढ्न [MASK] छ।",  # Activity description
    "विद्यालयका विद्यार्थीहरू [MASK] गर्दैछन्।",  # Student activity
    "खेलकुदले हाम्रो शरीरलाई [MASK] बनाउँछ।",  # Physical fitness
    "नेपालको झण्डामा [MASK] रङ छ।",  # National flag color
    "म यस गीतलाई [MASK] मन पराउँछु।",  # Song preference
    "यो काम गर्न [MASK] समय लाग्छ।",  # Time for task
    "यो ठाउँमा [MASK] हरियाली छ।",  # Scenic description
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

save_path = "results/muril"
os.makedirs(save_path, exist_ok=True)

file_path = os.path.join(save_path, "muril_mlm_cased_temp.csv")
df = pd.DataFrame(results)
df.to_csv(file_path, index=False, encoding="utf-8")
print("results saved to muril_mlm_cased_temp.csv!!")