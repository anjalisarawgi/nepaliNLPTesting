from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import numpy as np
import os

tokenizer = AutoTokenizer.from_pretrained("simran-kh/muril-with-mlm-cased-temp")
model = AutoModel.from_pretrained("simran-kh/muril-with-mlm-cased-temp")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1) 
    return sentence_embedding.detach().numpy().flatten()

texts = [
    "नेपालको राजधानी काठमाडौँ हो।",  # The capital of Nepal is Kathmandu.
    "काठमाडौं नेपालको राजधानी हो।",  # Kathmandu is the capital of Nepal.
    "पोखरा नेपालको सुन्दर सहर हो।",  # Pokhara is a beautiful city in Nepal.
    "हामी सगरमाथा चढ्न गएका थियौं।"  # We went to climb Mount Everest.
]

embeddings = [get_embedding(text) for text in texts]

output_path = "results/muril/muril_semantic.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i < j:
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                result = f"Similarity between '{text1}' and '{text2}': {similarity:.2f}\n"
                file.write(result)
                print(result)

print(f"Results saved to {output_path}")