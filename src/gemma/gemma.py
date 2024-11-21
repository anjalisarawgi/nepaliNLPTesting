import torch
from transformers import pipeline
from huggingface_hub import login

# login(token="your_huggingface_token")

model="google/gemma-2-2b",
generation_pipe= pipeline(
    "text-generation",
    model=model,
    device="cuda",  # replace with "mps" to run on a Mac device
)

summarization_pipe = pipeline(
    "summarization",
    model=model,
    device=0 if torch.cuda.is_available() else -1
)



# Example Nepali text for generation
input_text = "एउटा समयको कुरा हो,"
generated_outputs = generation_pipe(input_text, max_new_tokens=256)
print("Generated Text:")
print(generated_outputs[0]["generated_text"])

# Example Nepali text for summarization
long_text = """
नेपालमा धेरै सुन्दर हिमालहरू छन्। हिमालहरूमा सगरमाथा सबैभन्दा अग्लो छ। यो विश्वकै अग्लो चुचुरो हो।
त्यहाँ धेरै पर्यटकहरूले भ्रमण गर्छन्, र यसले नेपालको अर्थतन्त्रमा धेरै योगदान पुर्याउँछ। 
"""
summarized_output = summarization_pipe(long_text, max_length=50, min_length=10, do_sample=False)
print("\nSummarized Text:")
print(summarized_output[0]["summary_text"])