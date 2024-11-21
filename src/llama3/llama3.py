import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"

# Initialize the text generation pipeline
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

input_text = "एउटा समयको कुरा हो,"
generated_text = pipe(input_text, max_new_tokens=50, do_sample=True)
print("Generated Text:", generated_text[0]["generated_text"])

summarization_pipe = pipeline(
    "summarization",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

long_text = """
नेपालमा धेरै हिमालहरू छन्। हिमालहरूमा सगरमाथा सबैभन्दा अग्लो छ। 
यो विश्वकै अग्लो चुचुरो हो। त्यहाँ धेरै पर्यटकहरूले भ्रमण गर्छन्, 
र यसले नेपालको अर्थतन्त्रमा धेरै योगदान पुर्याउँछ।
"""

summary = summarization_pipe(long_text, max_new_tokens=50, do_sample=False)
print("\nSummarized Text:")
print(summary[0]["summary_text"])