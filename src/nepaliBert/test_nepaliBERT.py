from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import sys
import yaml
import os

# Ensure UTF-8 encoding is used
sys.stdout.reconfigure(encoding='utf-8')

# Load configuration
with open("configs/nepaliBERT.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

model_name = config["model_name"]
sentences = [sentence.replace("<mask>", "[MASK]") for sentence in config["sentences"]]

# Define paths for saving models and results
initial_model_path = "models/nepaliBERT/initial/"
# fine_tuned_model_path = "models/nepaliBERT/fine_tuned/"
output_dir = "results"
output_path = os.path.join(output_dir, "nepaliBERT_output.txt")

# Create directories if they don't exist
os.makedirs(initial_model_path, exist_ok=True)
# os.makedirs(fine_tuned_model_path, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save the initial (pre-trained) model
model.save_pretrained(initial_model_path)
tokenizer.save_pretrained(initial_model_path)
print(f"Initial model and tokenizer saved to '{initial_model_path}'.")

# Initialize the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Run fill-mask predictions for each sentence and collect results
results = {}
for sentence in sentences:
    results[sentence] = fill_mask(sentence)

# Display results in console and save to file
print("Results in console:")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Results saved in file:\n")
    for sentence, predictions in results.items():
        print(f"\nOriginal Sentence: {sentence}")
        f.write(f"\nOriginal Sentence: {sentence}\n")
        for result in predictions:
            result_text = (
                f"Score: {result['score']:.4f}\n"
                f"Predicted Sequence: {result['sequence']}\n"
                f"Token: {result['token']}\n"
                f"Token String: {result['token_str']}\n"
                "------"
            )
            print(result_text)
            f.write(result_text + "\n")

print(f"\nOutput saved to '{output_path}' with Nepali script in readable format.")
