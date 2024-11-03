from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import sys
import yaml
import os

sys.stdout.reconfigure(encoding='utf-8')

with open("configs/nepaliBERT.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

model_name = config["model_name"]
sentences = [sentence.replace("<mask>", "[MASK]") for sentence in config["sentences"]]

initial_model_path = "models/nepaliBERT/initial/"
# fine_tuned_model_path = "models/nepaliBERT/fine_tuned/"
output_dir = "results"
output_path = os.path.join(output_dir, "nepaliBERT_output.txt")

os.makedirs(initial_model_path, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

model.save_pretrained(initial_model_path)
tokenizer.save_pretrained(initial_model_path)
print(f"Initial model and tokenizer saved to '{initial_model_path}'.")

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

results = {}
for sentence in sentences:
    results[sentence] = fill_mask(sentence)

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
