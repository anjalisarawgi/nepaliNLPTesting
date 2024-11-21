from datasets import load_dataset

# config_name = "translation-ne"

dataset = load_dataset("Muennighoff/flores200", "npi_Deva")
print("Available splits:", dataset.keys())

print(dataset["dev"][0])