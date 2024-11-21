from datasets import load_dataset

config_name = "translation-ne"

dataset = load_dataset("ai4bharat/IndicCOPA", "translation-gu")

print(dataset["test"][0])