from datasets import load_dataset

dataset = load_dataset("satpalsr/indicCorpv2", "ne")
print("Available splits:", dataset.keys())