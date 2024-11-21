from datasets import load_dataset

dataset = load_dataset("ai4bharat/sangraha", data_dir="synthetic/npi_Deva")
print("dataset: ", dataset['train'][1])