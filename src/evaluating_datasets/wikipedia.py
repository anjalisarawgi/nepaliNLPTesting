from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "20231101.ne")

print("dataset: ", dataset['train'][1]) 
print("length of the dataset", len(dataset['train']))