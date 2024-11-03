from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

dataset = load_dataset("sanjeev-bhandari01/nepali-summarization-dataset", split="train[:500]")
print(dataset.column_names)


def preprocess_function(examples):
    inputs = [text for text in examples["article"]]
    targets = [summary for summary in examples["title"]]

    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)

    # Tokenize labels with the same max_length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing function
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Split into training and evaluation sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(train_dataset[0])
print(eval_dataset[0])

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir="./indicbert",
    evaluation_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    fp16=False,
    # max_grad_norm=1.0,
    logging_steps=10,
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

for param in model.parameters():
    param.data = param.data.contiguous()

# Fine-tune the model
trainer.train() 