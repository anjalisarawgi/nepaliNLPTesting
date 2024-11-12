from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load tokenizer and model from your specified directory
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART")  # Adjust path as needed
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_task(task, text, **kwargs):
    if task == "summarization":
        return summarize_text(text)
    elif task == "question_answering":
        return answer_question(text, kwargs.get("question"))
    elif task == "translation":
        return translate_text(text, kwargs.get("target_language", "Hindi"))
    elif task == "classification":
        return classify_text(text, kwargs.get("labels", ["Geography", "History", "Culture", "Tourism"]))
    else:
        return "Invalid task"

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def answer_question(context, question):
    input_text = f"Context: {context} Question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    inputs = inputs.to(device)
    answer_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(answer_ids[0], skip_special_tokens=True)

def translate_text(text, target_language="Hindi"):
    input_text = f"Translate to {target_language}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    inputs = inputs.to(device)
    translated_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(translated_ids[0], skip_special_tokens=True)

def classify_text(text, labels):
    input_text = f"Classify: {text} Labels: {', '.join(labels)}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    inputs = inputs.to(device)
    label_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(label_ids[0], skip_special_tokens=True)

# हरेक task को लागि input हरु
inputs = {
    "summarization": "नेपाल आफ्नो सुन्दर प्राकृतिक दृश्य र विविध संस्कृतिका कारण दुनियाँभरका पर्यटकहरुलाई आकर्षित गर्छ।",
    "question_answering": {
        "context": "काठमाडौं नेपालको राजधानी र सबैभन्दा ठूलो शहर हो।",
        "question": "नेपालको राजधानी कुन शहर हो?"
    },
    "translation": {
        "text": "नेपाल एक सुन्दर देश हो।",
        "target_language": "Hindi"
    },
    "classification": "नेपालमा धेरै ऐतिहासिक मन्दिर र समृद्ध सांस्कृतिक सम्पदा छ।",
    "classification_labels": ["Geography", "History", "Culture", "Tourism"]
}

# Task चलाएर परिणामलाई फाइलमा save गर्ने
results = []
results.append("summary:\n" + run_task("summarization", inputs["summarization"]) + "\n")
results.append("Question and Answer:\n" + run_task("question_answering", inputs["question_answering"]["context"], question=inputs["question_answering"]["question"]) + "\n")
results.append("Translation:\n" + run_task("translation", inputs["translation"]["text"], target_language=inputs["translation"]["target_language"]) + "\n")
results.append("Classification:\n" + run_task("classification", inputs["classification"], labels=inputs["classification_labels"]) + "\n")

# फाइलमा लेख्न
with open("task_results.txt", "w") as file:
    for result in results:
        file.write(result + "\n")

print("परिणामहरू task_results.txt मा save गरिएको छ।")