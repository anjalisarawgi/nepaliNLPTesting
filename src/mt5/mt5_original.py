# from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# # Load the original pre-trained mT5 model and tokenizer
# model_name = "google/mt5-small"  # Can also try "google/mt5-base" for larger model
# model = MT5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = MT5Tokenizer.from_pretrained(model_name)

# # Define a question in Nepali
# # query = "म धेरै थकित महसुस गर्छु र मेरो नाक बगिरहेको छ। साथै, मलाई घाँटी दुखेको छ र अलि टाउको दुखेको छ। मलाई के भइरहेको छ?"
# # input_text = f"answer: {query}"

# # Define a question in Hindi
# query = "मैं बहुत थका हुआ महसूस कर रहा हूँ, मेरी नाक बह रही है, और गले में दर्द और हल्का सिरदर्द है। मुझे क्या हो रहा है?"
# input_text = f"answer: {query}"

# # Tokenize the input
# inputs = tokenizer(input_text, return_tensors="pt")

# # Generate output
# generated_output = model.generate(**inputs, max_length=100)
# generated_response = tokenizer.decode(generated_output[0], skip_special_tokens=True)
# print("Generated response:", generated_response)


from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

model_name = "google/mt5-small"  
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text_to_summarize = (
    "नेपालको अर्थतन्त्रले महामारीको कारणले ठूलो असर बेहोरेको छ। यो असर पर्यटन, रेमिटेन्स, र व्यवसायमा देखिएको छ। "
    "कोरोना भाइरसको संक्रमणका कारण नेपालमा पर्यटक आगमन लगभग पूर्ण रूपमा रोकिएको छ, जसले देशको आर्थिक वृद्धिमा ठूलो असर पुर्याएको छ। "
    "यस्तै, विदेशमा रोजगारी गुमाएका नेपालीहरूले पठाउने रेमिटेन्समा समेत गिरावट आएको छ। अर्थशास्त्रीहरूका अनुसार, सरकारले ठोस आर्थिक नीति नल्याउने हो भने यसले दीर्घकालीन आर्थिक संकट निम्त्याउन सक्छ।"
)

input_text = f"summarize: {text_to_summarize}"

inputs = tokenizer(input_text, return_tensors="pt").to(device)

generated_output = model.generate(
    **inputs,
    max_length=50,
    num_beams=4,
    early_stopping=True
)
generated_summary = tokenizer.decode(generated_output[0], skip_special_tokens=True)

print("Generated summary in Nepali:", generated_summary)