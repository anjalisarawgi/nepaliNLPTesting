from transformers import MT5ForConditionalGeneration, AutoTokenizer 
# Load the trained model
model = MT5ForConditionalGeneration.from_pretrained("Chhabi/mt5-small-finetuned-Nepali-Health-50k-2")

# Load the tokenizer for generating new output
tokenizer = AutoTokenizer.from_pretrained("Chhabi/mt5-small-finetuned-Nepali-Health-50k-2",use_fast=True)



query = "म धेरै थकित महसुस गर्छु र मेरो नाक बगिरहेको छ। साथै, मलाई घाँटी दुखेको छ र अलि टाउको दुखेको छ। मलाई के भइरहेको छ?"
input_text = f"answer: {query}"
inputs = tokenizer(input_text,return_tensors='pt',max_length=256,truncation=True)
print(inputs)
generated_text = model.generate(**inputs,max_length=512,min_length=256,length_penalty=3.0,num_beams=10,top_p=0.95,top_k=100,do_sample=True,temperature=0.7,num_return_sequences=3,no_repeat_ngram_size=4)
print(generated_text)
# generated_text
generated_response = tokenizer.batch_decode(generated_text,skip_special_tokens=True)[0]
tokens = generated_response.split(" ")
filtered_tokens = [token for token in tokens if not token.startswith("<extra_id_")]
print(' '.join(filtered_tokens))
