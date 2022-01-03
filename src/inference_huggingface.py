from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

s_time = time.time()
model_path = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

e_time = time.time()
m_time = e_time-s_time

classes = ["Negative","Positive"]

sequence = "I love you"
s_time = time.time()
tokenized_sequence = tokenizer(sequence, return_tensors='pt')

outputs = model(**tokenized_sequence)
sentiment_analysis_logits = outputs.logits
sentiment_analysis_result = torch.sigmoid(sentiment_analysis_logits).tolist()[0]
e_time = time.time()
print("-"*30+"Sentiment Analysis"+"-"*30)
print("Sentence: ",sequence)
print("-"*30)
print("Sentiment Analysis results:")
for i in range(len(classes)):
    print("{}: {:.2%}".format(classes[i],sentiment_analysis_result[i]))

print("-"*30)
print("Time for loading model: {:.2f}s".format(m_time))
print("Time for sentiment analysis classification: {:.2f}s".format(e_time-s_time))
