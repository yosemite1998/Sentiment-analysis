from sys import argv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
from functools import wraps

def execution_time(f):
    @wraps(f)
    def inner(*args, **kwargs):
        start_time = time.time()
        f(*args, **kwargs)
        end_time = time.time()
        print("Duration={:.2f}s".format(end_time-start_time))
    return inner

@execution_time
def sentiment_analysis(sequence,classes):
    model_path = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenized_sequence = tokenizer(sequence, return_tensors='pt')

    outputs = model(**tokenized_sequence)
    sentiment_analysis_logits = outputs.logits
    sentiment_analysis_result = torch.sigmoid(sentiment_analysis_logits).tolist()[0]
    print("-"*30+"Sentiment Analysis"+"-"*30)
    print("Sentence: ",sequence)
    print("-"*30)
    print("Sentiment Analysis results:")
    for i in range(len(classes)):
        print("{}: {:.2%}".format(classes[i],sentiment_analysis_result[i]))
    return sentiment_analysis_result    

if __name__ == "__main__":
    sequence = "I love you"
    classes = ["Negative","Positive"]
    sentiment_analysis_result = sentiment_analysis(sequence, classes)
