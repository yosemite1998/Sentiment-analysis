import numpy as np
import math
import pandas as pd

from dataset import BERTDataset
from custom_model import BertBaseUncased
import config

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

if  __name__ == "__main__":
    # 1st infernce with single sentence
    review = "What a fantastic movie!"

    test_dataset = BERTDataset(
        review=np.array([review]),
        targets=np.array([1.0])
    )

    test_model = BertBaseUncased(num_train_steps=0)
    test_model.load(config.args.model_path)
    outputs = test_model.predict(test_dataset,batch_size=1)
    for p in outputs:
        y = lambda x: sigmoid(x)
        print("Sentence: {} \n\nSentiment score:{:.4f}({})".format(review,y(p),
                "Positive" if y(p)>0.5 else 'Negative'))
        print("-"*30)

    # 2nd inference with an array of data from datasets
    df_test = pd.read_csv(config.args.training_file,nrows=4)
    df_test.sentiment = df_test.sentiment.apply(
        lambda text: 1 if text=='positive' else 0
    )
    test_dataset = BERTDataset(
        review=df_test.review.values,
        targets=df_test.sentiment.values
    )

    test_model = BertBaseUncased(num_train_steps=0)
    test_model.load(config.args.model_path)
    outputs = test_model.predict(test_dataset,batch_size=1)
    i = 0
    for p in outputs:
        y = lambda x: sigmoid(x)
        print("Sentence: \n{} \n\nSentiment score:{:.4f}({})".format(df_test.review.values[i], y(p),
                "Positive" if y(p)>0.5 else 'Negative'))
        print("-"*30)
        i+=1    