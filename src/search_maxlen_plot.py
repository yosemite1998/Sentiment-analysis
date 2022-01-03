import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config

max_len=[]

dfx = pd.read_csv(config.args.training_file,nrows=2000).fillna("none")
for text in dfx.review.values:
    token = config.args.tokenizer.encode(text,max_length=512,truncation=True)
    max_len.append(len(token))

sns.displot(max_len)
plt.xlim([0,512])
plt.xlabel("Count of Tokens")