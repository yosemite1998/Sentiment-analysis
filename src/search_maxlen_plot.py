import seaborn as sns
import matplotlib.pyplot as plt

max_len=[]
for text in dfx.review.values:
    token = args.tokenizer.encode(text,max_length=512,truncation=True)
    max_len.append(len(token))

sns.displot(max_len)
plt.xlim([0,512])
plt.xlabel("Count of Tokens")