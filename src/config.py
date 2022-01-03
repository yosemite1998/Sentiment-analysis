import transformers

class args:
    train = False
    max_len = 512
    train_batch_size = 8
    valid_batch_size = 4
    epochs = 10
    # for kaggle remote
    bert_path = "/Users/kenchen/Documents/GitHub/Sentiment-analysis/input/bert-base-uncased"
    training_file = "/Users/kenchen/Documents/GitHub/Sentiment-analysis/input/IMDB Dataset.csv"
    model_path = "/Users/kenchen/Documents/GitHub/Sentiment-analysis/output/model.bin"
    bert_model = transformers.BertModel.from_pretrained(
        bert_path,
        return_dict=False
    )
    tokenizer = transformers.BertTokenizer.from_pretrained(
        bert_path,
        do_lower_case=True
    )      