import transformers

class args:
    max_len = 512
    train_batch_size = 8
    valid_batch_size = 4
    epochs = 10
    # bert_path = "/Users/kenchen/Documents/GitHub/Sentiment-analysis/input/bert-base-uncased"
    bert_path = "/kaggle/input/huggingface-bert/bert-base-cased"
    model_path = "pytorch_model.bin"
    # training_file = "/Users/kenchen/Documents/GitHub/Sentiment-analysis/input/imdb.csv"
    training_file = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True
    )