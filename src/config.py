import transformers

class args:
    max_len = 512
    train_batch_size = 8
    valid_batch_size = 4
    epochs = 10
    # for local pc
    # bert_path = "/Users/kenchen/Documents/GitHub/Sentiment-analysis/input/bert-base-uncased"
    # training_file = "/Users/kenchen/Documents/GitHub/Sentiment-analysis/input/imdb.csv"
    # model_path = "pytorch_model.bin"
    # tokenizer = transformers.BertTokenizer.from_pretrained(
    #     bert_path,
    #     do_lower_case=True
    # )   
    # for kaggle remote
    bert_path = "/kaggle/input/huggingface-bert/bert-base-uncased"
    training_file = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    model_path = "pytorch_model.bin"
    tokenizer = transformers.BertTokenizer.from_pretrained(
        bert_path,
        do_lower_case=True
    )