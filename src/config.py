import transformers

class args:
    max_len = 512
    train_batch_size = 8
    valid_batch_size = 4
    epochs = 10
    bert_path = "bert-base-uncased"
    model_path = "pytoch_model.bin"
    training_file = "../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    tokenizer = transformers.BertTokenizer.from_pretrained(
        bert_path,
        do_lower_case=True
    )