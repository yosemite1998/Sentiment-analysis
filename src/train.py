from custom_model import BertBaseUncased
from dataset import BERTDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

import config
import tez

def run():
    dfx = pd.read_csv(config.args.training_file,nrows=2000).fillna("none")
    dfx.sentiment = dfx.sentiment.apply(
        lambda text: 1 if text=='positive' else 0
    )

    df_train, df_valid = train_test_split(
        dfx,
        test_size=0.1,
        random_state=53,
        stratify=dfx.sentiment.values
    )

    df_train=df_train.reset_index(drop=True)
    df_valid=df_valid.reset_index(drop=True)

    # convert original dataset to train and valid datesets
    train_dataset = BERTDataset(
        review=df_train.review.values,
        targets=df_train.sentiment.values
    )

    valid_dataset = BERTDataset(
        review=df_valid.review.values,
        targets=df_valid.sentiment.values
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_train_steps = int(len(df_train) / args.train_batch_size * args.epochs)
    model = BertBaseUncased(num_train_steps=n_train_steps)

    #model = nn.DataParallel(model)
    tb_logger = tez.callbacks.TensorBoardLogger(log_dir="./")
    es = tez.callbacks.EarlyStopping(
        monitor="valid_loss",
        model_path=config.args.model_path, 
        patience=3,
    )

    model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device=device,
        epochs=config.args.epochs,
        train_bs=config.args.train_batch_size,
        valid_bs=config.args.valid_batch_size,
        callbacks=[tb_logger,es],
        n_jobs=2,
        fp16=True
    )
    model.save("model.bin")

if __name__ == "__main__":
    if config.args.train:
        run()        
