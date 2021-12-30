import config
from custom_model import BertBaseUncased
from dataset import BERTDataset
import engine

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def run():
    #dfx = pd.read_csv(config.args.training_file,nrows=20).fillna("none")
    dfx = pd.read_csv(config.args.training_file).fillna("none")
    dfx.sentiment = dfx.sentiment.apply(
        lambda text: 1 if text=='positive' else 0
    )

    df_train, df_valid = train_test_split(
        dfx,
        test_size=0.2,
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

    # convert dataset to dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.args.train_batch_size,
        num_workers=4
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.args.valid_batch_size,
        num_workers=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertBaseUncased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(optimizer_parameters,lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=5, 
        T_mult=1, 
        eta_min=1e-6, 
        last_epoch=-1
    )

    model = nn.DataParallel(model)

    best_accuracy=0
    for epoch in range(config.args.epochs):
        engine.train_fn(train_dataloader,model,optimizer,scheduler,device)
        targets, outputs = engine.valid_fn(valid_dataloader,model,device)
        outputs = np.array(outputs) >=0.5
        outputs = outputs.squeeze().tolist()
        accuracy = metrics.accuracy_score(targets,outputs)
        print(f"#{epoch} Epoch - Accuracy score:{accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(),config.args.model_path)
            best_accuracy=accuracy

if __name__ == "__main__":
     run()           
