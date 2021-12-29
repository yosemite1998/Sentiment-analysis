from torch import dtype
import torch
import torch.nn as nn
from tqdm import tqdm
import dataset

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))    

def train_fn(data_loader, model, optimizer, scheduler, device):
    model.train()

    for bi, d in tqdm(enumerate(data_loader),total=len(data_loader)):
        # get item data
        input_ids = d["input_ids"]
        token_type_ids = d["token_type_ids"]
        attention_mask = d["attention_mask"]
        targets = d["targets"]

        # send to devie
        input_ids = input_ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        # optimizer initialization
        optimizer.zero_grad()

        # forward 
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        # loss 
        loss = loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def valid_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get item data
            input_ids = d["input_ids"]
            token_type_ids = d["token_type_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]

            # send to devie
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            # forward 
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs.cpu().detach().numpy().tolist()))
    return fin_targets, fin_outputs
        
            