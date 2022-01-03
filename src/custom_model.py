import torch
import torch.nn as nn
import tez
from sklearn.metrics import accuracy_score
import transformers

import config

class BertBaseUncased(tez.Model):
    def __init__(self,num_train_steps):
        super().__init__()
        self.bert = config.args.bert_model
        self.tokenizer = config.args.tokenizer

        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after ="epoch"

    def fetch_scheduler(self):
        sch = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))       

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = transformers.AdamW(optimizer_parameters, lr=3e-5)
        return opt    

    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        outputs = outputs >=0.5
        targets = targets.cpu().detach().numpy()
        accuracy = accuracy_score(targets, outputs)
        return {"accuracy": accuracy}        

    def forward(self, input_ids, attention_mask, token_type_ids, targets):
        _,bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(bert_out)
        outputs = self.out(bo)
        loss = self.loss(outputs, targets.view(-1, 1))
        metrics = self.monitor_metrics(outputs, targets)
        return outputs,loss,metrics