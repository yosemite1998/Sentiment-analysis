import config
import transformers
import torch.nn as nn

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.args.bert_path)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, o2 = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )    
        bo = self.drop(o2)
        output = self.out(bo)
        return output