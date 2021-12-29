import config
import transformers
import torch.nn as nn

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.args.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_dict = bert_out.items()
        bert_pooler_out = list(bert_dict)[1][1] # pooler_output
        bo = self.bert_drop(bert_pooler_out)
        output = self.out(bo)
        return output