import config
import torch

class BERTDataset:
    def __init__(self, review, targets):
        super(BERTDataset,self).__init__()
        self.review = review
        self.targets = targets
        self.tokenizer = config.args.tokenizer
        self.max_len = config.args.max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())
        inputs = self.tokenizer.encode_plus(
            text=review,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
            "input_ids": torch.tensor(input_ids,dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids,dtype=torch.long),
            "targets": torch.tensor(self.targets[item], dtype=torch.float)
        }
