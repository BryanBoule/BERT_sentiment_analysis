import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o_2 = self.bert(ids, attention_mask=mask,
                           token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        return output
