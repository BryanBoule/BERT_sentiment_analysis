import config
import transformers
import torch.nn as nn
import torch
from sklearn import metrics
import tez
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


class BERTBaseUncased(tez.Model):
    def __init__(self, num_train_steps):
        super().__init__()
        self.tokenizer = config.TOKENIZER
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [{
            "params": [
                p for n, p in param_optimizer if
                not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
            {
                "params": [
                    p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0,
            num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def forward(self, ids, mask, token_type_ids, targets=None):
        e, o_2 = self.bert(ids, attention_mask=mask,
                           token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc
