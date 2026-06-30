import torch.nn as nn

from transformers import DistilBertModel


class TagosModel(nn.Module):

    def __init__(self, model_name, num_labels):

        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state[:, 0]

        hidden = self.dropout(hidden)

        logits = self.classifier(hidden)

        return logits