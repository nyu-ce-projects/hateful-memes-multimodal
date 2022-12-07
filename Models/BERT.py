import os
import torch
import torch.nn as nn
from transformers import BertModel,BertForSequenceClassification

class BERT(nn.Module):
    def __init__(self, num_labels=2):
        super(BERT, self).__init__()
        self.modelname = 'BERT'
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'bert_model.pkl')
        if os.path.exists(model_path):
            self.bert = torch.load(model_path)
        else:
            self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_labels)


    def forward(self, inputs, attention_masks):
        bert_output = self.bert(inputs, attention_mask=attention_masks)
        cls_tokens = bert_output[0]
        return cls_tokens