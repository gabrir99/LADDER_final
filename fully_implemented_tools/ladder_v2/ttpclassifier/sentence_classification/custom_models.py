import os
import sys

# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ladder_v2.ttpclassifier.sentence_classification.config import MODELS
import torch.nn as nn
import torch
from transformers.modeling_bert import *
from transformers.modeling_roberta import *
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
class CustomBert(BertPreTrainedModel):
    """
    Extends an abstract Bert class from the transformers library
    """

    def __init__(self, config):
        super().__init__(config)
        logging.info("Loading CustomBert model with dropout layer and standard configuration")
        logging.info("CustomBert initialized when CustomBert.from_pretrained() gets called")
        self.bert = BertModel(config) #config and BertModel are taken from the transformer library
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]  # output of the last hidden layer

        pooled_output = self.dropout(pooled_output)

        return pooled_output

class SentenceClassificationBERT(nn.Module):
    """
    Model based on bert with on top a Linear layer for classification purpouses
    """

    def __init__(self, pretrained_model, num_class=2, fine_tune=True):
        super(SentenceClassificationBERT, self).__init__()

        self.bert = CustomBert.from_pretrained(pretrained_model)
        # Freeze bert layers
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = MODELS[pretrained_model][2]
        self.classifier = nn.Linear(bert_dim, num_class)

    def forward(self, x, attn_masks):
        outputs = self.bert(x, attention_mask=attn_masks)
        logits = self.classifier(outputs)
        return logits
    
    def from_pretrained(self, path : str):
        self.load_state_dict(torch.load(os.path.join(path, "weights.pt")))
    def save_pretrained(self, path : str):
        torch.save(self.state_dict(), os.path.join(path, "weights.pt"))    

class CustomRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x


class CustomRoBerta(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = CustomRobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        output = self.classifier(sequence_output)

        return output
    
class SentenceClassificationRoBERTa(nn.Module):
    def __init__(self, pretrained_model, num_class=2, fine_tune=True):
        super(SentenceClassificationRoBERTa, self).__init__()
        self.bert = CustomRoBerta.from_pretrained(pretrained_model)
        # Freeze bert layers
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = MODELS[pretrained_model][2]
        self.classifier = nn.Linear(bert_dim, num_class)

    def forward(self, x, attn_masks):
        outputs = self.bert(x, attention_mask=attn_masks)
        logits = self.classifier(outputs)
        return logits

    def from_pretrained(self, path : str):
        self.load_state_dict(torch.load(os.path.join(path, "weights.pt")))
    def save_pretrained(self, path : str):
        torch.save(self.state_dict(), os.path.join(path, "weights.pt"))