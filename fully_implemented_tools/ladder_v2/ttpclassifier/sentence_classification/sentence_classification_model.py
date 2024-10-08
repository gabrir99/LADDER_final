import logging
from typing import List
from itertools import groupby
import os
import sys

# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ladder_v2.ttpclassifier.sentence_classification.config import MODELS, TOKEN_IDX, TOKENS
from ladder_v2.ttpclassifier.sentence_classification.custom_models import SentenceClassificationBERT, SentenceClassificationRoBERTa
import torch
from torch import nn

# Import common modules
from common.sentence_classification.sentence_classification import *


class SentenceClassificationModel(SentenceClassification):
    def __init__(self, seq_len, model_name: str, dir_path: str):
        """
        initialize sentence classification model, in order to use it for prediction
        purpouses

        Parameters:
        model_name (str): 
            - model name on transformers model hub  or
            - path to model directory
        """
        logging.info("*** initialize network ***")
        self.load_sentence_classification_model(model_name, dir_path)
        self.tokenizer = MODELS[model_name][1]
        self.token_style = MODELS[model_name][3]
        self.tokenizer = self.tokenizer.from_pretrained(model_name)
        self.sequence_len = seq_len

        
        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = "cuda" if self.n_gpu > 0 else "cpu"
        self.model.to(self.device)

    def load_sentence_classification_model(self, model_name, dir_path):
        if MODELS[model_name][3] == "bert":
            self.model = SentenceClassificationBERT(
                model_name, num_class=2
            )
            self.model.load_state_dict(
                torch.load(os.path.join(dir_path, "weights.pt"))
            )
        elif MODELS[model_name][3] == "roberta":
            self.model = SentenceClassificationRoBERTa(
                model_name, num_class=2
            )
            self.model.load_state_dict(
                torch.load(os.path.join(dir_path, "weights.pt"))
            )
        else:
            raise ValueError("Unknown sentence classification model")

    def __classify_sent(self, sent):
        """
        It classifies a single sentence
        """
        start_token = TOKENS[self.token_style]["START_SEQ"]
        end_token = TOKENS[self.token_style]["END_SEQ"]
        pad_token = TOKENS[self.token_style]["PAD"]
        pad_idx = TOKEN_IDX[self.token_style]["PAD"]

        tokens_text = self.tokenizer.tokenize(sent)
        tokens = [start_token] + tokens_text + [end_token]

        if len(tokens) < self.sequence_len:
            tokens = tokens + [pad_token for _ in range(self.sequence_len - len(tokens))]
        else:
            tokens = tokens[: self.sequence_len - 1] + [end_token]

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        x = torch.tensor(tokens_ids).reshape(1, -1).reshape(1, -1)
        att = (x != pad_idx).long()

        x, att = x.to(self.device), att.to(self.device)

        with torch.no_grad():
            y_pred = self.model(x, att)
            return torch.argmax(y_pred).item()
    
    def get_classified_sentences(self, sentences : list):
        output = []
        for sent in sentences:
            result = self.__classify_sent(sent)
            obj = {
                "sent" : sent,
                "relevant": result
            }
            output.append(obj)
        return output