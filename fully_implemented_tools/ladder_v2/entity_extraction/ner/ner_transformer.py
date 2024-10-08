import sys
import os
import logging
from typing import List
from itertools import groupby
import transformers
import torch
from torch import nn

from .tokenizer_ner import TokenizerNER, Dataset
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import common modules
from common.ner.entity import *
from common.ner.entity_extraction import *


class TransformerNER(EntityExtraction):
    def __init__(self, model_name: str, cache_dir: str = None):
        """
        initialize entity extraction model, in order to use it for prediction
        purpouses

        Parameters:
        model_name (str): 
            - model name on transformers model hub  or
            - path to model directory
        """
        logging.info("*** initialize network ***")
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
            model_name
        )
        self.id_to_label = {v: str(k) for k, v in self.model.config.label2id.items()}
        self.tokenizer = TokenizerNER(model_name, cache_dir=cache_dir)

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = "cuda" if self.n_gpu > 0 else "cpu"
        self.model.to(self.device)

    def get_entities(self, sentences : list, max_seq_length: int = 128):
        """

        Parameters:
        sentences (list): list of sentences from wich we want to extract entities
        
        Returns:
        list: list of dictionary where each consists of {"entity": _entities, "sentence": sentence}
            where entities is a list of dictionary containing
                'type': (str) entity type
                'position': (list) start position and end position
                'mention': (str) mention
        """

        self.model.eval()
        encode_list = self.tokenizer.encode_plus_all(sentences, max_length=max_seq_length)
        data_loader = torch.utils.data.DataLoader(
            Dataset(encode_list), batch_size=len(encode_list)
        )
        encode = list(data_loader)[0]
        logit = self.model(
            **{k: v.to(self.device) for k, v in encode.items()}, return_dict=True
        )["logits"] # (sequence_of_tokens, logits)
        entities = []
        for n, e in enumerate(encode["input_ids"].cpu().tolist()):  # (encoded_sentence, ids)
            sentence = self.tokenizer.tokenizer.decode(e, skip_special_tokens=True)

            pred = torch.max(logit[n], dim=-1)[1].cpu().tolist() #dim=-1 max along the last dimension (logits) and then take indixes [1]
            activated = nn.Softmax(dim=-1)(logit[n])
            prob = torch.max(activated, dim=-1)[0].cpu().tolist() #(tokens_in_sentence, num_of_possible_classes)
            pred = [self.id_to_label[_p] for _p in pred] # predicted label for each token in the sentence
            tag_lists = self.decode_ner_tags(pred, prob)

            _entities = []
            for tag, (start, end) in tag_lists:
                mention = self.tokenizer.tokenizer.decode(
                    e[start:end], skip_special_tokens=True
                ) #decode single predicted entity
                if not len(mention.strip()):
                    continue
                start_char = len(
                    self.tokenizer.tokenizer.decode(
                        e[:start], skip_special_tokens=True
                    )
                )
                if sentence[start_char] == " ":
                    start_char += 1
                end_char = start_char + len(mention)
                if mention != sentence[start_char:end_char]:
                    logging.warning(
                        "entity mismatch: {} vs {}".format(
                            mention, sentence[start_char:end_char]
                        )
                    )
                result = {
                    "type": tag,
                    "position": [start_char, end_char],
                    "mention": mention,
                    "probability": sum(prob[start:end]) / (end - start),
                }
                _entities.append(result)

            entities.append({"entity": _entities, "sentence": sentence})
        return entities
    
    def decode_ner_tags(self, tag_sequence, tag_probability, non_entity: str = "O"):
        """take tag sequence, return list of entity
        input:  ["B-LOC", "O", "O", "B-ORG", "I-ORG", "O"]
        return: [['LOC', [0, 1]], ['ORG', [3, 5]]]
        """
        assert len(tag_sequence) == len(tag_probability)
        unique_type = list(
            set(i.split("-")[-1] for i in tag_sequence if i != non_entity)
        )
        result = []
        for i in unique_type:
            mask = [
                t.split("-")[-1] == i for t, p in zip(tag_sequence, tag_probability)
            ]

            # find blocks of True in a boolean list
            group = list(map(lambda x: list(x[1]), groupby(mask)))
            length = list(map(lambda x: len(x), group))
            group_length = [
                [sum(length[:n]), sum(length[:n]) + len(g)]
                for n, g in enumerate(group)
                if all(g)
            ]

            # get entity
            for g in group_length:
                result.append([i, g])
        result = sorted(result, key=lambda x: x[1][0])
        return result
        
