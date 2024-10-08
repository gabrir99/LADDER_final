import csv
import json
import logging
from typing import List
from itertools import groupby
import os
import sys
import numpy as np


# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ladder_v2.relation_extraction.custom_model import RelationClassificationBERT, RelationClassificationRoBERTa
from ladder_v2.relation_extraction.config import MODELS

import torch
from torch import nn

#for now we will just consider, threat-actor, malware, identity, indicator, tool
map_stix_notation = {
    'Malware': 'malware',
    'Application': 'tool',
    'MalwareType': 'malware',
    'Person': 'identity',
    'Organization': 'identity',
    'Time': None,
    'OS': 'tool',
    'location': 'location',
    'ThreatActor': 'threat-actor',
    'Filepath-Win': 'x-file',
    'Filepath-Unix': 'x-file',
    'Email': 'indicator',
    'SHA256': 'indicator',
    'SHA1':'indicator',
    'CVE': 'indicator',
    'IP': 'ipv4-addr'
}


class RelationExtractionModel():
    def __init__(self, seq_len, model_name: str, dir_path: str, num_class:int, csv_rel_file: str):
        """
        initialize sentence classification model, in order to use it for prediction
        purpouses

        Parameters:
        model_name (str): 
            - model name on transformers model hub  or
            - path to model directory
        """
        logging.info("*** initialize network ***")
        self.load_relation_extraction_model(model_name, dir_path, num_class)
        self.existing_rel_dict = self.__read_csv_to_dict(csv_rel_file)
        self.tokenizer = MODELS[model_name][1]
        self.tokenizer = self.tokenizer.from_pretrained(model_name)
        self.sequence_len = seq_len
        self.model_name = model_name
        self.label_to_id = {
            "N/A": 0,
            "isA" : 1,
            "targets": 2,
            "uses":3,
            "hasAuthor":4,
            "has":5,
            "variantOf":6,
            "hasAlias":7,
            "indicates":8,
            "discoveredIn":9,
            "exploits": 10,
        }
        # Create the reverse mapping
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}        
        self.dataset_split = None

        
        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = "cuda" if self.n_gpu > 0 else "cpu"
        self.model.to(self.device)

    def load_relation_extraction_model(self, model_name, dir_path, num_class):
        if "roberta" in model_name:
            self.model =  RelationClassificationRoBERTa.from_pretrained(
                model_name,  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=num_class,  # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
            self.model.load_state_dict(
                torch.load(os.path.join(dir_path, "weights.pt"))
            )
        else:
            self.model = RelationClassificationBERT.from_pretrained(
                model_name,  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=num_class,  # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
            self.model.load_state_dict(
                torch.load(os.path.join(dir_path, "weights.pt"))
            )

    def __get_rel_from_sent(self, sent):
        """
        Returns one of the possible 10 predictable relationship types, that the model is able to extract
        """
        e1_id, e2_id = self.__extract_e1_e2_ids_based_on_used_model()

        encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.sequence_len,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
                truncation=True,
            )
        # Find e1(id:2487) and e2(id:2475) position
        # nonzero() returns indices of positions where the value was not false,
        # after that [0] we take first occurence of e1, and then [1] we take the index
        e1_pos, e2_pos = [], []
        pos1 = (encoded_dict["input_ids"] == e1_id).nonzero()[0][1].item()
        pos2 = (encoded_dict["input_ids"] == e2_id).nonzero()[0][1].item()
        e1_pos.append(pos1)
        e2_pos.append(pos2)
        # Convert the lists into tensors.
        input_id = encoded_dict["input_ids"].to(self.device)
        attention_mask = encoded_dict["attention_mask"].to(self.device)
        e1_pos = torch.tensor(e1_pos, device=self.device)
        e2_pos = torch.tensor(e2_pos, device=self.device)
        w = torch.ones(len(e1_pos), device=self.device)

        with torch.no_grad():
            logits = self.model(
                        input_id,
                        token_type_ids=None,
                        attention_mask=attention_mask,
                        e1_pos=e1_pos,
                        e2_pos=e2_pos,
                        w=w,
                    )[0]
            pred_flat = torch.argmax(logits, axis=1).item()
        return self.id_to_label[pred_flat]

    def get_relations_from_sentences(self, annotated_sent_list : list):
        """
        Parameters:
        annotated_sent_list (list): list of dictionary where each dict is made of the following 3 fields
        {"entity": _entities, "sentence": sentence, "annotated_sentences": annotated_list}
        where the annotated_list is made of dictionaries containing
                'annotated': sentence containing e1, e2
                'e1': the dictionary obj contained in _entities,
                'e2': the dictionary obj contained in _entities
        we will have a dictionary for each possible comb of entities that generates a valid stix relation

        Returns:
        annotated_sent_list (list): it adds for each dictionary in the list a new key "found_relations", that
        contains for each sentence information about the extracted relations in the format
                "annotated_sent":
                "found_rel":
                "e1":
                "e2":
        """
        
        for sent in annotated_sent_list:
            output = []
            for annotated_sent in sent["annotated_sent_comb"]:
                result = self.__get_rel_from_sent(annotated_sent["annotated"])
                obj = {
                    "annotate_sent" : annotated_sent["annotated"],
                    "found_rel": result,
                    "e1": annotated_sent["e1"],
                    "e2": annotated_sent["e2"]
                }
                output.append(obj)
            sent["found_relations"] = output
        return annotated_sent_list
    
    def __extract_e1_e2_ids_based_on_used_model(self):
        e1_id = -1
        e2_id = -1
        if self.model_name in ["bert-base-uncased", "bert-large-uncased"]:
            e1_id = 2487
            e2_id = 2475
        elif self.model_name in ["roberta-base", "roberta-large"]:
            e1_id = 134
            e2_id = 176
        elif self.model_name in ["xlm-roberta-base", "xlm-roberta-large"]:
            e1_id = 418
            e2_id = 304
        else:
            raise ValueError("Unknown Model")
        return e1_id,e2_id
    
    def annotate_sentences_based_on_found_entites(self, found_entities_list: list):
        """ 
        Objective is annotating the sentence extracted from the text with the e1, e2 annotation needed for
        the relation extraction model in order to classify a relation.
        When annotating a sentence we will consider all the possible combination between the entity in the sentence,
        taking into account the relations.csv file that contains all the possible exisiting relation among different entities.

        Parameters:
        found_entities_list (list): list of dictionary where each dict consists of {"entity": _entities, "sentence": sentence}
            where entities is a list of dictionary containing
                'type': (str) entity type
                'position': (list) start position and end position
                'mention': (str) mention

        Returns:
        annotated_sentences (list): list of dictionary where each dict is made of the following 3 fields
        {"entity": _entities, "sentence": sentence, "annotated_sentences": annotated_list}
        where the annotated_list is made of dictionaries containing
                'annotated': sentence containing e1, e2
                'e1': the dictionary obj contained in _entities,
                'e2': the dictionary obj contained in _entities
        we will have a dictionary for each possible comb of entities that generates a valid stix relation
        """
        for sent_obj in found_entities_list:
            entities_in_sent = sent_obj["entity"]
            sent_obj["annotated_sent_comb"] = []
            for i in range(len(entities_in_sent)):
                for j in range(i+1, len(entities_in_sent)):
                    e1 = entities_in_sent[i]
                    e2 = entities_in_sent[j]
                    if e1["position"][0] > e2["position"][0]:
                        tmp = e1
                        e1 = e2
                        e2 = tmp
                    ent1_type = map_stix_notation[ e1["type"]]
                    ent2_type = map_stix_notation[ e2["type"]]
                    possible_rel_list = []
                    if ent1_type and ent2_type:
                        try:
                            possible_rel_list.append(self.existing_rel_dict[ent1_type][ent2_type])
                        except KeyError:
                            logging.info(f"The following relation doesn't exist : {ent1_type} --> {ent2_type}")
                            logging.info(f"Check if reverse relations exist..")
                            try:
                                possible_rel_list.append(self.existing_rel_dict[ent2_type][ent1_type])
                            except KeyError:
                                print(f"Also the following relation doesn't exist : {ent2_type} --> {ent1_type}")


                    if len(possible_rel_list) > 0:
                        #TODO aggiungo di non aggiungere due volte relazione se già è stata inclusa in qualche modo, 
                        # ad esempio relazioni inverse           
                        annotated_sent =  self.__enclose_words(sent_obj["sentence"],  e1["position"][0], e1["position"][1], 
                                                               e2["position"][0], e2["position"][1])
                        sent_obj["annotated_sent_comb"].append({
                            "annotated": annotated_sent,
                            "e1": e1,
                            "e2": e2
                        })

        return found_entities_list

    def __enclose_words(self, sentence, start_pos_1, end_pos_1, start_pos_2, end_pos_2):
        # Enclose the second word first to avoid messing up the positions
        sentence = sentence[:start_pos_2] + "<e2>" + sentence[start_pos_2:end_pos_2] + "</e2>" + sentence[end_pos_2:]

        # # Adjust the second word positions after enclosing the first word
        # adjustment = len("<e2></e2>")
        # start_pos_1 += adjustment
        # end_pos_1 += adjustment

        # Enclose the first word
        sentence = sentence[:start_pos_1] + "<e1>" + sentence[start_pos_1:end_pos_1] + "</e1>" + sentence[end_pos_1:]

        return sentence
    
    def __read_csv_to_dict(self, csv_file="relations.csv"):
        result_dict = {}
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                src = row['src']
                dst = row['dst']
                type_ = row['type']
                reverse = row['reverse']

                if src not in result_dict:
                    result_dict[src] = {}

                if dst not in result_dict[src]:
                    result_dict[src][dst] = []

                result_dict[src][dst].append({'type': type_, 'reverse': reverse})
        return result_dict

    def __pretty_print_dict(self, dictionary):
        print(json.dumps(dictionary, indent=4))


