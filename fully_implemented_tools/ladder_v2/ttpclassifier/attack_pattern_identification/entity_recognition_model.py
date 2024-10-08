import logging
from typing import List
from itertools import groupby
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ladder_v2.ttpclassifier.attack_pattern_identification.config import MODELS, TOKEN_IDX, TOKENS
from ladder_v2.ttpclassifier.attack_pattern_identification.custom_model import EntityRecognition
import torch
from torch import nn
import nltk
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)



class EntityRecognitionModel():
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
        self.load_entity_recognition_model(model_name, dir_path)
        self.tokenizer = MODELS[model_name][1]
        self.token_style = MODELS[model_name][3]
        self.tokenizer = self.tokenizer.from_pretrained(model_name)
        self.sequence_len = seq_len
        self.entity_mapping = {"O": 0, "ATK": 1}

        
        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = "cuda" if self.n_gpu > 0 else "cpu"
        self.model.to(self.device)

    def load_entity_recognition_model(self, model_name, dir_path):
        self.model = EntityRecognition(
            model_name,
        )
        self.model.load_state_dict(
            torch.load(os.path.join(dir_path, "weights.pt"))
        )

    def __extract_attack_patterns(self, sent):
        """
        Tokenize sentence and add start_token, end_token and pad_token if necessary before calling the model on the 
        input tokens.
        The model will predict for each token if it is part of an attack pattern or not, after that function will return
        only a str containing sentences that are attack pattern

        Parameters:
        sent (str): sentence to tokenize

        Returns:
        str: return text containing sentences that are attack pattern separated by \n
        """
        words_original_case = nltk.word_tokenize(sent)
        words = [x.lower() for x in words_original_case]
        token_to_word_mapping = {}

        word_pos = 0
        x = [TOKEN_IDX[self.token_style]["START_SEQ"]]
        while word_pos < len(words):
            tokens = self.tokenizer.tokenize(words[word_pos])

            if len(tokens) + len(x) >= self.sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(self.tokenizer.convert_tokens_to_ids(tokens[i]))
                x.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
                token_to_word_mapping[len(x) - 1] = words_original_case[word_pos] #saves the mapping between tokens and words
                word_pos += 1
        x.append(TOKEN_IDX[self.token_style]["END_SEQ"])
        if len(x) < self.sequence_len:
            x = x + [TOKEN_IDX[self.token_style]["PAD"] for _ in range(self.sequence_len - len(x))]
        attn_mask = [1 if token != TOKEN_IDX[self.token_style]["PAD"] else 0 for token in x]

        x = torch.tensor(x).reshape(1, -1)
        attn_mask = torch.tensor(attn_mask).reshape(1, -1)
        x, attn_mask = x.to(self.device), attn_mask.to(self.device)

        ret = ""
        cur = ""
        cur_word_count = 0
        with torch.no_grad():
            y_pred = self.model(x, attn_mask)
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            x = x.view(-1)
            for i in range(y_pred.shape[0]): #i is the index of the token that has been predicted
                if x[i].item() == TOKEN_IDX[self.token_style]["PAD"]:
                    break

                token_pred = torch.argmax(y_pred[i]).item()

                # print(tokenizer.convert_ids_to_tokens(x[i].item()), token_pred)

                if i in token_to_word_mapping: #contains mapping last_token_pos to word to which tokens belongs mapping
                    if token_pred == self.entity_mapping["ATK"]: #for each input tokens it predict {"O": 0, "ATK": 1}
                        cur += token_to_word_mapping[i] + " "
                        cur_word_count += 1
                    else:
                        if len(cur) > 0 and cur_word_count >= 2 and self.__is_valid_step(cur):
                            ret += cur[:-1] + "\n" 
                            cur = ""
                            cur_word_count = 0
                        else:
                            cur = ""
                            cur_word_count = 0
            if len(cur) > 0 and cur_word_count >= 2:
                ret += cur[:-1] + "\n"
        return ret
    
    def __is_valid_step(self, text):
        """
        Check is text contains at least a verb.
        """
        verb_codes = {
            "VB",  # Verb, base form
            "VBD",  # Verb, past tense
            "VBG",  # Verb, gerund or present participle
            "VBN",  # Verb, past participle
            "VBP",  # Verb, non-3rd person singular present
            "VBZ",  # Verb, 3rd person singular present
        }
        pos = nltk.pos_tag(nltk.word_tokenize(text))
        for x in pos:
            if x[1] in verb_codes:
                return True
        return False
    
    def get_attack_patterns_from_sentences(self, sentences : list):
        output = []
        for sent in sentences:
            result = self.__extract_attack_patterns(sent)
            array = result.split('\n')
            obj = {
                "sent" : sent,
                "attack_patterns": [string.strip() for string in array if string.strip()]
            }
            output.append(obj)
        return output
    
    def get_attack_patterns_from_already_classified_sentences(self, classified_sent : list):
        output = []
        for sent in classified_sent:
            if sent["relevant"] == 0: #skipe sentences that are not relevant
                continue
            result = self.__extract_attack_patterns(sent["sent"])
            array = result.split('\n')
            obj = {
                "sent" : sent["sent"],
                "attack_patterns": [string.strip() for string in array if string.strip()]
            }
            output.append(obj)
        return output