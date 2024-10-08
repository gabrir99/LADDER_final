import os
import re
from itertools import chain, groupby
from typing import List

import transformers
import torch
from torch import nn

PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning


def additional_special_tokens(tokenizer):
    """
    Get additional special token for beginning/separate/ending, {'input_ids': [0], 'attention_mask': [1]}.
    Basically it returns the position of [CLS] and [SEP] inside the tokenized sentence tokenizer.encode_plus('sent1', 'sent2')

    Parameters:
    tokenizer (transformers.AutoTokenizer): model tokenizer

    Returns:
    int, int, int: sp_token_start, sp_token_sep, sp_token_end
    """

    # The encode_plus method returns a dictionary with keys such as 'input_ids' and 'attention_mask'
    encode_first = tokenizer.encode_plus(
        "sent1", "sent2"
    )  # [CLS] sent1 [SEP] sent2 [SEP]
    # group by block boolean
    # This creates a mask (a list of boolean values) indicating which tokens in the encoded input are special tokens. [CLS], [SEP]
    sp_token_mask = [i in tokenizer.all_special_ids for i in encode_first["input_ids"]]
    group = [list(g) for _, g in groupby(sp_token_mask)]
    length = [len(g) for g in group]
    group_length = [
        [sum(length[:n]), sum(length[:n]) + len(g)]
        for n, g in enumerate(group)
        if all(g)
    ]
    assert len(group_length) == 3, "more than 3 special tokens group: {}".format(group)
    sp_token_start = {
        k: v[group_length[0][0] : group_length[0][1]] for k, v in encode_first.items()
    }
    sp_token_sep = {
        k: v[group_length[1][0] : group_length[1][1]] for k, v in encode_first.items()
    }
    sp_token_end = {
        k: v[group_length[2][0] : group_length[2][1]] for k, v in encode_first.items()
    }
    return sp_token_start, sp_token_sep, sp_token_end


class TokenizerNER:
    """NER specific transform pipeline"""

    def __init__(self, transformer_tokenizer: str, cache_dir: str = None):
        """
        NER specific transform pipeline, basically all the preprocessing steps before giving the ids to the model in order
        to get a prediction

        Parameters:
        transform_tokenizer (str): name of the model for which we need to download the tokenizer
        cache_dir (str): dir in which we will cache informations
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            transformer_tokenizer, cache_dir=cache_dir
        )
        self.pad_ids = {
            "labels": PAD_TOKEN_LABEL_ID,
            "input_ids": self.tokenizer.pad_token_id,
            "__default__": 0,
        }  # id for padding for both input and labels
        self.prefix = self.__sp_token_prefix()
        # find special tokens to be added
        self.sp_token_start, _, self.sp_token_end = additional_special_tokens(
            self.tokenizer
        )
        self.sp_token_end["labels"] = [self.pad_ids["labels"]] * len(
            self.sp_token_end["input_ids"]
        )
        self.sp_token_start["labels"] = [self.pad_ids["labels"]] * len(
            self.sp_token_start["input_ids"]
        )

    def __sp_token_prefix(self):
        """
        Used to tokenize a fixed sentence for testing purpouses.
        """
        sentence_go_around = "".join(
            self.tokenizer.tokenize("get tokenizer specific prefix")
        )  # create a string with the tokens all merged together
        # the regex iterator is transformed to a list and we take the first match,
        # the match element has the span() method that return (start, end) and we access the first element, thus "start"
        # we then take the sentence_go_around[:start]
        return sentence_go_around[
            : list(re.finditer("get", sentence_go_around))[0].span()[0]
        ]

    def fixed_encode_en(self, tokens, labels: List = None, max_seq_length: int = 128):
        """
        Fixed encoding for language with halfspace in between words basically use the tokenizer class to obtain the list
        of ids for both sentence and labels thus dictonary containing:
        - attention_mask
        - input_ids
        - lable ids
        - other fields
        """
        encode = self.tokenizer.encode_plus(
            " ".join(tokens),
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
        )
        if labels:
            assert len(tokens) == len(labels)
            fixed_labels = list(
                chain(
                    *[
                        [label]
                        + [self.pad_ids["labels"]]
                        * (len(self.tokenizer.tokenize(word)) - 1)
                        for label, word in zip(labels, tokens)
                    ]
                )
            )
            fixed_labels = [self.pad_ids["labels"]] * len(
                self.sp_token_start["labels"]
            ) + fixed_labels
            fixed_labels = fixed_labels[
                : min(
                    len(fixed_labels), max_seq_length - len(self.sp_token_end["labels"])
                )
            ]
            fixed_labels = fixed_labels + [self.pad_ids["labels"]] * (
                max_seq_length - len(fixed_labels)
            )
            encode["labels"] = fixed_labels
        return encode

    def fixed_encode_ja(self, tokens, labels: List = None, max_seq_length: int = 128):
        """fixed encoding for language without halfspace in between words"""
        dummy = "@"
        # get special tokens at start/end of sentence based on first token
        encode_all = self.tokenizer.batch_encode_plus(tokens)
        # token_ids without prefix/special tokens
        # `wifi` will be treated as `_wifi` and change the tokenize result, so add dummy on top of the sentence to fix
        token_ids_all = [
            [
                self.tokenizer.convert_tokens_to_ids(
                    _t.replace(self.prefix, "").replace(dummy, "")
                )
                for _t in self.tokenizer.tokenize(dummy + t)
                if len(_t.replace(self.prefix, "").replace(dummy, "")) > 0
            ]
            for t in tokens
        ]

        for n in range(len(tokens)):
            if n == 0:
                encode = {
                    k: v[n][: -len(self.sp_token_end[k])] for k, v in encode_all.items()
                }
                if labels:
                    encode["labels"] = [self.pad_ids["labels"]] * len(
                        self.sp_token_start["labels"]
                    ) + [labels[n]]
                    encode["labels"] += [self.pad_ids["labels"]] * (
                        len(encode["input_ids"]) - len(encode["labels"])
                    )
            else:
                encode["input_ids"] += token_ids_all[n]
                # other attribution without prefix/special tokens
                tmp_encode = {k: v[n] for k, v in encode_all.items()}
                s, e = len(self.sp_token_start["input_ids"]), -len(
                    self.sp_token_end["input_ids"]
                )
                input_ids_with_prefix = tmp_encode.pop("input_ids")[s:e]
                prefix_length = len(input_ids_with_prefix) - len(token_ids_all[n])
                for k, v in tmp_encode.items():
                    s, e = len(self.sp_token_start["input_ids"]) + prefix_length, -len(
                        self.sp_token_end["input_ids"]
                    )
                    encode[k] += v[s:e]
                if labels:
                    encode["labels"] += [labels[n]] + [self.pad_ids["labels"]] * (
                        len((token_ids_all[n])) - 1
                    )

        # add special token at the end and padding/truncate accordingly
        for k in encode.keys():
            encode[k] = encode[k][
                : min(len(encode[k]), max_seq_length - len(self.sp_token_end[k]))
            ]
            encode[k] += self.sp_token_end[k]
            pad_id = (
                self.pad_ids[k]
                if k in self.pad_ids.keys()
                else self.pad_ids["__default__"]
            )
            encode[k] += [pad_id] * (max_seq_length - len(encode[k]))
        return encode

    def encode_plus_all(
        self,
        tokens: List,
        labels: List = None,
        language: str = "en",
        max_length: int = None,
    ):
        """
        Encodes all the input sentences, and input labels using the defined tokenizer by adding padding and truncating
        if necessary.

        Parameters:
        tokens (list): it a list of sentences, the single sentence is in string form
        
        Returns:
        list: list of dictionries, each dictionaries contains a sentence example thus token_ids, label_ids, attention_mask, ...
        """
        max_length = (
            self.tokenizer.max_len_single_sentence if max_length is None else max_length
        )
        # TODO: no padding for prediction
        shared_param = {
            "language": language,
            "pad_to_max_length": True,
            "max_length": max_length,
        }
        if labels:
            return [self.encode_plus(*i, **shared_param) for i in zip(tokens, labels)]
        else:
            return [self.encode_plus(i, **shared_param) for i in tokens]

    def encode_plus(
        self,
        tokens,
        labels: List = None,
        language: str = "en",
        max_length: int = None,
        pad_to_max_length: bool = False,
    ):
        """
        Encode single list of tokens and list of labels, using a different
        encoding function if the language is japanese

        Parameters:
        tokens (str): it is a single sentence to tokenize
        """
        if labels is None:
            return self.tokenizer.encode_plus(
                tokens,
                max_length=max_length,
                padding="max_length" if pad_to_max_length else None,
                truncation=True,
            )
        if language == "ja":
            return self.fixed_encode_ja(tokens, labels, max_length)
        else:
            return self.fixed_encode_en(tokens, labels, max_length)

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    def tokenize(self, *args, **kwargs):
        return self.tokenizer.tokenize(*args, **kwargs)
    

class Dataset(torch.utils.data.Dataset):
    """
    torch.utils.data.Dataset wrapper converting into tensor
    """

    float_tensors = ["attention_mask"]

    def __init__(self, data: List):
        """
        Takes the dataseat already tokenized and provide the needed apis,
        in order to make the dataloader work with it.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}
