from itertools import chain
import logging
import random
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from time import time
from typing import Dict
from ladder_v2.ttpclassifier.attack_pattern_identification.config import MODELS, TOKEN_IDX, TOKENS
from ladder_v2.entity_extraction.ner.tokenizer_ner import Dataset
from ladder_v2.ttpclassifier.attack_pattern_identification.custom_model import EntityRecognition
import transformers
import torch
from torch import nn
from glob import glob
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
PROGRESS_INTERVAL = 100

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
CACHE_DIR = "{}/.cache/tner".format(os.path.expanduser("~"))

class AtkPatternEntityExtractionTrainer:
    def __init__(self, config: dict):
        """
        initialize trainer
        """
        logging.info("*** initialize network ***")

        self.__initialize_internal_fields(config)
        self.__print_chekpoint_configuration()
        self.__set_random_seed_in_all_libraries()
        self.__initialize_default_values_remaining_fields()

    def __initialize_internal_fields(self, config):
        self.config = config
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./ckpt")
        self.dataset = self.config.get("dataset", "data")
        self.transformer_name = self.config.get("transformer_name", "roberta-base")
        self.random_seed = self.config.get("random_seed", 1)
        self.lr = self.config.get("lr", 1e-5)
        self.epochs = self.config.get("epochs", 30)
        self.batch_size = self.config.get("batch_size", 32)
        self.max_seq_length = self.config.get("max_seq_length", 64)
        self.task_name = self.config.get("task_name", "entity_recognition")
        self.decay = self.config.get("decay", 0)
        self.freeze_bert = self.config.get("freeze_bert", False)
        self.lstm_dim = self.config.get("lstm_dim", -1)
        self.gradient_clip = self.config.get("gradient_clip", -1)


    def __print_chekpoint_configuration(self):
        logging.info("checkpoint: {}".format(self.checkpoint_dir))
        for k, v in self.config.items():
            logging.info(" - [arg] {}: {}".format(k, str(v)))

    def __set_random_seed_in_all_libraries(self):
        random.seed(self.random_seed)
        transformers.set_seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def __initialize_default_values_remaining_fields(self):
        self.label_to_id = {"O": 0, "ATK": 1}
        self.id_to_label = None
        self.criterion = nn.CrossEntropyLoss()
        self.dataset_split = None
        self.optimizer = None
        self.fine_tune = True
        self.num_class = 2
        self.patience = 10
        self.balance = False
        self.n_gpu = torch.cuda.device_count()
        self.device = "cuda" if self.n_gpu > 0 else "cpu"
        self.model = None
        self.__epoch = 1
        self.__step = 0
        self.__train_called = False
        self.early_stop = False
        self.val_patience = (
            0  # successive iteration when validation acc did not improve
        )

        self.iteration_number = 0

    def train_entity_extraction_model(
        self,
        monitor_validation: bool = True,
        batch_size_validation: int = 1,
        max_seq_length_validation: int = 256,
    ):
        self.__setup_model_data_and_dataset()

        writer = SummaryWriter(
            log_dir=self.checkpoint_dir
        )  # used by tensorboard to store logs for later visualization

        data_loader = self.__setup_all_data_loaders(
            monitor_validation, batch_size_validation, max_seq_length_validation
        )

        # start experiment
        self.__training_loop(writer, data_loader)

        self.model.from_pretrained(self.checkpoint_dir)
        if data_loader["test"]:
            self.__epoch_valid(data_loader["test"], writer=writer, prefix="test")

        writer.close()
        logging.info("ckpt saved at {}".format(self.checkpoint_dir))
        self.is_trained = True
        self.__train_called = True

    def __setup_all_data_loaders(
        self, monitor_validation, batch_size_validation, max_seq_length_validation
    ):
        data_loader = {
            "train": self.__setup_loader("train", self.batch_size, self.max_seq_length)
        }
        if monitor_validation and "valid" in self.dataset_split.keys():
            data_loader["valid"] = self.__setup_loader(
                "valid", batch_size_validation, max_seq_length_validation
            )
        else:
            data_loader["valid"] = None

        if "test" in self.dataset_split.keys():
            data_loader["test"] = self.__setup_loader(
                "test", batch_size_validation, max_seq_length_validation
            )
        else:
            data_loader["test"] = None
        return data_loader
    
    def __setup_loader(self, data_type: str, batch_size: int, max_seq_length: int):
        """
        Create the data loader by trasforming the dataset in input using the self.tokenizer class

        """
        assert self.dataset_split, "run __setup_data firstly"
        if data_type not in self.dataset_split.keys():
            return None
        
        encoded_dataset = []
        for words, labels in zip(self.dataset_split[data_type]["data"], self.dataset_split[data_type]["label"]):
            assert len(words) == len(labels)
            tokens_text = list(chain(*[self.tokenizer.tokenize(word) for word in words]))
            tokens = [self.start_token] + tokens_text + [self.end_token]
            fixed_labels = list(
                chain(
                    *[
                        [label]+[label]* (len(self.tokenizer.tokenize(word)) - 1)
                        for label, word in zip(labels, words)
                    ]
                )
            )
            labels_id = [PAD_TOKEN_LABEL_ID] + fixed_labels + [PAD_TOKEN_LABEL_ID]
            assert len(labels_id) == len(tokens)
            # padding or removing tokens if sequenc > self.sequence_len
            if len(tokens) < max_seq_length:
                tokens = tokens + [
                    self.pad_token for _ in range(max_seq_length - len(tokens))
                ]
                labels_id = labels_id + [ 
                    PAD_TOKEN_LABEL_ID for _ in range(max_seq_length - len(labels_id))
                ]
            else:
                tokens = tokens[: max_seq_length - 1] + [self.end_token]
                labels_id = labels_id[: max_seq_length - 1] + [PAD_TOKEN_LABEL_ID]


            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attn_mask = [id != self.pad_idx for id in tokens_ids]
            features = {
                "tokens_ids": tokens_ids,
                "attention_mask": attn_mask,
                "labels": labels_id
            }
            encoded_dataset.append(features)

        data_obj = Dataset(encoded_dataset)


        return torch.utils.data.DataLoader(
            data_obj,
            batch_size=batch_size,
            shuffle=True,
        )
    
    def __setup_model_data_and_dataset(self):
        """
        Creates the following things:
        - datasets to utilize
        - model, tokenizer
        - optimezer and scheduler
        - moves everything to GPU if present
        """
        if self.model is not None:
            return
     
        self.__load_model_from_huggingface_to_finetune()

        self.__create_optimizer_with_decay()

        #self.__create_scheduler_using_optimizer()

        # GPU allocation
        self.model.to(self.device)

        # GPU mixture precision
        #self.__reduce_memory_usage_fp16()

        # multi-gpus
        #self.__parallelize_training_if_possible()
    
    def __create_optimizer_with_decay(self):
        # optimizer
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.decay
        )
    def __load_model_from_huggingface_to_finetune(self):
        
        self.__get_dataset()

        self.__setup_tokenizer()
        # model
        self.__load_model()

    def __get_dataset(self):
        (
            self.dataset_split,
            self.label_to_id,
            self.unseen_entity_set,
        ) = self.__get_dataset_ner(
            fix_label_dict=True,
        )
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}

    def __get_dataset_ner(
        self,
        custom_data_path: str = None,
        fix_label_dict: bool = False,
    ):
        """
        Fetch NER dataset, if data_names is not only one it will merge togheter different dataset and
        returns a bigger dataset that is the merge of the passed names.

        Parameter:
        custom_data_path (str): Filepath to custom dataset
        custom_data_language (str): Language for custom_data_path dataset
        fix_label_dict (bool): Fixing given label_to_id dictionary (ignore label not in the dictionary in dataset)

        Returns:
        unified_data: dict
            A dataset consisting of 'train'/'valid' (only 'train' if more than one data set is used)
        label_to_id: dict
            A dictionary of label to id
        language: str
            Most frequent language in the dataset
        """
        assert (
            self.dataset or custom_data_path
        ), "either `dataset` or `custom_data_path` should be not None"
        data_names = self.dataset if self.dataset else []
        data_names = [data_names] if type(data_names) is str else data_names
        custom_data_path = [custom_data_path] if custom_data_path else []
        data_list = data_names + custom_data_path
        logging.info("target dataset: {}".format(data_list))
        data = []
        unseen_entity_set = {}
        param = dict(
            fix_label_dict=fix_label_dict,
        )
        for (
            d
        ) in (
            data_list
        ):  # processes all the different datasets and puts them in a list of datasets
            param["label_to_id"] = self.label_to_id
            data_split_all, label_to_id, ues = self.__get_dataset_ner_single(
                d, **param
            )
            data.append(data_split_all)
            unseen_entity_set = (
                ues
                if len(unseen_entity_set) == 0
                else unseen_entity_set.intersection(ues)
            )
        if (
            len(data) > 1
        ):  # if we want to train on different datasets together it will chain them togheters.
            unified_data = {
                "train": {
                    "data": list(chain(*[d["train"]["data"] for d in data])),
                    "label": list(chain(*[d["train"]["label"] for d in data])),
                }
            }
        else:
            unified_data = data[0]
        # use the most frequent language in the data
        return unified_data, label_to_id, unseen_entity_set

    def __get_dataset_ner_single(
        self,
        data_name: str = None,
        label_to_id: dict = None,
        fix_label_dict: bool = False,
        cache_dir: str = None,
    ):
        """
        Download dataset file and return dictionary including training/validation split, fucntion
        created to handle only custom dataset.
        It returns the processed dataset ready for use.

        :param data_name: data set name or path to the data
        :param label_to_id: fixed dictionary of (label: id). If given, ignore other labels
        :param fix_label_dict: not augment label_to_id based on dataset if True
        :param lower_case: convert to lower case
        :param custom_language
        :param allow_new_entity
        :return: formatted data, label_to_id
        """
        entity_first = False
        cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        data_path = os.path.join(cache_dir, data_name)
        logging.info("data_name: {}".format(data_name))

        # for custom data
        data_path = data_name
        if not os.path.exists(data_path):
            raise ValueError("unknown dataset: %s" % data_path)
        else:
            files = glob("{}/*.txt".format(data_path))
            logging.info("formatting custom dataset from {}".format(data_path))
            files_info = {}
            for _file in files:
                _file = os.path.basename(_file)
                if _file == "train.txt":
                    files_info["train"] = _file
                elif _file in ["valid.txt", "val.txt", "validation.txt", "dev.txt"]:
                    files_info["valid"] = _file
                elif _file == "test.txt":
                    files_info["test"] = _file
            assert (
                "train" in files_info
            ), "training set not found, make sure you have `train.txt` in the folder"
            logging.info("found following files: {}".format(files_info))
            logging.info(
                "note that files should be named as either `valid.txt`, `test.txt`, or `train.txt` "
            )

        label_to_id = dict() if label_to_id is None else label_to_id
        data_split_all, unseen_entity_set, label_to_id = self.__decode_all_files(
            files_info,
            data_path,
            label_to_id=label_to_id,
            fix_label_dict=fix_label_dict,
            entity_first=entity_first,
        )

    
        return data_split_all, label_to_id, unseen_entity_set

    def __decode_all_files(
        self,
        files: Dict,
        data_path: str,
        label_to_id: Dict,
        fix_label_dict: bool,
        entity_first: bool = False,
    ):
        """
        Opens the valid, text and train file and builds the custom datasets.

        Returns:
        dict, list, dict: data_split (dictionary containing test, train, val datasets), unseen_entity, label_to_id
        """
        data_split = dict()
        unseen_entity = None
        for (
            name,
            filepath,
        ) in (
            files.items()
        ):  # creates the dictionary containing test, train, val datasets
            label_to_id, unseen_entity_set, data_dict = self.__decode_file(
                filepath,
                data_path=data_path,
                label_to_id=label_to_id,
                fix_label_dict=fix_label_dict,
                entity_first=entity_first,
            )
            if unseen_entity is None:
                unseen_entity = unseen_entity_set
            else:
                unseen_entity = unseen_entity.intersection(unseen_entity_set)
            data_split[name] = data_dict
            logging.info(
                "dataset {0}/{1}: {2} entries".format(
                    data_path, filepath, len(data_dict["data"])
                )
            )
        return data_split, unseen_entity, label_to_id

    def __decode_file(
        self,
        file_name: str,
        data_path: str,
        label_to_id: Dict,
        fix_label_dict: bool,
        entity_first: bool = False,
    ):
        """
        Open a single file and loads the dataset, normalize the tags and trasform to BIO tagging if to_bio=True,

        Parameters:

        Returns:
        dict, list, dict: label_to_id, unseen_entity_label, dict containing the data and the labels {data: [..] labels: [..]}
        """
        inputs, labels, seen_entity = [], [], []
        past_mention = "O"
        with open(os.path.join(data_path, file_name), "r", encoding="utf8") as f:
            sentence, entity = [], []
            for n, line in enumerate(f):
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(sentence) != 0:
                        assert len(sentence) == len(entity)
                        inputs.append(sentence)
                        labels.append(entity)
                        sentence, entity = [], []
                else:
                    ls = line.split()
                    if len(ls) < 2:
                        continue
                    # Examples could have no label for mode = "test"
                    if entity_first:
                        tag, word = ls[0], ls[-1]
                    else:
                        word, tag = ls[0], ls[-1]
                    if tag == "junk":
                        continue
                    # if word in STOPWORDS:
                    #     continue
                    sentence.append(word)

                    # if label dict is fixed, unknown tag type will be ignored
                    if tag not in label_to_id.keys() and fix_label_dict:
                        tag = "O"
                    elif tag not in label_to_id.keys() and not fix_label_dict:
                        label_to_id[tag] = len(label_to_id)

                    entity.append(label_to_id[tag])

        id_to_label = {v: k for k, v in label_to_id.items()}
        unseen_entity_id = set(label_to_id.values()) - set(list(chain(*labels)))
        unseen_entity_label = {id_to_label[i] for i in unseen_entity_id}
        return label_to_id, unseen_entity_label, {"data": inputs, "label": labels}

    def __training_loop(self, writer, data_loader):
        start_time = time()
        best_f1_score = -1
        logging.info(
            "*** start training from step {}, epoch {} ***".format(
                self.__step, self.__epoch
            )
        )
        try:
            while True:
                end_training = self.__epoch_train(data_loader["train"], writer=writer)
                self.__release_cache()
                if data_loader["valid"]:
                    best_f1_score = (
                        self.__execute_valid_epoch_and_save_model_if_best_score(
                            writer, data_loader, best_f1_score
                        )
                    )

                    self.__release_cache()
                if end_training or self.early_stop:
                    break
                self.__epoch += 1
        except RuntimeError:
            logging.exception("*** RuntimeError ***")

        except KeyboardInterrupt:
            logging.info("*** KeyboardInterrupt ***")

        logging.info(
            "[training completed, {} sec in total]".format(time() - start_time)
        )
        if best_f1_score < 0:
            self.model.save_pretrained(self.checkpoint_dir)
            self.tokenizer.tokenizer.save_pretrained(self.checkpoint_dir)

    def __execute_valid_epoch_and_save_model_if_best_score(
        self, writer, data_loader, best_f1_score
    ):
        try:
            metric = self.__epoch_valid(
                data_loader["valid"], writer=writer, prefix="valid"
            )
            if metric["f1"] > best_f1_score:
                best_f1_score = metric["f1"]
                self.model.save_pretrained(self.checkpoint_dir)
                self.tokenizer.save_pretrained(self.checkpoint_dir)
                self.val_patience = 0
            else:
                self.val_patience +=1
                if self.val_patience == self.patience:
                    self.early_stop = True
        except RuntimeError:
            logging.exception("*** RuntimeError: skip validation ***")
        return best_f1_score
    
    def __epoch_train(self, data_loader, writer):
        """single epoch training: returning flag which is True if training has been completed"""
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0
        for encode in data_loader:# (input ids, attention masks, output labels ids)xbatch_size
            encode = {k: v.to(self.device) for k, v in encode.items()}
            x, y, att = encode["tokens_ids"], encode["labels"], encode["attention_mask"]
            y_pred = self.model(x, att)
            y_pred = y_pred.view(-1, y_pred.shape[2])
            y = y.view(-1)
            loss = self.criterion(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1).view(-1)

            att = att.view(-1)
            total += torch.sum(att).item() - 2 * x.shape[0]
            correct += (
                torch.sum(att * (y_pred == y).long()).item() - 2 * x.shape[0]
            )
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()
            train_loss += loss.item()

            # log instantaneous accuracy, loss, and learning rate
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]["lr"]
            self.__log_train_epoch_step(writer, inst_loss, inst_lr, train_loss, total, correct, self.__step+1)
            self.__step += 1

        if self.__epoch >= self.epochs:
            logging.info("reached maximum epochs")
            return True

        return False
    
    def __log_train_epoch_step(self, writer, inst_loss, inst_lr, train_loss, total, correct, iteration):
        avg_loss = train_loss / iteration
        train_acc = correct / total
        if writer:
            writer.add_scalar("train/loss", avg_loss, self.__step)
            writer.add_scalar("train/learning_rate", inst_lr, self.__step)
        if self.__step % PROGRESS_INTERVAL == 0:
            logging.info(
                "[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f, accuracy: %.3f"
                % (self.__epoch, self.__step, avg_loss, inst_lr, train_acc)
            )

    def __epoch_valid(
        self,
        data_loader,
        prefix,
        writer=None
    ):
        """single epoch validation/test"""
        # aggregate prediction and true label
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        tp = np.zeros(1 + len(self.label_to_id), dtype=np.int64)
        fp = np.zeros(1 + len(self.label_to_id), dtype=np.int64)
        fn = np.zeros(1 + len(self.label_to_id), dtype=np.int64)
        cm = np.zeros((len(self.label_to_id), len(self.label_to_id)), dtype=np.int64)

        with torch.no_grad():
            for encode in data_loader:# (input ids, attention masks, output labels ids)xbatch_size
                encode = {k: v.to(self.device) for k, v in encode.items()}
                x, y, att = encode["tokens_ids"], encode["labels"], encode["attention_mask"]
                y_pred = self.model(x, att)
                
                y = y.view(-1)
                y_pred = y_pred.view(-1, y_pred.shape[2])
                loss = self.criterion(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1).view(-1)
                val_loss += loss.item()
                att = att.view(-1)
                # subtract 2 for start and end sequence tokens
                correct += (
                    torch.sum(att * (y_pred == y).long()).item() - 2 * x.shape[0]
                )
                total += torch.sum(att).item() - 2 * x.shape[0]


                self.__compute_confusion_matrix_for_non_bio_tagging(tp, fp, fn, cm, x, y, att, y_pred)
        # ignore first index which is for no entity
        tp_score = tp[1]
        fp_score = fp[1]
        fn_score= fn[1]
        precision = tp_score / (tp_score + fp_score)
        recall = tp_score / (tp_score + fn_score)
        f1 = 2 * precision * recall / (precision + recall)

      
        # compute metrics
        metric = {
            "f1": f1 * 100,
            "recall": recall * 100,
            "precision": precision * 100,
        }

        self.__compute_report_and_log_info(prefix, writer, cm, metric)
        return metric

    def __compute_confusion_matrix_for_non_bio_tagging(self, tp, fp, fn, cm, x, y, att, y_pred):
        sos_token = TOKEN_IDX[self.token_style]["START_SEQ"]
        eos_token = TOKEN_IDX[self.token_style]["END_SEQ"]
        x = x.view(-1)
        
        for i in range(y.shape[0]):
            input_token = x[i].item()
            if (
                        att[i] == 0
                        or input_token == sos_token
                        or input_token == eos_token
                    ):
                        # do not count these as they are trivially no-entity tokens
                continue
            cor = y[i]
            prd = y_pred[i]
            if cor == prd:
                tp[cor] += 1
            else:
                fn[cor] += 1
                fp[prd] += 1
            cm[cor][prd] += 1
    
    def __convert_true_ids_and_pred_ids_to_label_in_a_sentence(
        self, unseen_entity_set, _true, _pred, b, _pred_list, _true_list
    ):
        for s in range(len(_true[b])):  # iterate over each element in a sequence
            if _true[b][s] != PAD_TOKEN_LABEL_ID:
                _true_list.append(
                    self.id_to_label[_true[b][s]]
                )  # transform id to label
                if unseen_entity_set is None:
                    _pred_list.append(self.id_to_label[_pred[b][s]])
                else:
                    __pred = self.id_to_label[_pred[b][s]]
                    if __pred in unseen_entity_set:
                        _pred_list.append("O")
                    else:
                        _pred_list.append(__pred)
        assert len(_pred_list) == len(_true_list)


    def __compute_report_and_log_info(self, prefix, writer, confusion_matrix, metric):
        try:
            summary = confusion_matrix
            logging.info("[epoch {}] ({}) \n {}".format(self.__epoch, prefix, summary))
            logging.info("f1 score: {}".format(metric["f1"]))
            logging.info("recall: {}".format(metric["recall"]))
            logging.info("precision: {}".format(metric["precision"]))
        except Exception:
            logging.exception("classification_report raises error")
            summary = ""
        metric["summary"] = summary
        if writer:
            writer.add_scalar("{}/f1".format(prefix), metric["f1"], self.__epoch)
            writer.add_scalar(
                "{}/recall".format(prefix), metric["recall"], self.__epoch
            )
            writer.add_scalar(
                "{}/precision".format(prefix), metric["precision"], self.__epoch
            )       
    def __load_model(self):
        self.model = EntityRecognition(
            self.transformer_name,
            freeze_bert=self.freeze_bert,
            lstm_dim=self.lstm_dim,
        )

    def __setup_tokenizer(self):
        tokenizer = MODELS[self.transformer_name][1]  # (AutoModel, AutoTokenizer, 768, 'roberta'),
        self.tokenizer = tokenizer.from_pretrained(self.transformer_name)
        self.token_style = MODELS[self.transformer_name][
            3
        ]  # (AutoModel, AutoTokenizer, 768, 'roberta')
        self.start_token = TOKENS[self.token_style]["START_SEQ"]
        self.end_token = TOKENS[self.token_style]["END_SEQ"]
        self.pad_token = TOKENS[self.token_style]["PAD"]
        self.pad_idx = TOKEN_IDX[self.token_style]["PAD"]

    
    def __release_cache(self):
        """
        Called after an epoch, used to empty the cache,
        if the used device is cuda.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

