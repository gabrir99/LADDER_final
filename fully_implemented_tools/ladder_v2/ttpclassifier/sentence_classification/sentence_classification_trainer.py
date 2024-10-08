from itertools import chain
import logging
import random
import os
import sys

# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from time import time
from typing import Dict
from ladder_v2.ttpclassifier.sentence_classification.custom_models import SentenceClassificationBERT, SentenceClassificationRoBERTa
from ladder_v2.ttpclassifier.sentence_classification.sentence_classification_dataset import Dataset
from ladder_v2.ttpclassifier.sentence_classification.config import MODELS, TOKEN_IDX, TOKENS
import transformers
import torch
from torch import nn
from glob import glob
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

PROGRESS_INTERVAL = 100

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

class SentenceClassificationTrainer:
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
        self.transformer_name = self.config.get("transformer_name", "bert-base-uncased")
        self.random_seed = self.config.get("random_seed", 1)
        self.lr = self.config.get("lr", 1e-5)
        self.epochs = self.config.get("epochs", 20)
        self.batch_size = self.config.get("batch_size", 32)
        self.max_seq_length = self.config.get("max_seq_length", 64)
        self.task_name = self.config.get("task_name", "atk-pattern")
        self.decay = self.config.get("decay", 0)

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

    def train_classification_model(
        self,
        monitor_validation: bool = True,
        batch_size_validation: int = 1,
        max_seq_length_validation: int = 128,
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
        for row in self.dataset_split[data_type]:
            text = row["text"]
            label = row["label"]

            tokens_text = self.tokenizer.tokenize(text)
            tokens = [self.start_token] + tokens_text + [self.end_token]
            # padding or removing tokens if sequenc > self.sequence_len
            if len(tokens) < max_seq_length:
                tokens = tokens + [
                    self.pad_token for _ in range(max_seq_length - len(tokens))
                ]
            else:
                tokens = tokens[: max_seq_length - 1] + [self.end_token]


            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attn_mask = [id != self.pad_idx for id in tokens_ids]
            features = {
                "tokens_ids": tokens_ids,
                "attention_mask": attn_mask,
                "label": label
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
        
        self.dataset_split = self.__get_dataset()

        self.__setup_tokenizer()
        # model
        self.__load_model()

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
            x, y, att = encode["tokens_ids"], encode["label"], encode["attention_mask"]
            y_pred = self.model(x, att)
            loss = self.criterion(y_pred, y)  # crossentropy computation

            # shape[0] number of samples in the batch
            train_loss += (
                loss.item() * y.shape[0]
            )  # loss scaled for the number of elements in the batch
            total += y.shape[0]  # total number of exampled that have been processed
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log instantaneous accuracy, loss, and learning rate
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]["lr"]
            self.__log_train_epoch_step(writer, inst_loss, inst_lr, train_loss, total, correct)
            self.__step += 1

        if self.__epoch >= self.epochs:
            logging.info("reached maximum epochs")
            return True

        return False
    
    def __log_train_epoch_step(self, writer, inst_loss, inst_lr, train_loss, total, correct):
        avg_loss = train_loss / total
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
        writer=None,
    ):
        """single epoch validation/test"""
        # aggregate prediction and true label
        self.model.eval()
        test_loss = 0
        total = 0
        correct = 0

        y_true_all = np.zeros((0,), dtype=int)
        y_pred_all = np.zeros((0,), dtype=int)

        with torch.no_grad():
            for encode in data_loader:# (input ids, attention masks, output labels ids)xbatch_size
                encode = {k: v.to(self.device) for k, v in encode.items()}
                x, y, att = encode["tokens_ids"], encode["label"], encode["attention_mask"]
                y_pred = self.model(x, att)
                loss = self.criterion(y_pred, y)

                test_loss += (
                    loss.item() * y.shape[0]
                )  # loss scaled for the number of elements in the batch
                total += y.shape[0]
                correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

                y_true_all = np.concatenate([y_true_all, y.cpu().detach().numpy()])
                y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                y_pred_all = np.concatenate([y_pred_all, y_pred])

        # compute metrics
        metric = {
            "f1": f1_score(y_true_all, y_pred_all, average="binary") * 100,
            "recall": recall_score(y_true_all, y_pred_all, average="binary") * 100,
            "precision": precision_score(y_true_all, y_pred_all, average="binary") * 100,
        }
    
        self.__compute_report_and_log_info(prefix, writer, y_pred_all, y_true_all, metric)
        return metric

    def __compute_report_and_log_info(self, prefix, writer, y_pred_all, y_true_all, metric):
        try:
            summary = classification_report(y_true_all, y_pred_all, digits=4)
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
        if MODELS[self.transformer_name][3] == "bert":
            self.model = SentenceClassificationBERT(
                pretrained_model=self.transformer_name,
                num_class=self.num_class,
                fine_tune=self.fine_tune,
            )
        elif MODELS[self.transformer_name][3] == "roberta":
            self.model = SentenceClassificationRoBERTa(
                pretrained_model=self.transformer_name,
                num_class=self.num_class,
                fine_tune=self.fine_tune,
            )
        else:
            raise ValueError("Unknown model")

    def __setup_tokenizer(self):
        tokenizer = MODELS[self.transformer_name][1]  # (AutoModel, AutoTokenizer, 768, 'roberta'),
        self.tokenizer = tokenizer.from_pretrained(self.transformer_name)
        token_style = MODELS[self.transformer_name][
            3
        ]  # (AutoModel, AutoTokenizer, 768, 'roberta')
        self.start_token = TOKENS[token_style]["START_SEQ"]
        self.end_token = TOKENS[token_style]["END_SEQ"]
        self.pad_token = TOKENS[token_style]["PAD"]
        self.pad_idx = TOKEN_IDX[token_style]["PAD"]


    def __get_dataset(
        self,
        custom_data_path: str = None,
    ):
        """
        Fetch dataset, if data_names is not only one it will merge togheter different dataset and
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
        for (
            d
        ) in (
            data_list
        ):  # processes all the different datasets and puts them in a list of datasets
            data_split_all = self.__get_dataset_single(
                d,
            )
            data.append(data_split_all)

        if (
            len(data) > 1
        ):  # if we want to train on different datasets together it will chain them togheters.
            unified_data = {
                "train": {
                    "data": list(chain(*[d["train"]["text"] for d in data])),
                    "label": list(chain(*[d["train"]["label"] for d in data])),
                }
            }
        else:
            unified_data = data[0]
            return unified_data
    
    def __get_dataset_single(
        self,
        data_name: str = None,
    ):
        """
        Download dataset file and return dictionary including training/validation split, fucntion
        created to handle only custom dataset.
        It returns the processed dataset ready for use.

        :param data_name: data set name or path to the data
        :param label_to_id: fixed dictionary of (label: id). If given, ignore other labels
        :param lower_case: convert to lower case
        :return: formatted data, label_to_id
        """


        logging.info("data_name: {}".format(data_name))

        # for custom data
        data_path = data_name
        if not os.path.exists(data_path):
            raise ValueError("unknown dataset: %s" % data_path)
        else:
            files = glob("{}/*.csv".format(data_path))
            logging.info("formatting custom dataset from {}".format(data_path))
            files_info = {}
            for _file in files:
                _file = os.path.basename(_file)
                if _file == "train.csv":
                    files_info["train"] = _file
                elif _file in ["valid.csv", "val.csv", "validation.csv", "dev.csv"]:
                    files_info["valid"] = _file
                elif _file == "test.csv":
                    files_info["test"] = _file
            assert (
                "train" in files_info
            ), "training set not found, make sure you have `train.csv` in the folder"
            logging.info("found following files: {}".format(files_info))
            logging.info(
                "note that files should be named as either `valid.csv`, `test.csv`, or `train.csv` "
            )

        data_split_all = self.__decode_all_files(
            files_info,
            data_path,
        )

        return data_split_all

    def __decode_all_files(
        self,
        files: Dict,
        data_path: str,
    ):
        """
        Opens the valid, text and train file and builds the custom datasets.

        Returns:
        dict, list, dict: data_split (dictionary containing test, train, val datasets), unseen_entity, label_to_id
        """
        data_split = dict()
        for (
            name,
            filepath,
        ) in (
            files.items()
        ):  # creates the dictionary containing test, train, val datasets
            data_list = self.__decode_file(
                filepath,
                data_path=data_path,
            )
            data_split[name] = data_list
            logging.info(
                "dataset {0}/{1}: {2} entries".format(
                    data_path, filepath, len(data_list)
                )
            )
        return data_split

    def __decode_file(
        self,
        file_name: str,
        data_path: str,
    ):
        """
        Open a single file and loads the dataset, normalize the tags and trasform to BIO tagging if to_bio=True,

        Parameters:

        Returns:
        dict, list, dict: label_to_id, unseen_entity_label, dict containing the data and the labels {data: [..] labels: [..]}
        """
        inputs, labels, seen_entity = [], [], []
        past_mention = "O"
        df = pd.read_csv(
            os.path.join(data_path, file_name), sep="\t"
        )  # uses the \t for creating a new row thus sentence and its label will be extracted as two different rows
        _data = []
        for _, row in df.iterrows():
            _data.append({"text":row[0], "label":row[1]})
        if self.balance:
            return self.__balance_dataset(_data, self.num_class)
        else:
            return _data
        
    def __balance_dataset(data, num_class):   
        """
        Creates a new dataset that is balanced.

        Parameters:
        data (list): list of list containing [[sent, label], ..]
        num_class (int): number of target classes in the dataset

        Returns:
        list: list of list containing [[sent, label], ..]
        """
        # get count
        count = {}
        for x in data:
            label = x[1]
            if label not in count:
                count[label] = 0
            count[label] += 1

        # minimum count
        min_count = 99999999
        for _, v in count.items():
            min_count = min(min_count, v)

        # filter
        random.shuffle(data)
        new_data = []
        count_rem = [min_count] * num_class  # [min_count, min_count] label can be 0, 1
        for x in data:
            label = x[1]
            if count_rem[label] > 0:
                new_data.append(x)
            count_rem[label] -= 1

        return new_data
    
    def __release_cache(self):
        """
        Called after an epoch, used to empty the cache,
        if the used device is cuda.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

