from collections import Counter
from itertools import chain
import json
import logging
import random
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ladder_v2.relation_extraction.custom_model import RelationClassificationBERT, RelationClassificationRoBERTa
from ladder_v2.relation_extraction.config import MODELS
from time import time
from typing import Dict
import transformers
import torch
from torch import nn
from glob import glob
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from sklearn.metrics import classification_report

PROGRESS_INTERVAL = 100

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

class RelationExtractionTrainer:
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
        self.epochs = self.config.get("epochs", 30)
        self.batch_size = self.config.get("batch_size", 8)
        self.max_seq_length = self.config.get("max_seq_length", 512)
        self.task_name = self.config.get("task_name", "relation_extraction")
        self.eps = self.config.get("eps",1e-8)
        self.num_class = self.config.get("num_labels",11)


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
        self.optimizer = None
        self.patience = 10
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


    def train_relation_extraction_model(
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

        self.model.module.from_pretrained_custom(self.checkpoint_dir)
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
            "train": self.__setup_loader(RandomSampler, "train", self.batch_size, self.max_seq_length)
        }
       
        if "test" in self.dataset_split.keys():
            data_loader["test"] = self.__setup_loader(SequentialSampler,
                "test", batch_size_validation, max_seq_length_validation
            )
        else:
            data_loader["test"] = None
        
        if monitor_validation and "valid" in self.dataset_split.keys():
            # data_loader["valid"] = self.__setup_loader(
            #     "valid", batch_size_validation, max_seq_length_validation
            # )
            pass
        else:
            data_loader["valid"] = data_loader["test"]

        return data_loader
    
    def __setup_loader(self, sampler, data_type: str, batch_size: int, max_seq_length: int):
        """
        Based on the model used the ids representing entity e1, e2 will change,
        - use tokenizer to encode each sentence in a pytorch tensor,
        - for each encoded sentence find ids of e1, e2 and recover the starting index of e1, e2
        - creates a TensorDataset and returns it

        Parameters:
        sentence_train (list): list of str, that represents sentences
        sentence_train_label (list): list of int, representing extracted relation from the sentence

        Returns:
        Create the data loader by trasforming the dataset in input using the self.tokenizer class

        """
        assert self.dataset_split, "run __setup_data firstly"
        if data_type not in self.dataset_split.keys():
            return None
        
        e1_id, e2_id = self.__extract_e1_e2_ids_based_on_used_model()

        input_ids = []
        attention_masks = []
        labels = []
        e1_pos = []
        e2_pos = []

        # pre-processing sentences to BERT pattern
        for i in range(len(self.dataset_split[data_type]["data"])):
            encoded_dict = self.tokenizer.encode_plus(
                self.dataset_split[data_type]["data"][i],  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_seq_length,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
                truncation=True,
            )
            try:
                # Find e1(id:2487) and e2(id:2475) position
                # nonzero() returns indices of positions where the value was not false,
                # after that [0] we take first occurence of e1, and then [1] we take the index
                pos1 = (encoded_dict["input_ids"] == e1_id).nonzero()[0][1].item()
                pos2 = (encoded_dict["input_ids"] == e2_id).nonzero()[0][1].item()
                e1_pos.append(pos1)
                e2_pos.append(pos2)
                # Add the encoded sentence to the list.
                input_ids.append(encoded_dict["input_ids"])
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict["attention_mask"])
                labels.append(self.dataset_split[data_type]["label"][i])
            except:
                pass

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attention_masks = torch.cat(attention_masks, dim=0).to(self.device)
        labels = torch.tensor(labels, device="cuda")
        e1_pos = torch.tensor(e1_pos, device="cuda")
        e2_pos = torch.tensor(e2_pos, device="cuda")
        w = torch.ones(len(e1_pos), device="cuda")

        dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, w)


        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler(dataset),
        )

    def __extract_e1_e2_ids_based_on_used_model(self):
        e1_id = -1
        e2_id = -1
        if self.transformer_name in ["bert-base-uncased", "bert-large-uncased"]:
            e1_id = 2487
            e2_id = 2475
        elif self.transformer_name in ["roberta-base", "roberta-large"]:
            e1_id = 134
            e2_id = 176
        elif self.transformer_name in ["xlm-roberta-base", "xlm-roberta-large"]:
            e1_id = 418
            e2_id = 304
        else:
            raise ValueError("Unknown Model")
        return e1_id,e2_id
    
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
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # GPU mixture precision
        #self.__reduce_memory_usage_fp16()

        # multi-gpus
        #self.__parallelize_training_if_possible()
    
    def __create_optimizer_with_decay(self):
        # optimizer
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, eps= self.eps
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
        ) = self.__get_dataset_relation_extraction(
            fix_label_dict=True,
        )
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}

    def __get_dataset_relation_extraction(
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
        param = dict()
        for (
            d
        ) in (
            data_list
        ):  # processes all the different datasets and puts them in a list of datasets
            param["label_to_id"] = self.label_to_id
            data_split_all, label_to_id = self.__get_dataset_relation_extraction_single(
                d, **param
            )
            data.append(data_split_all)
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
        return unified_data, label_to_id

    def __get_dataset_relation_extraction_single(
        self,
        data_name: str = None,
        label_to_id: dict = None,
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

        logging.info("data_name: {}".format(data_name))

        # for custom data
        data_path = data_name
        if not os.path.exists(data_path):
            raise ValueError("unknown dataset: %s" % data_path)
        else:
            files = glob("{}/*.json".format(data_path))
            logging.info("formatting custom dataset from {}".format(data_path))
            files_info = {}
            for _file in files:
                _file = os.path.basename(_file)
                if _file == "train_sentence.json":
                    files_info["train"] = files_info.setdefault("train", {})
                    # Now you can safely append to the list
                    files_info["train"]["sent"] = _file
                elif _file == "train_label_id.json":
                    files_info["train"] = files_info.setdefault("train", {})
                    # Now you can safely append to the list
                    files_info["train"]["labels"] = _file
                elif _file == "test_sentence.json":
                    files_info["test"] = files_info.setdefault("test", {})
                    # Now you can safely append to the list
                    files_info["test"]["sent"] = _file
                elif _file == "test_label_id.json":
                    files_info["test"] = files_info.setdefault("test", {})
                    # Now you can safely append to the list
                    files_info["test"]["labels"] = _file
            assert (
                "train" in files_info
            ), "training set not found, make sure you have `train.txt` in the folder"
            logging.info("found following files: {}".format(files_info))
            logging.info(
                "note that files should be named as either `valid.txt`, `test.txt`, or `train.txt` "
            )

        label_to_id = dict() if label_to_id is None else label_to_id
        data_split_all, label_to_id = self.__decode_all_files(
            files_info,
            data_path,
            label_to_id=label_to_id,
        )

    
        return data_split_all, label_to_id, 

    def __decode_all_files(
        self,
        files: Dict,
        data_path: str,
        label_to_id: Dict,
    ):
        """
        Opens the valid, text and train file and builds the custom datasets.

        Returns:
        dict, list, dict: data_split (dictionary containing test, train, val datasets), unseen_entity, label_to_id
        """
        data_split = dict()
        for (
            name,
            dict_obj,
        ) in (
            files.items()
        ):  # creates the dictionary containing test, train, val datasets
            label_to_id , data_dict = self.__decode_file(
                dict_obj,
                data_path=data_path,
                label_to_id=label_to_id,
            )
            data_split[name] = data_dict
            logging.info(
                "dataset {0}/{1}: {2} entries".format(
                    data_path, dict_obj, len(data_dict["data"])
                )
            )
        return data_split, label_to_id

    def __decode_file(
        self,
        dict_obj: dict,
        data_path: str,
        label_to_id: Dict,
    ):
        """
        Open a single file and loads the dataset, normalize the tags and trasform to BIO tagging if to_bio=True,

        Parameters:

        Returns:
        dict, list, dict: label_to_id, unseen_entity_label, dict containing the data and the labels {data: [..] labels: [..]}
        """
        sentence_val = json.load(open(os.path.join(data_path, dict_obj["sent"]), "r"))
        sentence_val_label = json.load(open(os.path.join(data_path, dict_obj["labels"]), "r"))

        assert len(sentence_val) == len(sentence_val_label)
        return label_to_id, {"data": sentence_val, "label": sentence_val_label}

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
            self.model.module.save_pretrained(self.checkpoint_dir)
            self.tokenizer.save_pretrained(self.checkpoint_dir)

    def __execute_valid_epoch_and_save_model_if_best_score(
        self, writer, data_loader, best_f1_score
    ):
        try:
            metric = self.__epoch_valid(
                data_loader["valid"], writer=writer, prefix="valid"
            )
            if metric["f1"] > best_f1_score:
                best_f1_score = metric["f1"]
                self.model.module.save_pretrained(self.checkpoint_dir)
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

        for batch in data_loader:# (input ids, attention masks, output labels ids)xbatch_size
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_e1_pos = batch[3].to(self.device)
            b_e2_pos = batch[4].to(self.device)
            b_w = batch[5].to(self.device)

            self.model.zero_grad()
           # Perform a forward pass (evaluate the model on this training batch)
            loss, logits, _ = self.model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                e1_pos=b_e1_pos,
                e2_pos=b_e2_pos,
                w=b_w,
            )

             # Accumulate the training loss over all of the batches
            train_loss += loss.sum().item()
            # Perform a backward pass to calculate the gradients.
            loss.sum().backward()
            # Clip the norm of the gradients to 1.0 to prevent exploding gradients problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient
            inst_lr = self.optimizer.param_groups[0]["lr"]
            inst_loss = loss.sum().cpu().detach()
            self.optimizer.step()
            self.__log_train_epoch_step(writer, inst_loss, inst_lr, train_loss, self.__step+1)
            self.__step += 1

        if self.__epoch >= self.epochs:
            logging.info("reached maximum epochs")
            return True

        return False
    
    def __log_train_epoch_step(self, writer, inst_loss, inst_lr, train_loss,iteration):
        avg_loss = train_loss / iteration
        if writer:
            writer.add_scalar("train/loss", avg_loss, self.__step)
            writer.add_scalar("train/learning_rate", inst_lr, self.__step)
        if self.__step % PROGRESS_INTERVAL == 0:
            logging.info(
                "[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f"
                % (self.__epoch, self.__step, avg_loss, inst_lr)
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
        # Tracking variables
        total_eval_loss = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        model_predictions = []

        with torch.no_grad():
            for batch in data_loader:# (input ids, attention masks, output labels ids)xbatch_size
                # Unpack this training batch from our dataloader.
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                b_e1_pos = batch[3].to(self.device)
                b_e2_pos = batch[4].to(self.device)
                b_w = batch[5].to(self.device)

                (loss, logits, _) = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    e1_pos=b_e1_pos,
                    e2_pos=b_e2_pos,
                    w=b_w,
                )
                
                # Accumulate the validation loss.
                total_eval_loss += loss.sum().item()
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()
                all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
                all_ground_truth = np.concatenate(
                    (all_ground_truth, labels_flat), axis=None
                )

                for ii in range(b_input_ids.shape[0]):
                    decoded = self.tokenizer.decode(b_input_ids[ii], skip_special_tokens=False)
                    model_predictions.append(
                        [decoded, int(labels_flat[ii]), int(pred_flat[ii])]
                    )


        prec_micro, recall_micro, f1_micro, summary = self.__score(all_ground_truth, all_prediction)

      
        # compute metrics
        metric = {
            "f1": f1_micro * 100,
            "recall": recall_micro * 100,
            "precision": prec_micro * 100,
        }

        self.__compute_report_and_log_info(prefix, writer, summary, metric)
        return metric
    
    # cited: https://github.com/INK-USC/DualRE/blob/master/utils/scorer.py#L26
    def __score(self, key, prediction, verbose=True, no_relation=-1):
        key = key.astype(int)
        summary = None
        prediction = prediction.astype(int)
        if self.num_class == 11:
            summary = classification_report(
                    key,
                    prediction,
                    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    target_names=[
                        "N/A",
                        "isA",
                        "targets",
                        "uses",
                        "hasAuthor",
                        "has",
                        "variantOf",
                        "hasAlias",
                        "indicates",
                        "discoveredIn",
                        "exploits",
                    ],
                    digits=4,
                )
        else:
            raise ValueError("Number of labels is not correct!")

        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]

            if gold == no_relation and guess == no_relation:
                pass
            elif gold == no_relation and guess != no_relation:
                guessed_by_relation[guess] += 1
            elif gold != no_relation and guess == no_relation:
                gold_by_relation[gold] += 1
            elif gold != no_relation and guess != no_relation:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1

        # Print the aggregate score
        if verbose:
            logging.info("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro = float(sum(correct_by_relation.values())) / float(
                sum(guessed_by_relation.values())
            )
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(
                sum(gold_by_relation.values())
            )
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        logging.info("SET NO_RELATION ID: ", no_relation)
        logging.info("Precision (micro): {:.3%}".format(prec_micro))
        logging.info("   Recall (micro): {:.3%}".format(recall_micro))
        logging.info("       F1 (micro): {:.3%}".format(f1_micro))
        return prec_micro, recall_micro, f1_micro, summary

    def __compute_report_and_log_info(self, prefix, writer, summary, metric):
        try:
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
        # Load models
        if "roberta" in self.transformer_name:
            self.model = RelationClassificationRoBERTa.from_pretrained(
                self.transformer_name,  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=self.num_class,  # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
        else:
            self.model = RelationClassificationBERT.from_pretrained(
                self.transformer_name,  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=self.num_class,  # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )

    def __setup_tokenizer(self):
        tokenizer = MODELS[self.transformer_name][1]  # (AutoModel, AutoTokenizer, 768, 'roberta'),
        self.tokenizer = tokenizer.from_pretrained(self.transformer_name)

    
    def __release_cache(self):
        """
        Called after an epoch, used to empty the cache,
        if the used device is cuda.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

