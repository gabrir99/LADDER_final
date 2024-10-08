import hashlib
from itertools import chain
import json
import logging
import os
import random
import shutil
import requests
from glob import glob
from tokenizer_ner import Dataset, TokenizerNER
import transformers
import torch
from torch import nn
from typing import Dict, List
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from time import time
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

class TrainerNER:
    def __init__(self, config: dict):
        """
        initialize trainer
        """
        logging.info("*** initialize network ***")

        self.__initialize_internal_fields(config)
        self.__check_if_already_trained()
        self.__print_chekpoint_configuration()
        self.__set_random_seed_in_all_libraries()
        self.__initialize_default_values_remaining_fields()

    def __initialize_default_values_remaining_fields(self):
        self.dataset_split = None
        self.language = None
        self.unseen_entity_set = None
        self.optimizer = None
        self.scheduler = None
        self.scale_loss = None
        self.n_gpu = torch.cuda.device_count()
        self.device = "cuda" if self.n_gpu > 0 else "cpu"
        self.model = None
        self.__epoch = 1
        self.__step = 0
        self.label_to_id = None
        self.id_to_label = None
        self.__train_called = False

    def __set_random_seed_in_all_libraries(self):
        random.seed(self.random_seed)
        transformers.set_seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

    def __print_chekpoint_configuration(self):
        logging.info("checkpoint: {}".format(self.checkpoint_dir))
        for k, v in self.transformer_training_configurations.items():
            logging.info(" - [arg] {}: {}".format(k, str(v)))

    def __initialize_internal_fields(self, config):
        self.config = config
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./ckpt")
        self.dataset = self.config.get("dataset", "dataset/150")
        self.transformer_name = self.config.get("transformer_name", "xlm-roberta-base")
        self.random_seed = self.config.get("random_seed", 1)
        self.lr = self.config.get("lr", 5e-6)
        self.epochs = self.config.get("epochs", 20)
        self.warmup_step = self.config.get("warmup_step", 0)
        self.weight_decay = self.config.get("weight_decay", 1e-7)
        self.batch_size = self.config.get("batch_size", 32)
        self.max_seq_length = self.config.get("max_seq_length", 64)
        self.fp16 = self.config.get("fp16", False)
        self.max_grad_norm = self.config.get("max_grad_norm", 1)
        self.lower_case = self.config.get("lower_case", False)
        self.num_worker = self.config.get("num_worker", 0)
        self.cache_dir = self.config.get("cache_dir", None)

    def __check_if_already_trained(self):
        """
        - Check if the model has already been trained and uploaded, to hugginface,
        if this is the case generate a local folder where it saves all the parameters
        download from the given url.
        - If the model doesn't exist on hugginface creates a new folder and creates a file
        parameter.json containing the information of the parameters used for training the model
        """
        self.is_trained = True
        try:
            # load checkpoint on huggingface.transformers that trained with TNER
            url = "https://huggingface.co/{}/raw/main/parameter.json".format(
                self.transformer_name
            )
            self.transformer_training_configurations = json.loads(
                requests.get(url).content
            )
            logging.info(
                "load TNER finetuned checkpoint: {}".format(self.transformer_name)
            )
            self.checkpoint_dir = self.__issue_new_checkpoint(self.checkpoint_dir)
        except json.JSONDecodeError:
            if os.path.exists(self.transformer_name):
                # load local checkpoint that trained with TNER
                logging.info("load local checkpoint: {}".format(self.transformer_name))
            else:
                # new check point for finetuning
                self.is_trained = False
                logging.info("create new checkpoint")

            self.checkpoint_dir = self.__issue_new_checkpoint(self.checkpoint_dir)

    def __issue_new_checkpoint(self, checkpoint_dir):
        """
        Methods for issuing a new checkpoint, and performs the following actions:
        - checks if checkpoint_dir exists and cleans it if there are partial files
        - if parameter.json not already exists it creates it
        - if it finds the parameter.json with same parameters it exits with an error, otherwise
        it creates a new tmp.json file.
        - if tmp.json has been created it computes the md5, and creates a new subdirectory named as the md5
        and move the tmp.json file there and renames it as parameter.json

        Parameters:
        checkpoint_dir (str): name of the directory containing the checkpoints

        Returns:
        dict : dictionary containing the parameters that have been saved in the directory.
        """
        checkpoints = self.__cleanup_checkpoint_dir(checkpoint_dir)
        if len(checkpoints) == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            # create a new checkpoint
            with open("{}/parameter.json".format(checkpoint_dir), "w") as f:
                json.dump(self.config, f)

            self.transformer_training_configurations = self.config
            return checkpoint_dir
        else:
            if len(checkpoints) != 0:
                for _dir in checkpoints:
                    with open("{}/parameter.json".format(_dir), "r") as f:
                        if self.config == json.load(f):
                            exit("find same configuration at: {}".format(_dir))
            # create a new checkpoint
            with open("{}/tmp.json".format(checkpoint_dir), "w") as f:
                json.dump(self.config, f)
            _id = self.__md5("{}/tmp.json".format(checkpoint_dir))
            new_checkpoint_dir = "{}_{}".format(checkpoint_dir, _id)
            os.makedirs(new_checkpoint_dir, exist_ok=True)
            shutil.move(
                "{}/tmp.json".format(checkpoint_dir),
                "{}/parameter.json".format(new_checkpoint_dir),
            )
            self.transformer_training_configurations = self.config
            return new_checkpoint_dir

    def __md5(self, file_name):
        """get MD5 checksum"""
        hash_md5 = hashlib.md5()
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def __cleanup_checkpoint_dir(self,checkpoint_dir):
        """
        It searches inside all the subdir of checkpoint_dir if there are the following files:
        - parameter.json
        - pythorch_model.bin
        - tokenizer_config.json

        If it finds the following files return the sub-directories names, otherwise deletes all the contents
        inside the subdirectories and returns the sub-directories names.

        Parameters:
        checpoint_dir (str): directory that should contain the checpoints

        Returns:
        **str : list of sub-dir inside the checkpoint_dir
        """
        all_dir = glob("{}*".format(checkpoint_dir))
        if len(all_dir) == 0:
            return []
        for _dir in all_dir:
            if (
                os.path.exists("{}/parameter.json".format(checkpoint_dir))
                and os.path.exists("{}/pytorch_model.bin".format(checkpoint_dir))
                and os.path.exists("{}/tokenizer_config.json".format(checkpoint_dir))
            ):
                pass
            else:
                logging.info("removed incomplete checkpoint {}".format(_dir))
                shutil.rmtree(_dir)
        return glob("{}*".format(checkpoint_dir))

    def train_transformer_model(
        self,
        monitor_validation: bool = True,
        batch_size_validation: int = 1,
        max_seq_length_validation: int = 128,
    ):
        """
        Train NER model, creates all the needed data_loaders. When validation, reach an f1 score better than the
        previous saves the new parameters of the model, and tokenizer in the checkpoint_dir.

        Parameters:
        monitor_validation (bool): Display validation result at the end of each epoch
        batch_size_validation (int): Batch size for validation monitoring
        max_seq_length_validation (int): Max seq length for validation monitoring
        """
        if self.__train_called:
            raise ValueError("`train` can be called once per instant")
        if self.is_trained:
            logging.warning("finetuning model, that has been already finetuned")
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
                if end_training:
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
                self.tokenizer.tokenizer.save_pretrained(self.checkpoint_dir)
        except RuntimeError:
            logging.exception("*** RuntimeError: skip validation ***")
        return best_f1_score

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
        is_train = data_type == "train"
        features = self.tokenizer.encode_plus_all(
            tokens=self.dataset_split[data_type]["data"],
            labels=self.dataset_split[data_type]["label"],
            language=self.language,
            max_length=max_seq_length,
        )
        data_obj = Dataset(features)

        if is_train:
            assert len(data_obj) >= batch_size, (
                "training data only has {0} entries and batch size"
                "exceeded {0} < {1}, please make sure the batch size "
                "is at least less than the entire training data size.".format(
                    len(data_obj), batch_size
                )
            )
        return torch.utils.data.DataLoader(
            data_obj,
            num_workers=self.num_worker,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
        )
    
    def __release_cache(self):
        """
        Called after an epoch, used to empty the cache,
        if the used device is cuda.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

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
        if self.is_trained:
            self.__load_model_from_huggingface_already_trained()
        else:
            self.__load_model_from_huggingface_to_finetune()

        self.__create_optimizer_with_decay()

        self.__create_scheduler_using_optimizer()

        # GPU allocation
        self.model.to(self.device)

        # GPU mixture precision
        self.__reduce_memory_usage_fp16()

        # multi-gpus
        self.__parallelize_training_if_possible()

    def __parallelize_training_if_possible(self):
        if self.n_gpu > 1:
            # multi-gpu training (should be after apex fp16 initialization)
            self.model = torch.nn.DataParallel(self.model.cuda())
            logging.info("using `torch.nn.DataParallel`")
        logging.info("running on %i GPUs" % self.n_gpu)

    def __reduce_memory_usage_fp16(self):
        """
        Mixed precision training uses both 16-bit and 32-bit floating-point types to speed up training
        and reduce memory usage on NVIDIA GPUs that support it (like Volta and newer architectures). It typically involves:
            Casting certain parts of the computation to float16 (half-precision) to accelerate operations.
            Using float32 (single-precision) for parts of the computation where precision is critical.
        """
        if self.fp16:
            try:
                # Apex is a PyTorch extension that provides tools for mixed precision and distributed training
                from apex import amp  # noqa: F401

                self.model, self.optimizer = amp.initialize(
                    self.model,
                    self.optimizer,
                    opt_level="O1",
                    max_loss_scale=2**13,
                    min_loss_scale=1e-5,
                )
                self.master_params = amp.master_params
                self.scale_loss = amp.scale_loss
                logging.info("using `apex.amp`")
            except ImportError:
                logging.exception(
                    "Skip apex: please install apex from https://www.github.com/nvidia/apex to use fp16"
                )

    def __create_scheduler_using_optimizer(self):
        """
        Purpose: Learning rate schedulers are used to improve the training of neural networks by adjusting
        the learning rate over the course of training. This adjustment can help in achieving better convergence,
        avoiding steep changes in the loss landscape, and potentially finding better minima.
        Warmup: The warmup phase involves gradually increasing the learning rate from a very small value (or zero)
        to the initial learning rate set by the optimizer. This helps the model stabilize at the beginning of training.
        Constant Schedule: After the warmup phase, the scheduler keeps the learning rate constant throughout
        the remaining epochs or steps of training.
        """
        # scheduler
        self.scheduler = transformers.get_constant_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.warmup_step
        )

    def __create_optimizer_with_decay(self):
        """
        optimizer
        This list includes parameter names for which weight decay should not be applied.
        Typically, bias terms and layer normalization weights are excluded from weight decay.
        Selective Regularization: By applying weight decay selectively, the code ensures that only the
        appropriate parameters are regularized. Regularizing bias terms or layer normalization weights
        can negatively affect the training process.
        Basically L2 regularization, for avoiding overfitting of the model, penalize large weights
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=self.lr, eps=1e-8
        )

    def __load_model_from_huggingface_to_finetune(self):
        (
            self.dataset_split,
            self.label_to_id,
            self.language,
            self.unseen_entity_set,
        ) = self.__get_dataset_ner()
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}
        config = transformers.AutoConfig.from_pretrained(
            self.transformer_name,
            num_labels=len(self.label_to_id),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            cache_dir=self.cache_dir,
        )

        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.transformer_name, config=config
        )

        self.tokenizer = TokenizerNER(self.transformer_name, cache_dir=self.cache_dir)
        
    def __load_model_from_huggingface_already_trained(self):
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.transformer_name
        )  # donwload the model itself
        self.tokenizer = TokenizerNER(self.transformer_name, cache_dir=self.cache_dir)
        self.label_to_id = (
            self.model.config.label2id
        )  # map containing label --> id used from the tokenizer
        (
            self.dataset_split,
            self.label_to_id,
            self.language,
            self.unseen_entity_set,
        ) = self.__get_dataset_ner(
            fix_label_dict=True,
        )
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}

    def __epoch_train(self, data_loader, writer):
        """single epoch training: returning flag which is True if training has been completed"""
        self.model.train()
        for i, encode in enumerate(data_loader, 1):
            # update model
            encode = {k: v.to(self.device) for k, v in encode.items()}
            self.optimizer.zero_grad()
            loss = self.model(**encode, return_dict=True)["loss"]
            loss = self.__compute_loss_and_clip_grad(loss)

            # optimizer and scheduler step
            self.optimizer.step()
            self.scheduler.step()

            # log instantaneous accuracy, loss, and learning rate
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]["lr"]
            self.__log_train_epoch_step(writer, inst_loss, inst_lr)
            self.__step += 1

        if self.__epoch >= self.epochs:
            logging.info("reached maximum epochs")
            return True

        return False

    def __compute_loss_and_clip_grad(self, loss):
        if self.n_gpu > 1:
            loss = loss.mean()
        if self.fp16:
            with self.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.master_params(self.optimizer), self.max_grad_norm
                )
        else:
            loss.backward()
            """
                This function performs the actual gradient clipping. If the total norm of 
                the gradients exceeds the specified maximum value (max_grad_norm), 
                it rescales the gradients so that their norm is equal to max_grad_norm.
                """
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        return loss

    def __log_train_epoch_step(self, writer, inst_loss, inst_lr):
        if writer:
            writer.add_scalar("train/loss", inst_loss, self.__step)
            writer.add_scalar("train/learning_rate", inst_lr, self.__step)
        if self.__step % PROGRESS_INTERVAL == 0:
            logging.info(
                "[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f"
                % (self.__epoch, self.__step, inst_loss, inst_lr)
            )

    def __epoch_valid(
        self,
        data_loader,
        prefix,
        writer=None,
        unseen_entity_set: set = None,
        entity_span_prediction: bool = False,
    ):
        """single epoch validation/test"""
        # aggregate prediction and true label
        self.model.eval()
        seq_pred, seq_true = [], []
        for encode in data_loader:
            encode = {k: v.to(self.device) for k, v in encode.items()}
            labels_tensor = encode.pop("labels")
            logit = self.model(**encode, return_dict=True)["logits"]
            _true = labels_tensor.cpu().detach().int().tolist()
            _pred = (
                torch.max(logit, 2)[1].cpu().detach().int().tolist()
            )  # (batch_size, sequence_length, num_classes)
            for b in range(len(_true)):  # iterate over each batch
                _pred_list, _true_list = [], []
                self.__convert_true_ids_and_pred_ids_to_label_in_a_sentence(
                    unseen_entity_set, _true, _pred, b, _pred_list, _true_list
                )
                if len(_true_list) > 0:
                    if entity_span_prediction:
                        # ignore entity type and focus on entity position
                        _true_list = [
                            i if i == "O" else "-".join([i.split("-")[0], "entity"])
                            for i in _true_list
                        ]
                        _pred_list = [
                            i if i == "O" else "-".join([i.split("-")[0], "entity"])
                            for i in _pred_list
                        ]
                    seq_true.append(_true_list)
                    seq_pred.append(_pred_list)

        # compute metrics
        metric = {
            "f1": f1_score(seq_true, seq_pred) * 100,
            "recall": recall_score(seq_true, seq_pred) * 100,
            "precision": precision_score(seq_true, seq_pred) * 100,
        }

        self.__compute_report_and_log_info(prefix, writer, seq_pred, seq_true, metric)
        return metric

    def __compute_report_and_log_info(self, prefix, writer, seq_pred, seq_true, metric):
        try:
            summary = classification_report(seq_true, seq_pred, digits=4)
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

    def __get_dataset_ner_single(
        self,
        data_name: str = None,
        label_to_id: dict = None,
        fix_label_dict: bool = False,
        lower_case: bool = False,
        custom_language: str = "en",
        allow_new_entity: bool = True,
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
        to_bio = False
        language = "en"
        cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        data_path = os.path.join(cache_dir, data_name)
        logging.info("data_name: {}".format(data_name))

        # for custom data
        data_path = data_name
        language = custom_language
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
                elif _file in ["valid.txt", "val.txt", "validation.txt"]:
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
            to_bio=to_bio,
            allow_new_entity=allow_new_entity,
        )

        if lower_case:
            logging.info("convert into lower cased")
            data_split_all = {
                k: {
                    "data": [[ii.lower() for ii in i] for i in v["data"]],
                    "label": v["label"],
                }
                for k, v in data_split_all.items()
            }
        return data_split_all, label_to_id, language, unseen_entity_set

    def __decode_all_files(
        self,
        files: Dict,
        data_path: str,
        label_to_id: Dict,
        fix_label_dict: bool,
        entity_first: bool = False,
        to_bio: bool = False,
        allow_new_entity: bool = False,
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
                to_bio=to_bio,
                allow_new_entity=allow_new_entity,
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
        to_bio: bool = False,
        allow_new_entity: bool = False,
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
                    if word in STOPWORDS:
                        continue
                    sentence.append(word)

                    # convert tag into unified label set
                    if tag != "O":  # map tag by custom dictionary
                        location = tag.split("-")[0]
                        mention = "-".join(tag.split("-")[1:])
                        if to_bio and mention == past_mention:
                            location = "I"
                        elif to_bio:
                            location = "B"

                        fixed_mention = [
                            k for k, v in SHARED_NER_LABEL.items() if mention in v
                        ]  # normalize different tags to same tag_name
                        if len(fixed_mention) == 0 and allow_new_entity:
                            tag = "-".join([location, mention])
                        elif len(fixed_mention) == 0:
                            tag = "O"
                        else:
                            tag = "-".join([location, fixed_mention[0]])
                        past_mention = mention
                    else:
                        past_mention = "O"

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

    def __get_dataset_ner(
        self,
        custom_data_path: str = None,
        custom_data_language: str = "en",
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
        languages = []
        unseen_entity_set = {}
        param = dict(
            fix_label_dict=fix_label_dict,
            lower_case=self.lower_case,
            custom_language=custom_data_language,
        )
        for (
            d
        ) in (
            data_list
        ):  # processes all the different datasets and puts them in a list of datasets
            param["label_to_id"] = self.label_to_id
            data_split_all, label_to_id, language, ues = self.__get_dataset_ner_single(
                d, **param
            )
            data.append(data_split_all)
            languages.append(language)
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
        freq = list(
            map(
                lambda x: (x, len(list(filter(lambda y: y == x, languages)))),
                set(languages),
            )
        )
        language = sorted(freq, key=lambda x: x[1], reverse=True)[0][0]
        return unified_data, label_to_id, language, unseen_entity_set


PROGRESS_INTERVAL = 100
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
STOPWORDS = ["None", "#"]
PANX = [
    "ace",
    "bg",
    "da",
    "fur",
    "ilo",
    "lij",
    "mzn",
    "qu",
    "su",
    "vi",
    "af",
    "bh",
    "de",
    "fy",
    "io",
    "lmo",
    "nap",
    "rm",
    "sv",
    "vls",
    "als",
    "bn",
    "diq",
    "ga",
    "is",
    "ln",
    "nds",
    "ro",
    "sw",
    "vo",
    "am",
    "bo",
    "dv",
    "gan",
    "it",
    "lt",
    "ne",
    "ru",
    "szl",
    "wa",
    "an",
    "br",
    "el",
    "gd",
    "ja",
    "lv",
    "nl",
    "rw",
    "ta",
    "war",
    "ang",
    "bs",
    "eml",
    "gl",
    "jbo",
    "map-bms",
    "nn",
    "sa",
    "te",
    "wuu",
    "ar",
    "ca",
    "en",
    "gn",
    "jv",
    "mg",
    "no",
    "sah",
    "tg",
    "xmf",
    "arc",
    "cbk-zam",
    "eo",
    "gu",
    "ka",
    "mhr",
    "nov",
    "scn",
    "th",
    "yi",
    "arz",
    "cdo",
    "es",
    "hak",
    "kk",
    "mi",
    "oc",
    "sco",
    "tk",
    "yo",
    "as",
    "ce",
    "et",
    "he",
    "km",
    "min",
    "or",
    "sd",
    "tl",
    "zea",
    "ast",
    "ceb",
    "eu",
    "hi",
    "kn",
    "mk",
    "os",
    "sh",
    "tr",
    "zh-classical",
    "ay",
    "ckb",
    "ext",
    "hr",
    "ko",
    "ml",
    "pa",
    "si",
    "tt",
    "zh-min-nan",
    "az",
    "co",
    "fa",
    "hsb",
    "ksh",
    "mn",
    "pdc",
    "simple",
    "ug",
    "zh-yue",
    "ba",
    "crh",
    "fi",
    "hu",
    "ku",
    "mr",
    "pl",
    "sk",
    "uk",
    "zh",
    "bar",
    "cs",
    "fiu-vro",
    "hy",
    "ky",
    "ms",
    "pms",
    "sl",
    "ur",
    "bat-smg",
    "csb",
    "fo",
    "ia",
    "la",
    "mt",
    "pnb",
    "so",
    "uz",
    "be-x-old",
    "cv",
    "fr",
    "id",
    "lb",
    "mwl",
    "ps",
    "sq",
    "vec",
    "be",
    "cy",
    "frr",
    "ig",
    "li",
    "my",
    "pt",
    "sr",
    "vep",
]
VALID_DATASET = [
    "conll2003",
    "wnut2017",
    "ontonotes5",
    "mit_movie_trivia",
    "mit_restaurant",
    "fin",
    "bionlp2004",
    "bc5cdr",
] + [
    "panx_dataset_{}".format(i) for i in PANX
]  # 'wiki_ja', 'wiki_news_ja'
CACHE_DIR = "{}/.cache/tner".format(os.path.expanduser("~"))

# Shared label set across different dataset
SHARED_NER_LABEL = {
    "location": ["LOCATION", "LOC", "location", "Location"],
    "organization": ["ORGANIZATION", "ORG", "organization"],
    "person": ["PERSON", "PSN", "person", "PER"],
    "date": ["DATE", "DAT", "YEAR", "Year"],
    "time": ["TIME", "TIM", "Hours"],
    "artifact": ["ARTIFACT", "ART", "artifact"],
    "percent": ["PERCENT", "PNT"],
    "other": ["OTHER", "MISC"],
    "money": ["MONEY", "MNY", "Price"],
    "corporation": ["corporation"],  # Wnut 17
    "group": ["group", "NORP"],
    "product": ["product", "PRODUCT"],
    "rating": ["Rating", "RATING"],  # restaurant review
    "amenity": ["Amenity"],
    "restaurant": ["Restaurant_Name"],
    "dish": ["Dish"],
    "cuisine": ["Cuisine"],
    "actor": ["ACTOR", "Actor"],  # movie review
    "title": ["TITLE"],
    "genre": ["GENRE", "Genre"],
    "director": ["DIRECTOR", "Director"],
    "song": ["SONG"],
    "plot": ["PLOT", "Plot"],
    "review": ["REVIEW"],
    "character": ["CHARACTER"],
    "ratings average": ["RATINGS_AVERAGE"],
    "trailer": ["TRAILER"],
    "opinion": ["Opinion"],
    "award": ["Award"],
    "origin": ["Origin"],
    "soundtrack": ["Soundtrack"],
    "relationship": ["Relationship"],
    "character name": ["Character_Name"],
    "quote": ["Quote"],
    "cardinal number": ["CARDINAL"],  # OntoNote 5
    "ordinal number": ["ORDINAL"],
    "quantity": ["QUANTITY"],
    "law": ["LAW"],
    "geopolitical area": ["GPE"],
    "work of art": ["WORK_OF_ART", "creative-work"],
    "facility": ["FAC"],
    "language": ["LANGUAGE"],
    "event": ["EVENT"],
    "dna": ["DNA"],  # bionlp2004
    "protein": ["protein"],
    "cell type": ["cell_type"],
    "cell line": ["cell_line"],
    "rna": ["RNA"],
    "chemical": ["Chemical"],  # bc5cdr
    "disease": ["Disease"],
}
