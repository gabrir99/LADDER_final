import os
import subprocess

SUMMARIZER_DIR = os.path.abspath(os.getcwd())  # current home dir
folder_path = os.path.join(SUMMARIZER_DIR, "data")

BERT_DIR = os.path.join(SUMMARIZER_DIR, "bert")
BERT_CLASFFIER = os.path.join(BERT_DIR, "run_classifier.py")
BERT_CONFIG_DIR = os.path.join(BERT_DIR, "uncased_L-12_H-768_A-12")

VOCAB_PATH = os.path.join(BERT_CONFIG_DIR, "vocab.txt")
CONFIG_PATH = os.path.join(BERT_CONFIG_DIR, "bert_config.json")


filename = "file.tsv"
_dir = "{0}".format(filename[:-4])
if not os.path.exists(_dir):
    os.makedirs(_dir)
cmdCommand = r"python {0} --task_name=mrpc --do_predict=true --data_dir={1} --vocab_file={2} --bert_config_file={3} --init_checkpoint=/Users/admin/bert/kia_output/model.ckpt-468 --max_seq_length=128 --output_dir=.\\{4}\\".format(
    BERT_CLASFFIER, folder_path, VOCAB_PATH, CONFIG_PATH, _dir
)
process = subprocess.Popen(cmdCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print(output)
