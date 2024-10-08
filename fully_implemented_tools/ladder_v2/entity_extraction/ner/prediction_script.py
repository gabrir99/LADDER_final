import argparse
import json
import logging
import os
import nltk
from ner_transformer import TransformerNER
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def extract_entities_from_txt(text, ner_model: TransformerNER):
    logging.info("Starting extraction")
    sents = extract_sentences(text)
    entities_list = ner_model.get_entities(sents, max_seq_length= 64)
    return entities_list

def extract_sentences(text):
    text = remove_consec_newline(text)
    text = text.replace("\t", " ")
    text = text.replace("'", "'")
    sents_nltk = nltk.sent_tokenize(text)
    sents = []
    for x in sents_nltk:
        sents += x.split("\n")
    return sents

def remove_consec_newline(s):
    ret = s[0]
    for x in s[1:]:
        if not (x == ret[-1] and ret[-1]=='\n'):
            ret += x
    return ret

def write_found_entities_to_file(args, result):
    logging.info("Saving to file : {}".format(args.save_file))
    with open(os.path.join(args.save_file), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

def open_threat_report(args):
    with open(os.path.join(args.input_file), "r", encoding="utf-8") as f:
        text = f.read()
    return text

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on document")

    parser.add_argument(
        "--input-file", default="../../../resources/litepower.txt", type=str, help="input document path"
    )
    parser.add_argument(
        "--save-file",
        default="../../../resources/outputs/ladder/ner.json",
        type=str,
        help="path for output document",
    )
    parser.add_argument(
        "--models-dir",
        default="models/xlm-roberta-base",
        type=str,
        help="path for output document",
    )
    args = parser.parse_args()
    return args   
     
if __name__ == "__main__":
    args = parse_args()
    text = open_threat_report(args)
    model = TransformerNER(args.models_dir)
    ner_entities = extract_entities_from_txt(text, model);
    write_found_entities_to_file(args, ner_entities)