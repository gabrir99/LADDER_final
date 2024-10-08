import argparse
import json
import logging as log
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ladder_v2.ttpclassifier.attack_pattern_identification.config import TOKEN_IDX, TOKENS
from ladder_v2.ttpclassifier.attack_pattern_identification.entity_recognition_model import EntityRecognitionModel
import nltk
from ladder_v2.ttpclassifier.attack_pattern_identification.custom_model import *

log.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=log.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def extract_attack_patterns_from_txt(text, entity_recognition_model: EntityRecognitionModel):
    log.info("Starting extraction")
    sents = extract_sentences(text)
    entities_list = entity_recognition_model.get_attack_patterns_from_sentences(sents)
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

def write_found_attack_patterns_to_file(args, result):
    log.info("Saving to file : {}".format(args.save_file))
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
        default="../../../resources/outputs/ladder/attack_patterns.json",
        type=str,
        help="path for output document",
    )
    parser.add_argument(
        "--attack-pattern-identification-model",
        default="roberta-base",
        type=str,
        help="name of the model",
    )
    parser.add_argument(
        "--sequence-length-sentence",
        default=256,
        type=int,
        help="sequence length for model",
    )
    parser.add_argument(
        "--models-dir",
        default="models/roberta-base",
        type=str,
        help="path to model weights",
    )
    args = parser.parse_args()
    return args   
     
    
if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    args = parse_args()

    text = open_threat_report(args)
    model = EntityRecognitionModel(args.sequence_length_sentence, args.attack_pattern_identification_model, args.models_dir)

    attack_patterns = extract_attack_patterns_from_txt(text, model);
    write_found_attack_patterns_to_file(args, attack_patterns)