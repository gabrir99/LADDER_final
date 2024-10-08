import argparse
import json
import logging
import os
import sys


# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ladder_v2.ttpclassifier.mapping_to_framework.map_attack_pattern import AttackPatternFrameworkMapping
from ladder_v2.ttpclassifier.attack_pattern_identification.entity_recognition_model import EntityRecognitionModel
from ladder_v2.ttpclassifier.sentence_classification.sentence_classification_model import SentenceClassificationModel
import nltk
logging.getLogger("sentence_transformers").setLevel(logging.WARNING) #added to remove tqdm processing bar when computing embeddings
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


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


def extract_classified_sentences_from_txt(text, classification_model: SentenceClassificationModel):
    logging.info("Starting extraction")
    sents = extract_sentences(text)
    entities_list = classification_model.get_classified_sentences(sents)
    return entities_list

def extract_attack_patterns_from_classified_sent(classified_sentences, entity_recognition_model: EntityRecognitionModel):
    logging.info("Starting extraction")
    entities_list = entity_recognition_model.get_attack_patterns_from_already_classified_sentences(classified_sentences)
    return entities_list

def map_extracted_attack_patterns(attack_patterns, attack_pattern_mapper: AttackPatternFrameworkMapping):
    logging.info("Starting extraction of attack_patterns")
    sent_obj, mapped = attack_pattern_mapper.get_mapped_attack_patterns(attack_patterns)
    return sent_obj

def write_found_entities_to_file(save_path, result):
    logging.info("Saving to file : {}".format(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

def open_threat_report(args):
    with open(os.path.join(args.input_file), "r", encoding="utf-8") as f:
        text = f.read()
    return text

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on document")

    parser.add_argument(
        "--input-file", default="../../resources/litepower.txt", type=str, help="input document path"
    )
    parser.add_argument(
        "--save-dir",
        default="../../resources/outputs/ladder",
        type=str,
        help="path for output document",
    )
    parser.add_argument(
        "--sentence-classification-model",
        default="bert-base-uncased",
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
        "--models-sent-clas-dir",
        default="sentence_classification/models/bert-base-uncased",
        type=str,
        help="path to model weights",
    )
    parser.add_argument(
        "--attack-pattern-identification-model",
        default="roberta-base",
        type=str,
        help="name of the model",
    )
    parser.add_argument(
        "--models-atk-ident-dir",
        default="attack_pattern_identification/models/roberta-base",
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
     #using this model instead of sentence-transformers/all-mpnet-base-v2, because of a bug.
    sent_classification = SentenceClassificationModel(args.sequence_length_sentence, args.sentence_classification_model, args.models_sent_clas_dir )
    attack_pattern_identification = EntityRecognitionModel(args.sequence_length_sentence, args.attack_pattern_identification_model, args.models_atk_ident_dir)
    attack_pattern_mapper = AttackPatternFrameworkMapping(
        SentenceTransformer("all-MiniLM-L6-v2"),
        "MITRE",
        "mapping_to_framework\\enterprise-techniques.csv"
    )

    classified_sentences = extract_classified_sentences_from_txt(text, sent_classification);
    write_found_entities_to_file(os.path.join(args.save_dir, "productive_sents.json"), classified_sentences)

    attack_patterns = extract_attack_patterns_from_classified_sent(classified_sentences, attack_pattern_identification);
    write_found_entities_to_file(os.path.join(args.save_dir, "attack_patterns.json"), attack_patterns)

    mapped_attack_pattern = map_extracted_attack_patterns(attack_patterns, attack_pattern_mapper)
    write_found_entities_to_file(os.path.join(args.save_dir, "mapped_attack_patterns.json"), mapped_attack_pattern)
