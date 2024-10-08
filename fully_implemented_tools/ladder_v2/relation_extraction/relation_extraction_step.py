import argparse
import json
import logging
import os
import sys


# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ladder_v2.relation_extraction.relation_extraction_model import RelationExtractionModel
from ladder_v2.entity_extraction.ner.ner_transformer import TransformerNER
from ladder_v2.entity_extraction.reg.ioc_extractor import IOCExtractor
import nltk

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def extract_entities_from_txt(text, ner_model: TransformerNER, regex_model: IOCExtractor):
    logging.info("Starting extraction")
    sents = extract_sentences(text)
    entities_list = ner_model.get_entities(sents, max_seq_length= 64)
    ioc_list = regex_model.get_entities(sents)

    final_list = merge_ner_and_regex_entities(entities_list, ioc_list)

    return final_list

def merge_ner_and_regex_entities(entities_list, ioc_list):
    final_list = []
    assert len(entities_list) == len(ioc_list), "something went wrong"
    for i in range(len(entities_list)):
        entity = {
            "entity": entities_list[i]["entity"]+ ioc_list[i]["entity"],
            "sentence": entities_list[i]["sentence"]
        }
        final_list.append(entity)
    return final_list

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
        "--models-ner-dir",
        default="../entity_extraction/ner/models/xlm-roberta-base",
        type=str,
        help="path for output document",
    )
    parser.add_argument(
        "--rel-extr-model",
        default="bert-base-uncased",
        type=str,
        help="name of the model",
    )
    parser.add_argument(
        "--sequence-length",
        default=512,
        type=int,
        help="sequence length for model",
    )
    parser.add_argument(
        "--models-rel-extr-dir",
        default="models/bert-base-uncased",
        type=str,
        help="path to model weights",
    )
    args = parser.parse_args()
    return args   
     
if __name__ == "__main__":
    args = parse_args()
    text = open_threat_report(args)
    model_ner = TransformerNER(args.models_ner_dir)
    model_regex = IOCExtractor()
    all_entities = extract_entities_from_txt(text, model_ner, model_regex);
    write_found_entities_to_file(os.path.join(args.save_dir, "all_entites.json"), all_entities)

    relation_extraction_model = RelationExtractionModel(args.sequence_length, args.rel_extr_model, args.models_rel_extr_dir, 11, "relations.csv")
    
    annotated_sent_list = relation_extraction_model.annotate_sentences_based_on_found_entites(all_entities)
    write_found_entities_to_file(os.path.join(args.save_dir, "annotated_sent_for_rel_extr.json"), annotated_sent_list)
    
    extracted_rel = relation_extraction_model.get_relations_from_sentences(annotated_sent_list)
    write_found_entities_to_file(os.path.join(args.save_dir, "extracted_stix_rel.json"), extracted_rel)
