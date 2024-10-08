import argparse
import json
import logging
import os
import sys
import nltk
from sentence_transformers import SentenceTransformer
# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fully_implemented_tools.common.stix.stix_generator import StixGenerator
from ladder_v2.entity_extraction.entity_extraction_step import extract_entities_from_txt
from ladder_v2.entity_extraction.ner.ner_transformer import TransformerNER
from ladder_v2.entity_extraction.reg.ioc_extractor import IOCExtractor
from ladder_v2.relation_extraction.relation_extraction_model import RelationExtractionModel
from ladder_v2.ttpclassifier.attack_pattern_identification.entity_recognition_model import EntityRecognitionModel
from ladder_v2.ttpclassifier.mapping_to_framework.map_attack_pattern import AttackPatternFrameworkMapping
from ladder_v2.ttpclassifier.sentence_classification.sentence_classification_model import SentenceClassificationModel
from ladder_v2.ttpclassifier.ttp_classifier_step import extract_attack_patterns_from_classified_sent, extract_classified_sentences_from_txt, map_extracted_attack_patterns


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


def write_found_entities_to_file(save_path, result):
    logging.info("Saving to file : {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)

def open_threat_report(args):
    with open(os.path.join(args.input_file), "r", encoding="utf-8") as f:
        text = f.read()
    return text

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on document")

    parser.add_argument(
        "--input-file", default="../resources/litepower.txt", type=str, help="input document path"
    )
    parser.add_argument(
        "--save-dir",
        default="../resources/outputs/ladder",
        type=str,
        help="path for output document",
    )
    parser.add_argument(
        "--models-entity-extr-dir",
        default="entity_extraction/ner/models/xlm-roberta-base",
        type=str,
        help="path to ner model",
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
        default="ttpclassifier/sentence_classification/models/bert-base-uncased",
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
        default="ttpclassifier/attack_pattern_identification/models/roberta-base",
        type=str,
        help="path to model weights",
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
        default="relation_extraction/models/bert-base-uncased",
        type=str,
        help="path to model weights",
    )
    args = parser.parse_args()
    return args   
     
if __name__ == "__main__":
    args = parse_args()
    text = open_threat_report(args)

    ### Entity Extraction
    model_ner = TransformerNER(args.models_entity_extr_dir)
    model_regex = IOCExtractor()
    all_entities = extract_entities_from_txt(text, model_ner, model_regex);
    write_found_entities_to_file(os.path.join(args.save_dir, "all_entities.json"), all_entities)


    ### Attack Pattern Extraction
    sent_classification = SentenceClassificationModel(args.sequence_length_sentence, args.sentence_classification_model, args.models_sent_clas_dir )
    attack_pattern_identification = EntityRecognitionModel(args.sequence_length_sentence, args.attack_pattern_identification_model, args.models_atk_ident_dir)
    attack_pattern_mapper = AttackPatternFrameworkMapping(
        SentenceTransformer("all-MiniLM-L6-v2"),
        "MITRE",
        "ttpclassifier/mapping_to_framework/enterprise-techniques.csv"
    )
    classified_sentences = extract_classified_sentences_from_txt(text, sent_classification);
    write_found_entities_to_file(os.path.join(args.save_dir, "productive_sents.json"), classified_sentences)

    attack_patterns = extract_attack_patterns_from_classified_sent(classified_sentences, attack_pattern_identification);
    write_found_entities_to_file(os.path.join(args.save_dir, "attack_patterns.json"), attack_patterns)

    mapped_attack_pattern = map_extracted_attack_patterns(attack_patterns, attack_pattern_mapper)
    write_found_entities_to_file(os.path.join(args.save_dir, "mapped_attack_patterns.json"), mapped_attack_pattern)

    ### Relation Extraction
    relation_extraction_model = RelationExtractionModel(args.sequence_length, args.rel_extr_model, args.models_rel_extr_dir, 11, "relation_extraction/relations.csv")
    
    annotated_sent_list = relation_extraction_model.annotate_sentences_based_on_found_entites(all_entities)
    write_found_entities_to_file(os.path.join(args.save_dir, "annotated_sent_for_rel_extr.json"), annotated_sent_list)
    
    extracted_rel = relation_extraction_model.get_relations_from_sentences(annotated_sent_list)
    write_found_entities_to_file(os.path.join(args.save_dir, "extracted_stix_rel.json"), extracted_rel)

    stix_generator = StixGenerator()
    bundle = stix_generator.generate_stix_report(all_entities, mapped_attack_pattern, extracted_rel)
    logging.info(bundle)
    bundle_json = json.loads(bundle.serialize(pretty=False))

    write_found_entities_to_file(os.path.join(args.save_dir, "stix_bundle.json"), bundle_json)

