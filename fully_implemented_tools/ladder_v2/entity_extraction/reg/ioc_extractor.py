"""
Author: Antony Shenouda
"""

import logging
import os
import re
import configparser as ConfigParser
import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
# Import common modules
from common.ner.entity import *
from common.ner.entity_extraction import *
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
class IOCExtractor(EntityExtraction):
    def __init__(self):
        """
        initialize entity extraction model
        """
        self.pat = self.__load_patterns("data/patterns.ini")
    

    def get_entities(self, sentences : list):
        """

        Parameters:
        sentences (list): list of sentences from wich we want to extract entities
        
        Returns:
        list: list of dictionary where each consists of {"entity": _entities, "sentence": sentence}
            where entities is a list of dictionary containing
                'type': (str) entity type
                'position': (list) start position and end position
                'mention': (str) mention
        """
        entities = []
        for sent in sentences:
            ioc_in_sent = []
            founded_matches_list= self.__find_all_matched_regex_in_sentence(sent)
            for i in range(len(founded_matches_list)):
                list_for_ioc_type = founded_matches_list[i]["value"]
                for ioc in list_for_ioc_type:
                    matches = re.finditer(ioc, sent)
                    for match in matches:
                        
                        if not self.__ioc_already_found(match, ioc_in_sent, ioc):
                            result = {
                                "type": founded_matches_list[i]["type"],
                                "position": [match.start(), match.end()-1],
                                "mention": ioc,
                                "probability": 1,
                            }
                            ioc_in_sent.append(result)
            
            entities.append({"entity": ioc_in_sent, "sentence": sent})
        return entities

    def __ioc_already_found(self, match, ioc_in_sent, ioc):
        for ent in ioc_in_sent:
            if ent["mention"] == ioc and ent["position"][0] == match.start():
                return True
        return False

    def __find_all_matched_regex_in_sentence(self, sent):
        """
        Finds all matches for a set of regular expression patterns in a given string.

        Parameters:
        self (str): The input string to search for matches.

        Returns:
        list: A list of lists, where each inner list contains matches for a specific pattern.
        """
        lst = []
        for key, value in self.pat.items():
            found = re.findall(value, sent)
            if found:
                lst.append({"type": key, "value": found})
        return lst

    def __load_patterns(self, path):
        patterns = {}
        module_dir = os.path.dirname(__file__)
        # path = "./data/patterns.ini"
        full_path = os.path.join(module_dir, path)

        config = ConfigParser.ConfigParser()
        with open(full_path) as f:
            config.readfp(f)
        for ind_type in config.sections():
            try:
                ind_pattern = config.get(ind_type, "pattern")
            except:
                continue
            if ind_pattern:
                ind_regex = re.compile(ind_pattern, re.IGNORECASE | re.M)
                patterns[ind_type] = ind_regex
        return patterns