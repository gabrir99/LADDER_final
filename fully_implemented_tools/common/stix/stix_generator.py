import logging
from stix2 import ThreatActor,AttackPattern, Malware, Bundle, Identity, Tool, Indicator, Relationship
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class StixGenerator():
    def __init__(self):
        super().__init__()
    
    def generate_stix_report_ttpdrill(self, all_entities, mapped_attack_patterns, extracted_rel):
        """
        Generate the stix report

        Parameters:
        - all_entities (list): list of dictionary where each consists of {"entity": _entities, "sentence": sentence}
            where entities is a list of dictionary containing
                'type': (str) entity type
                'position': (list) start position and end position
                'mention': (str) mention
        - mapped_attack_patterns (list): containing a list of all the ttp found in the text with minimum found distance and the line
        from which they have been extrapolated.
        - extracted_rel (list): it adds for each dictionary in the list a new key "found_relations", that
        contains for each sentence information about the extracted relations in the format
                "annotated_sent":
                "found_rel":
                "e1":
                "e2":
        """

         #TODO CAPISCO CHE RELAZIONE GENERARE rivedo paper
        most_freq_malware_stix_obj = None
        most_frequent_malware_name = self.__find_most_frequent_executable(all_entities)
        unique_list_of_entities = self.__extract_unique_set_of_entities(all_entities)
        unique_list_of_attack_patterns = self.__extract_unique_set_of_attack_patterns_ttpdrill(mapped_attack_patterns)
        
        stix_obj_list = []
        for ent_obj in unique_list_of_entities:
            if ent_obj["mention"].lower() == most_frequent_malware_name:
                ent_obj["type"] = "Malware"
                ent_obj["descr"] = [att["ttp_name"] for att in unique_list_of_attack_patterns]

            stix_obj = self.__create_stix_object(ent_obj["mention"],ent_obj)
            if stix_obj is not None:
                stix_obj_list.append(stix_obj)
                if most_frequent_malware_name == ent_obj["mention"].lower():
                    most_freq_malware_stix_obj = stix_obj


        stix_attack_pattern_list = []
        for attack_obj in unique_list_of_attack_patterns:
            attack_stix = self.__create_stix_attack_pattern(attack_obj)
            stix_attack_pattern_list.append(attack_stix)

        stix_relations = []
        for stix_attack in stix_attack_pattern_list:
            relationship = Relationship(most_freq_malware_stix_obj, "using", stix_attack)
            stix_relations.append(relationship)

        stix_obj_list += stix_relations
        stix_obj_list += stix_attack_pattern_list
        bundle = Bundle(objects=stix_obj_list)
        return bundle
    

    def generate_stix_report(self, all_entities, mapped_attack_patterns, extracted_rel):
        """
        Generate the stix report

        Parameters:
        - all_entities (list): list of dictionary where each consists of {"entity": _entities, "sentence": sentence}
            where entities is a list of dictionary containing
                'type': (str) entity type
                'position': (list) start position and end position
                'mention': (str) mention
        - mapped_attack_patterns (list): containing a list of all the ttp found in the text with minimum found distance and the line
        from which they have been extrapolated.
        - extracted_rel (list): it adds for each dictionary in the list a new key "found_relations", that
        contains for each sentence information about the extracted relations in the format
                "annotated_sent":
                "found_rel":
                "e1":
                "e2":
        """


        most_freq_malware_stix_obj = None
        most_frequent_malware_name = self.__find_most_frequent_malware(all_entities)
        unique_list_of_entities = self.__extract_unique_set_of_entities(all_entities)
        unique_list_of_relations = self.__extract_unique_set_of_relations(extracted_rel)
        unique_list_of_attack_patterns = self.__extract_unique_set_of_attack_patterns(mapped_attack_patterns)
        
        stix_obj_list = []
        for ent_obj in unique_list_of_entities:
            stix_obj = self.__create_stix_object(ent_obj["mention"],ent_obj)
            if stix_obj is not None:
                stix_obj_list.append(stix_obj)
                if most_frequent_malware_name == ent_obj["mention"].lower():
                    most_freq_malware_stix_obj = stix_obj
                for rel in unique_list_of_relations:
                    if rel['e1']['mention'] == ent_obj["mention"].lower():
                        rel['stix_e1'] = stix_obj
                    elif rel['e2']['mention'] == ent_obj["mention"].lower():
                        rel['stix_e2'] = stix_obj

        stix_attack_pattern_list = []
        for attack_obj in unique_list_of_attack_patterns:
            attack_stix = self.__create_stix_attack_pattern(attack_obj)
            stix_attack_pattern_list.append(attack_stix)

        stix_relations = []
        for rel in unique_list_of_relations:
            if rel.get('stix_e1', None) and rel.get('stix_e2', None):
                relationship = Relationship(rel['stix_e1'], rel['found_rel'], rel['stix_e2'])
                stix_relations.append(relationship)
            else:
                logging.error("[ERRORE]: ", rel)

        for stix_attack in stix_attack_pattern_list:
            relationship = Relationship(most_freq_malware_stix_obj, "using", stix_attack)
            stix_relations.append(relationship)

        stix_obj_list += stix_relations
        stix_obj_list += stix_attack_pattern_list
        bundle = Bundle(objects=stix_obj_list)
        return bundle

    def __extract_unique_set_of_attack_patterns(self, mapped_attack_patterns):
        unique_list = []
        attack_pattern_set = set()
        for sent_obj in mapped_attack_patterns:
            found_attack_patterns = sent_obj.get("mapped_attack_patterns", [])
            for attack_pattern in found_attack_patterns:
                key = attack_pattern["ttp_id"]
                if key not in attack_pattern_set:
                    attack_pattern_set.add(key)
                    unique_list.append(attack_pattern)
        return unique_list
    
    def __extract_unique_set_of_attack_patterns_ttpdrill(self, mapped_attack_patterns):
        unique_list = []
        attack_pattern_set = set()
        for attack_pattern in mapped_attack_patterns:
            key = attack_pattern["ttp_id"]
            if key not in attack_pattern_set:
                attack_pattern_set.add(key)
                unique_list.append(attack_pattern)
        return unique_list
    
    def __extract_unique_set_of_entities(self, all_entities):
        unique_list = []
        entity_dict = dict()
        for sent_obj in all_entities:
            entity_list_per_sent = sent_obj["entity"]
            for ent_obj in entity_list_per_sent:
                name = ent_obj["mention"].lower()
                type = ent_obj["type"].lower()

                ent_obj["descr"] = sent_obj["sentence"]
                entity_dict[name] = entity_dict.get(name, {})
                entity_dict[name][type] = entity_dict[name].get(type, [])
                entity_dict[name][type].append(ent_obj)
        
        for key, dict_type in entity_dict.items():
            max_type = None
            max_len = -1
            for type, array_of_ent in dict_type.items():
                if max_len < len(array_of_ent):
                    max_len = len(array_of_ent)
                    max_type = type
            unique_list.append(dict_type[max_type][0])
                
        return unique_list

    def __extract_unique_set_of_relations(self, extracted_rel):
        unique_list = []
        relations_set = set()
        for sent_obj in extracted_rel:
            found_relations = sent_obj["found_relations"]
            for rel in found_relations:
                e1_name = rel["e1"]["mention"].lower()
                e2_name = rel["e2"]["mention"].lower()
                key = e1_name+"#"+e2_name+"#"+rel["found_rel"]
                if key not in relations_set:
                    relations_set.add(key)
                    unique_list.append(rel)
        return unique_list
    
    def __find_most_frequent_malware(self, all_entities):
        """
        Returns the name of the most frequent malware
        """
        entity_count = dict()
        for sent_obj in all_entities:
            entity_list_per_sent = sent_obj["entity"]
            for ent_obj in entity_list_per_sent:
                type = ent_obj["type"]
                name = ent_obj["mention"].lower()
                if type == "Malware":
                    entity_count[name] = entity_count.get(name, 0) + 1

        return max(entity_count, key=entity_count.get)

    def __find_most_frequent_executable(self, all_entities):
        """
        Returns the name of the most frequent malware
        """
        entity_count = dict()
        for sent_obj in all_entities:
            entity_list_per_sent = sent_obj["entity"]
            for ent_obj in entity_list_per_sent:
                type = ent_obj["type"]
                name = ent_obj["mention"].lower()
                if type == "Executable":
                    entity_count[name] = entity_count.get(name, 0) + 1

        return max(entity_count, key=entity_count.get)
    
    def __create_stix_object(self, name, entity):
        ## Entities should also eventualy contain, relationships objects, or maybe is better to first create obj and then pass it to a new function that
        ## generates the relationships.
        ## The rag system should be added inside this step.

        obj = None
        type = entity['type']
        try:
            if type == 'ThreatActor':
                obj = ThreatActor(name=name, description=entity["descr"])
            elif type == 'MalwareType' or type == 'Malware':
                obj = Malware(name=name, is_family="false", description=entity["descr"])
            elif type == 'Person' or type == 'Organization':
                obj = Identity(name=name, description=entity["descr"])
            elif type == 'OS' or type == 'Application':
                obj = Tool(name=name, description=entity["descr"])
            elif type == 'IP':   
                obj = Indicator(name="Malicious IP", description=entity["descr"], pattern_type="stix", pattern="[ipv4-addr:value = '{}']".format(name))
            elif type == 'Email':
                obj = Indicator(name="Malicious Email", description=entity["descr"], pattern_type="stix", pattern="[email-message:from_ref.value = '{}']".format(name))
            elif type == 'MD5':
                obj = Indicator(name="Malicious Hash", description=entity["descr"], pattern_type="stix", pattern="[file:hashes.'MD5' = '{}']".format(name))
            elif type == 'File':
                obj = Indicator(name="Malicious File", description=entity["descr"], pattern_type="stix", pattern="[file:name = '{}']".format(name))
            elif type == 'Registry':
                obj = Indicator(name="Suspicious Registry Entry", description=entity["descr"], pattern_type="stix", pattern="[windows-registry-key:key = '{}']".format(name.replace("\\", "\\\\")))            
            else:
                logging.info('Ignoring the following entry : {}'.format(entity["mention"]))
        except Exception as e:
            logging.info(e)
            logging.info(name+ " - "+ type)

        return obj

    def __create_stix_attack_pattern(self, attack_pattern_obj):
        obj = AttackPattern(
            name= attack_pattern_obj["ttp_name"],
            description=attack_pattern_obj["ttp_descr"],
            external_references=[
                {
                    "source_name": "mitre-attack",
                    "external_id": attack_pattern_obj["ttp_id"],
                    "url": "https://attack.mitre.org/techniques/"+attack_pattern_obj["ttp_id"]
                }
            ]
        )
        return obj