import logging
import pandas as pd
from scipy import spatial
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AttackPatternFrameworkMapping():
    def __init__(self, sentence_transformer_model, framework_name: str, ttp_file: str):
        """
        initialize attack pattern framework loading files related 
        to this framework
        """
        self.model = sentence_transformer_model
        self.framework_name = framework_name
        self.ttp_file = ttp_file
        self.attack_pattern_dict = {} 
        self.technique_mapping = {}
        self.embedding_cache = {}

        self.__extract_mitre_ttps_from_csv(ttp_file)


    def get_mapped_attack_patterns(self, sent_obj: list, th=0.6):
        """

        Parameters:
        sent_obj (list): list of obj of type {"sent" : sentence of the text, "attack_patterns": [..]}
        where attack_patterns is a list of previously extracted part of sentence that could possible be mapped to an attack pattern
        
        Returns:
        new_sent_obj (list): list enriched with the new entry for each object "mapped_attack_patterns" containing,
        the mappings to the chosen framework.
        mapped (dict): containing a unique list of all the ttp found in the text with minimum found distance and the line
        from which they have been extrapolated.
        """
        mapped = dict()
        for obj in sent_obj:
            if len(obj["attack_patterns"]) > 0:
                for s in obj["attack_patterns"]:
                    _id, dist = self.__get_mitre_id(s)
                    if dist < th:
                        mapped_ttp = {
                            "report_text": s,
                            "ttp_id": _id,
                            "ttp_name": self.attack_pattern_dict[_id][0][0],
                            "ttp_descr": self.attack_pattern_dict[_id][0][1],
                            "sent_emb_dist": dist
                        }
                        obj["mapped_attack_patterns"] = obj.setdefault("mapped_attack_patterns", [])
                        # Now you can safely append to the list
                        obj["mapped_attack_patterns"].append(mapped_ttp)

                        if _id not in mapped:
                            # Assigning a tuple (dist, s) to mapped[_id]
                            mapped[_id] = dist, self
                        else:
                            if dist < mapped[_id][0]:
                                mapped[_id] = dist, s
        return sent_obj, mapped
    
    def __get_mitre_id(self, text):
        min_dist = 25
        ret = None
        for k, tech_list in self.attack_pattern_dict.items():
            for v in tech_list:
                # v[0] -> attack pattern title, v[1] -> description
                d = (0.5*self.__get_embedding_distance(text, v[0]) + 0.5*self.__get_embedding_distance(text, v[1]))
                if d < min_dist:
                    min_dist = d
                    ret = k
        return ret, min_dist

    def __get_embedding(self, txt):
        if txt in self.embedding_cache:
            return self.embedding_cache[txt]
        emb = self.model.encode([txt])[0]
        self.embedding_cache[txt] = emb
        return emb

    def __get_embedding_distance(self, txt1, txt2):
        p1 = self.__get_embedding(txt1)
        p2 = self.__get_embedding(txt2)
        score = spatial.distance.cosine(p1, p2)
        return score  
     
    def __extract_mitre_ttps_from_csv(self, ttp_file="enterprise-techniques.csv"):
        logging.info("Starting extract_mitre_ttps_from_csv method")
        df = pd.read_csv(ttp_file)

        prev_id = None
        
        for _, row in df.iterrows():
            _id = row['ID']
            if not pd.isnull(_id):
                self.attack_pattern_dict[_id] = [[row['Name'], row['Description']]]
                prev_id = _id
                self.technique_mapping[row['Name']] = _id
            else:
                self.attack_pattern_dict[prev_id].append([row['Name'], row['Description']])
                self.technique_mapping[row['Name']] = prev_id

        logging.info("Ending extract_mitre_ttps_from_csv method")

