#!python
# -*- coding: utf-8 -*-
# @author: Kun


"""
Author: Kun
Date: 2021-09-16 11:21:47
LastEditTime: 2021-09-23 17:56:16
LastEditors: Kun
Description: 
FilePath: /ThreatReportExtractor/role_generator.py
"""

from nlp_extractor.subject_verb_object_extract import SubjectVerbObjectExtractor

# nlp = spacy.load("en_core_web_lg")
from allennlp.predictors.predictor import Predictor
from nltk.stem.wordnet import WordNetLemmatizer
from preprocessings import *
from data_loader.ioc_loader import iocs
from data_loader.pattern_loader import load_lists

from project_config import SEC_PATTERNS_FILE_PATH

# main_verbs = load_lists(SEC_PATTERNS_FILE_PATH)['verbs']
# main_verbs = main_verbs.replace("'", "").strip('][').split(', ')


class RoleGenerator(object):
    def __init__(self, nlp, main_verbs) -> None:
        """
        Perform the semantic role labeling during the graph generation step.

        Parameters:
        nlp (Language): Tagger, parser, and named entity recognizer.
        main_verbs (list): A list of main verbs to use during homogenization.
        """
        super(RoleGenerator, self).__init__()

        self.nlp = nlp  # self.svo_extractor.nlp

        self.svo_extractor = SubjectVerbObjectExtractor(nlp)

        self.main_verbs = main_verbs

    def colon_seprator_multiplication(self, stri):
        """
        Processes sentences containing a colon (:) by replicating the left part of the colon
        for each indicator of compromise (IOC) found on the right part.

        Parameters:
        stri (str): The input string containing sentences with potential colon-separated IOCs.

        Returns:
        str: The processed string with replicated sentences for each IOC found.
        """
        print("[colon_seprator_multiplication] stri: ", stri)
        coref_reference_list = load_lists(SEC_PATTERNS_FILE_PATH)["TFL"]
        coref_reference_list = (
            coref_reference_list.replace("'", "").strip("][").split(", ")
        )
        stri = stri.rstrip()  # removees any white spaces from the end of the string
        stri = stri.rstrip(".")  # removes any dots from the end of the string
        result = ""
        for item in sent_tokenize(stri):
            flag = False
            for refrence in coref_reference_list:
                if refrence in item and ":" in item:
                    sentence_splits = item.split(
                        ":", 1
                    )  # When set to 1, the string will be split into at most two parts.
                    y = iocs.list_of_iocs(item.split(":", 1)[1])
                    if y:
                        sentence_replicas = [sentence_splits[0].rstrip(":")] * len(y)
                        for i in range(len(sentence_replicas)):
                            result += (
                                sentence_replicas[i].replace(refrence, y[i]) + " . "
                            )
                    else:
                        item = sentence_splits[0].replace(refrence, sentence_splits[1])
                        result += item

                    flag = True
                    break
            if flag == False:
                result += item
                result += " "

        print("[colon_seprator_multiplication] result: ", result)

        if result.rstrip("")[-1] != ".":
            result += "."
        return result

    def roles(self, sentences):
        """
        Extracts semantic roles from a list of sentences using a semantic role labeling (SRL) model.

        Args:
        sentences (list): A list of strings where each string represents a sentence to analyze.

        Returns:
        list: A list of lists containing extracted semantic roles from each sentence. Each inner list
            represents semantic roles extracted from a sentence in the format ['ROLE: value', ...].
            The roles extracted include 'ARG0', 'ARG1', 'V' (verb), and others as determined by the SRL model.

        Notes:
            - Uses AllenNLP for semantic role labeling.
            - Performs additional extraction using dependency parsing (DP SVO and Naive SVO) if initial SRL predictions
            do not yield sufficient roles (less than 3 roles per sentence).
            - Lemmatizes verbs and filters out non-relevant roles based on predefined lists ('main_verbs',
            'removeable_items_list', etc.).
            - Outputs are normalized to ensure consistency in role representation across different sentences.

        Raises:
            Exception: If there's an error during sentence processing using the SRL model.

        Example:
            >>> sentences = ["He opened the door.", "They write the code quickly."]
            >>> results = roles(sentences)
            >>> print(results)
            [['ARG0: He', 'V: open', 'ARG1: the door'], ['ARG0: They', 'V: write', 'ARG1: the code', 'ARGM-MNR: quickly']]
        """
        my_svo_triplet = []
        all_nodes = []
        for i in range(len(sentences)):
            # public SRL model https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz
            predictor = Predictor.from_path("srl-model.tar.gz")
            predictions = predictor.predict(sentences[i])
            """
            predictions=
            {
            'verbs': 
                [
                    {'verb': 'connects', 
                    'description': '[ARG2: The malware] [V: connects] [ARG1: to the remote ip : * ( remote ip : * ) server] .', 
                    'tags': ['B-ARG2', 'I-ARG2', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']
                    }
                ], 
            'words': ['The', 'malware', 'connects', 'to', 'the', 'remote', 'ip', ':', '*', '(', 'remote', 'ip', ':', '*', ')', 'server', '.']
            }
            """
            lst = []
            nodes = []
            for k in predictions["verbs"]:
                if (
                    k["description"].count("[") > 1
                ):  # add all the verbs containing not generating empty array
                    lst.append(
                        k["description"]
                    )  # append string containing array with: ARG1, V, ARG2
            for jj in range(len(lst)):
                nodes.append([])
                for j in re.findall(
                    r"[^[]*\[([^]]*)\]", lst[jj]
                ):  # basically find this 3 [] [ARG2: The malware] [V: connects] [ARG1: to the remote ip : * ( remote ip : * ) server]
                    nodes[jj].append(j)
            print("*****sentence:", sentences[i], "*****nodes: ", nodes)

            # if the extracted relationship has a main verb and one of the other 2 elements is an IOCs add a new nod ARG-NEW
            for lis_ in nodes:
                for indx in range(len(lis_)):
                    if (
                        lis_[0].split(":", 1)[0].lower().strip() == "v"
                        and lis_[0].split(":", 1)[1].lower().strip() in self.main_verbs
                    ):
                        n = len(lis_)
                        for j in range(1, len(lis_)):
                            if lis_[j].split(":", 1)[0].lower() != "v":
                                if len(iocs.list_of_iocs(lis_[j].split(":", 1)[1])) > 0:
                                    lis_.insert(0, " ARG-NEW: *")

            maxlength = 0
            if nodes:
                maxlength = max((len(i) for i in nodes))
            if (
                nodes == [] or maxlength < 3
            ):  # IF USING SRL we are not able to find anything we try to use Dependency Parsing
                print("****DP SVO****")
                tokens = self.nlp(sentences[i])
                svos = self.svo_extractor.findSVOs(tokens)
                if svos:
                    for sv in range(len(svos)):
                        if len(svos[sv]) == 3:
                            print(
                                "Dependency SVO(s):",
                                [
                                    "ARG0: " + svos[sv][0],
                                    "V: " + svos[sv][1],
                                    "ARG1: " + svos[sv][2],
                                ],
                            )
                            nodes.append(
                                [
                                    "ARG0: " + svos[sv][0],
                                    "V: " + svos[sv][1],
                                    "ARG1: " + svos[sv][2],
                                ]
                            )
                print("Dependency-SVO added nodes: ", nodes)

                print("****Naive SVO****")
                breakers = []
                subj, obj = "", ""
                doc = self.nlp(sentences[i])
                for token in doc:
                    if token.pos_ == "VERB":
                        breakers.append(str(token))
                if len(breakers) != 0:
                    for vb in breakers:
                        subj = "subj: " + sentences[i].split(vb)[0]
                        obj = "obj: " + sentences[i].split(vb)[1]
                        vrb = "v: " + vb
                        lst = []
                        lst.append(subj)
                        lst.append(vrb)
                        lst.append(obj)
                        nodes.append(lst)
                print("Naive Nodes: ", nodes)

            if nodes != []:
                zero_dunplicate_removed = []
                for i in nodes:
                    zero_dunplicate_removed.append(list(dict.fromkeys(i)))
                no_zero_nodes = []
                for i in zero_dunplicate_removed:
                    if "." in i:
                        i.remove(".")
                        no_zero_nodes.append(i)
                    else:
                        no_zero_nodes = zero_dunplicate_removed

                no_zero_nodes_plus_3 = []
                for i in no_zero_nodes:
                    if len(i) > 2:
                        no_zero_nodes_plus_3.append(i)

                removeable_items_list = [
                    "both",
                    "also",
                    "that",
                    "would",
                    "could",
                    "immediately",
                    "usually",
                    "for",
                    "when",
                    "then",
                    "will",
                    "which",
                    "first",
                    "second",
                    "third",
                    "forth",
                    "fifth",
                    "internally",
                    "where",
                    "while",
                    "either",
                    "nither",
                    "when",
                    "sever",
                    "successfully",
                    "also",
                    "to",
                    "above",
                    "already",
                    "recently",
                    "may",
                    "however",
                    "can",
                    "once loaded",
                    "in fact",
                    "in this way",
                    "all",
                    "actually",
                    "inadvertently",
                    "instead",
                    "when copying themselves",
                    "automatically",
                    "should",
                    "can",
                    "could",
                    "necessarily",
                    "if found",
                    "randomly",
                    "again",
                    "still",
                    "generally",
                    "slowly",
                    "ever",
                    "shall",
                    "newly",
                    "However",
                    "when executed",
                    "subsequently",
                ]

                # lammarizer
                for i in range(len(no_zero_nodes_plus_3)):
                    for index, item in enumerate(no_zero_nodes_plus_3[i]):
                        if item.split(": ")[0] == "V":
                            word = item.split(": ")[1]
                            no_zero_nodes_plus_3[i][
                                index
                            ] = "V: " + WordNetLemmatizer().lemmatize(
                                item.split(": ")[1].lower(), "v"
                            )

                for i in range(len(no_zero_nodes_plus_3)):
                    if no_zero_nodes_plus_3[i]:
                        for index, item in enumerate(no_zero_nodes_plus_3[i]):
                            if "ARGM-MOD:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-ADV:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-TMP:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-MNR:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "R-ARG1:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "R-ARG0:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-DIS:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-PRP:" in item:
                                if (
                                    item.split(": ", 1)[1].lower()
                                    in removeable_items_list
                                ):
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                for i in range(len(no_zero_nodes_plus_3)):
                    if no_zero_nodes_plus_3[i]:
                        for index, item in enumerate(no_zero_nodes_plus_3[i]):
                            if "ARGM-MOD:" in item:
                                if item.split(": ")[1].lower() in removeable_items_list:
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-ADV:" in item:
                                if item.split(": ")[1].lower() in removeable_items_list:
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-TMP:" in item:
                                if item.split(": ")[1].lower() in removeable_items_list:
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "ARGM-MNR:" in item:
                                if item.split(": ")[1].lower() in removeable_items_list:
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "R-ARG1:" in item:
                                if item.split(": ")[1].lower() in removeable_items_list:
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                            if "R-ARG0:" in item:
                                if item.split(": ")[1].lower() in removeable_items_list:
                                    del no_zero_nodes_plus_3[i][index]
                                else:
                                    print("##### NEW Exception: ", item)

                v_unlink = [
                    "delete",
                    "clear",
                    "remove",
                    "erase",
                    "wipe",
                    "purge",
                    "expunge",
                ]
                v_write = [
                    "entrench",
                    "exfiltrate",
                    "store",
                    "drop",
                    "drops",
                    "install",
                    "place",
                    "deploy",
                    "implant",
                    "write",
                    "putfile",
                    "compose",
                    "create",
                    "creates",
                    "copy",
                    "copies",
                    "save",
                    "saved",
                    "saves",
                    "add",
                    "adds",
                    "modify",
                    "modifies",
                    "append",
                    "appends",
                ]
                v_read = [
                    "survey",
                    "download",
                    "navigate",
                    "locate",
                    "read",
                    "gather",
                    "extract",
                    "extracts",
                    "obtain",
                    "acquire",
                    "check",
                    "checks",
                    "detect",
                    "detects",
                    "record",
                    "records",
                ]
                v_exec = [
                    "use",
                    "execute",
                    "executed",
                    "run",
                    "ran",
                    "launch",
                    "call",
                    "perform",
                    "list",
                    "invoke",
                    "inject",
                    "open",
                    "opened",
                    "target",
                    "resume",
                    "exec",
                ]
                v_mmap = ["allocate", "assign"]
                v_fork = [
                    "clone",
                    "clones",
                    "spawned",
                    "spawn",
                    "spawns",
                    "issue",
                    "set",
                ]
                v_setuid = ["elevate", "impersonated"]
                v_send = [
                    "send",
                    "sent",
                    "transfer",
                    "post",
                    "postsinformation",
                    "postsinformations",
                    "move",
                    "transmit",
                    "deliver",
                    "push",
                    "redirect",
                    "redirects",
                ]
                v_receive = ["receive", "accept", "take", "get", "gets", "collect"]
                v_connect = [
                    "click",
                    "browse",
                    "browses",
                    "connect",
                    "connected",
                    "portscan",
                    "connects",
                    "alerts",
                    "communicates",
                    "communicate",
                ]
                v_chmod = [
                    "chmod",
                    "change permission",
                    "changes permission",
                    "permision-modifies",
                    "modifies permission",
                    "modify permission",
                ]
                v_load = ["load", "loads"]
                v_exit = [
                    "terminate",
                    "terminates",
                    "stop",
                    "stops",
                    "end",
                    "finish",
                    "break off",
                    "abort",
                    "conclude",
                ]
                v_2D = {"collect": ("read", "receive"), "open": ("exec", "fork")}

                for i in range(len(no_zero_nodes_plus_3)):
                    for index, item in enumerate(no_zero_nodes_plus_3[i]):
                        if item.split(": ")[0] == "V":
                            if item.split(": ")[1] in v_unlink:
                                no_zero_nodes_plus_3[i][index] = "V: " + "unlink"
                            elif item.split(": ")[1] in v_write:
                                no_zero_nodes_plus_3[i][index] = "V: " + "write"
                            elif item.split(": ")[1] in v_read:
                                no_zero_nodes_plus_3[i][index] = "V: " + "read"
                            elif item.split(": ")[1] in v_exec:
                                no_zero_nodes_plus_3[i][index] = "V: " + "exec"
                            elif item.split(": ")[1] in v_mmap:
                                no_zero_nodes_plus_3[i][index] = "V: " + "mmap"
                            elif item.split(": ")[1] in v_fork:
                                no_zero_nodes_plus_3[i][index] = "V: " + "fork"
                            elif item.split(": ")[1] in v_setuid:
                                no_zero_nodes_plus_3[i][index] = "V: " + "setuid"
                            elif item.split(": ")[1] in v_send:
                                no_zero_nodes_plus_3[i][index] = "V: " + "send"
                            elif item.split(": ")[1] in v_receive:
                                no_zero_nodes_plus_3[i][index] = "V: " + "receive"
                            elif item.split(": ")[1] in v_connect:
                                no_zero_nodes_plus_3[i][index] = "V: " + "connect"
                            elif item.split(": ")[1] in v_chmod:
                                no_zero_nodes_plus_3[i][index] = "V: " + "chmod"
                            elif item.split(": ")[1] in v_load:
                                no_zero_nodes_plus_3[i][index] = "V: " + "load"
                            elif item.split(": ")[1] in v_exit:
                                no_zero_nodes_plus_3[i][index] = "V: " + "exit"
            else:
                continue
            all_nodes += no_zero_nodes_plus_3
            if my_svo_triplet:
                all_nodes += my_svo_triplet
        print("*****all_nodes:::", all_nodes)
        return all_nodes

    def fix_srl_spacing(self, li):
        """
        Fix spacing issues in the elements of a nested list.

        Args:
            li (list): A nested list where each sublist contains strings.

        Returns:
            list: A nested list with spacing issues corrected. Each string element
            in the original list and its sublists is processed to ensure
            consistency in spacing, particularly around file extensions.
        """
        new_lst = []
        for i in li:
            x = []
            for j in i:
                files = re.findall(
                    load_patterns(RE_PATTERNS_FILE_PATH)["*_extension"], j
                )
                if files:
                    for k in files:
                        j = j.replace(k, re.sub(" ", " *", k))
                    x.append(j)
                else:
                    x.append(j)
            new_lst.append(x)
        return new_lst

    def negation_clauses(self, srl_input):
        """
        Filters out sentences containing conditional clauses.

        Args:
            srl_input (list): A list of sentences or elements to filter.

        Returns:
            list: A list of sentences from `srl_input` that do not contain
            conditional clauses such as 'if', 'otherwise', or 'when'.

        Example:
            Suppose `srl_input` is ['If it rains, bring an umbrella.',
            'The meeting will proceed as planned.', 'Otherwise, we will reschedule.'].
            Calling `filter_conditional_clauses(srl_input)` would return:
            ['The meeting will proceed as planned.']

        Notes:
            This method filters out sentences containing common conditional
            clauses ('if', 'otherwise', 'when') to focus on sentences that do not
            include such clauses. It helps in selecting sentences that describe
            actions or events without introducing conditions or alternatives.
        """
        conditional_clause = ["if", "otherwise", "when"]
        purged = []
        purged[:] = [
            x
            for x in srl_input
            if "ARGM-NEG" in str(x)
            if "if" and "when" not in str(x).lower()
        ]
        positive_sentences = [i for i in srl_input if i not in purged]
        return positive_sentences

    ####################################################################################

    def group_partials(self, strings):
        """
        This method iterates over a sorted list of strings. For each string,
        it checks if the previous string (`prev`) is a substring of the current
        string (`s`). If not, it yields the previous string (`prev`) and updates
        `prev` to the current string (`s`). The process continues until all strings
        are processed.

        Args:
            strings (list): A list of strings to be grouped.

        Yields:
            str: Unique strings based on a specific condition.
        """
        try:
            s = ""
            it = iter(sorted(strings))
            prev = next(it)
            for s in it:
                if s.find(prev) == -1:
                    yield prev
                prev = s
            yield s
        except StopIteration:
            return

    def process_convert(self, lst):
        """
        Process and convert each string in the list by wrapping it with '/-' and '-/',
        and replacing spaces with hyphens.

        Args:
            lst (list): A list of strings to be processed and converted.

        Returns:
            list: A list of processed strings with the specified transformations.

        Example:
            If `lst` is ['abc def', 'ghi jkl']:
            Calling `process_convert(lst)` would return ['/-abc-def-/', '/-ghi-jkl-/'].

        Notes:
            - Each string in `lst` is first wrapped with '/-' and '-/'.
            - Any spaces (' ') in each string are replaced with hyphens ('-').

        """
        lst = ["/-" + i.strip() + "-/" for i in lst]
        return [i.replace(" ", "-") for i in lst]

    ####################################################################################

    def astriks(self, lis):
        """
        Process a list of strings `lis` and update each string based on certain conditions and transformations.

        Args:
            lis (list): A list of strings to be processed.

        Returns:
            list: A nested list where each sublist contains processed strings based on the conditions.

        Notes:
            - The method performs various transformations on each string in `lis`.
            - It checks for the presence of `":"` in each string to determine the structure.
            - It looks for application processes in the strings and processes them accordingly.
            - IOCs (Indicators of Compromise) associated with the strings are processed and appended as necessary.
            - Different conditions are applied based on whether IOCs are present or not, and based on the content of `leftnode`.
        """
        apps_process = load_lists(SEC_PATTERNS_FILE_PATH)["APPs-PROCESS"]
        apps_process = apps_process.replace("'", "").strip("][").split(" , ")
        updated_list = [[] for x in range(len(lis))]
        for jj, lst in enumerate(lis):
            for i in range(len(lst)):
                if ":" not in lst[i]:
                    lst[i] = "TMP: " + lst[i]
                leftnode = lst[i].split(":", 1)[0]
                rightnode = lst[i].split(":", 1)[1]
                lOFioc = iocs.list_of_iocs(rightnode)
                found_app = [app for app in apps_process if app in rightnode.lower()]
                if len(found_app) > 1:
                    found_app = list(self.group_partials(found_app))
                    found_app = self.process_convert(found_app)
                else:
                    found_app = self.process_convert(found_app)
                if not lOFioc:
                    if leftnode.lower() != "v" and not found_app:
                        updated_list[jj].append(leftnode + ": *")
                    elif leftnode.lower() == "v":
                        updated_list[jj].append(lst[i])
                    elif leftnode.lower() != "v" and found_app:
                        for process in found_app:
                            updated_list[jj].append(leftnode + ": " + process)
                elif len(lOFioc) == 1:
                    updated_list[jj].append(leftnode + ": " + lOFioc[0])
                elif len(lOFioc) >= 2:
                    updated_list[jj].append(leftnode + ": " + lOFioc[0])
                    for index in range(1, len(lOFioc)):
                        updated_list[jj].append("ARG-NEW: " + lOFioc[index])
        return updated_list

    ####################################################################################

    def triplet_builder(self, mylist):
        """
        Build triplets from a list of lists (`mylist`), where each sublist represents a potential triplet.

        Args:
            mylist (list): A list of lists where each sublist contains elements to form triplets.

        Returns:
            list: A list of lists, where each inner list represents a triplet (e.g., ['subject', 'verb', 'object']).

        Notes:
            - This method constructs triplets based on the content and structure of `mylist`.
            - Triplets are formed by combining elements from each sublist in `mylist`.
            - It ensures that verbs ('v') are positioned between other elements in the triplets.
        """
        triplet = []
        for i in range(len(mylist)):
            xv = mylist[i]
            if len(mylist[i]) > 3:
                for index, value in enumerate(mylist[i]):
                    l = len(mylist[i])
                    if "v" in value.lower() and index != 0 and index != l - 1:
                        for k in mylist[i][:index]:
                            for j in mylist[i][index + 1 :]:
                                triplet.append((k, value, j))
            elif len(mylist[i]) == 3:
                triplet.append(mylist[i])
        return list(map(list, triplet))

    ####################################################################################
