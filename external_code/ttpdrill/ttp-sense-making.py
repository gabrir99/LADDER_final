from utilities import FileReader, StanfordServer
from relation_miner import relation_miner
from ontology_reader import ReadOntology
from BM25 import BM25, BM25L, BM25Okapi, BM25Plus


def getGhaithOntology(isStemmer):
    from ontology_reader import ParseGhaithOntology

    ontology = ParseGhaithOntology(isStemmer)
    ontology_dict = ontology.read_csv()
    what_list, list_map_dict = ontology.parse_ontology(ontology_dict)
    ontology = ReadOntology()
    ttp_df = ontology.read_mitre_TTP()
    return what_list, list_map_dict, ttp_df


def getOntology(isStemmer):
    """
    Get the following information from the files containing the ontology informations based on the mitre framework.

    Returns:
    list: new_list is a list containing all the content of the ontology list but lemmatized or stemmed
    dict: list_map_dict {index_of_new_list -> mitre_id}
    dict: ttp_df  {mitre_id -> {tecnique: .., tactic: ...}}
    """
    from ontology_reader import ReadOntology

    file_name = "resources/ontology_details.csv"
    ontology = ReadOntology()
    ontology_df = ontology.read_csv(file_name)
    ontology_dict = ontology.data_frame.to_dict("records")
    __ = ontology.split_ontology_list(ontology_dict)
    what_list = list()
    for i in __:
        what_list.append(
            [
                i["Id"],
                i["action_what"],
                i["action_where"],
                i["why_what"],
                i["why_where"],
            ]
        )
    #    print(what_list[0])
    what_list, list_map_dict = combine_parsed_ontology_in_bow(what_list, isStemmer)
    ttp_df = ontology.read_mitre_TTP()
    # for key, val in ttp_df.items():
    #     print(key, val['TECHNIQUE'] )
    # print(ttp_df)

    # file_name = 'resources/ontology_details_1.csv'
    # file_name = 'resources/ontology_details.csv'
    # file_name = 'resources/export_dataframe.csv'
    # ontology = ReadOntology()
    # ontology_df = ontology.read_csv(file_name)
    # ontology_dict = ontology.refine_ontology()
    # stem_list = ontology.stem()
    # # for ont  in stem_list:
    # #     print(ont['action_what'])
    # ontology.print_ontology(stem_list)
    #
    #
    #
    # what_list = list()
    # for i in stem_list:
    #     what_list.append([i['Id'], i['action_what'], i['action_where']])
    # #what_list[0]

    return what_list, list_map_dict, ttp_df


def combine_parsed_ontology_in_bow(what_list, isStemmer):
    """
    For each entry in the ontology takes all the values, concatenates everything and then
    applies the stemmer or lemmatizer.

    Parameters:
    isStemmer (bool):
    what_list (list): list of lists of the kind
    [ [i['Id'], i['action_what'], i['action_where'], i['why_what'], i['why_where']], [..], ..]

    Returns:
    list: new_list is a list containing all the content of the what list but lemmatized or stemmed
    dict: list_map_dict {index -> mitre_id}
    """
    list_map_dict = dict()
    new_list = list()
    for index in enumerate(what_list):
        # print(tuples)
        a = list()
        for attribute in index[1]:
            if type(attribute) is str:
                a.append(attribute.strip())
            elif type(attribute) is list:
                for each_sttribute in attribute:
                    for each_word in each_sttribute.split(" "):
                        if isStemmer:
                            a.append(stemmer.stem(each_word.strip()))
                        else:
                            a.append(lemmatizer.lemmatize(each_word.strip()))
        a = utilities.remove_stopwords(a)
        list_map_dict[index[0]] = a[0]
        new_list.append(a[1:])

    return new_list, list_map_dict


def getReportExtraction(
    isFile,
    isStemmer,
    isServerRestart,
    file_name="reports/fireeye_fin7_application_shimming.txt",
):
    """
    Report extraction by instantiating:
    - the stanford server
    - processing the text

    Returns:
    list: list of bow for each sentence (ex.)
    [
        {
            "text": ....,
            "bow": ["..", ..]
        },
        ..
    ]
    """
    #    file_name = 'reports/fireeye_fin7_application_shimming.txt'
    stanfordServer = StanfordServer()
    if isServerRestart:
        stanfordServer.startServer()
    stanfordNLP = stanfordServer.get_stanforcorenlp()
    # print(stanfordNLP)

    preprocess_tools = FileReader(file_name)
    # text_list = list of str, contains valid sentences with no fullstop or file path with them
    if isFile:
        text = preprocess_tools.read_file()
        text = text.replace("\n", " ")
        text_list = preprocess_tools.get_sent_tokenize(text)
    else:
        text = file_name
        text_list = preprocess_tools.get_sent_tokenize(text)
    # print(text_list)

    r_miner = relation_miner(stanfordNLP)
    extracted_infor_list = r_miner.all_imp_stuff(text_list)
    extracted_list = get_all(extracted_infor_list, isStemmer)

    return extracted_list


def get_all(temp_list, isStemmer):
    """
    Generate stemmed or lemmatized bag of word format of each sentence.

    Paarameters:
    isStemmer (bool): if true uses PorterStemmer otherwise use Lemmatizer
    templ_list (list):
    [
            {
                'what' : {},
                'where' : {},
                'where_attribute' : {},
                'why' : {},
                'when' : {},
                'how' : {},
                'subject' : {},
                'action' : {},
                'text' : "....",
            },
            ...
    ]

    Returns:
    list: list of bow for each sentence (ex.)
    [
        {
            "text": ....,
            "bow": ["..", ..]
        },
        ..
    ]
    """
    all_list = list()
    for temp_dict in temp_list:
        all_dict = dict()
        tt_list = list()
        ## basically recreates the sentence but lemmatized or stemmed in a bow format
        for key, val in temp_dict.items():
            if (
                key == "what"
                or key == "where"
                or key == "where_attribute"
                or key == "why"
                or key == "when"
                or key == "how"
                or key == "subject"
            ):
                for __ in val:
                    if isStemmer:
                        tt_list.append(stemmer.stem(__))
                    else:
                        tt_list.append(lemmatizer.lemmatize(__))
        all_dict["text"] = temp_dict["text"]
        all_dict["bow"] = tt_list
        all_list.append(all_dict)
    return all_list


from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
import utilities


def buildBM25(what_list):
    """
    :param what_list:
    :return:
    """
    """
    Example of using BM25
    """
    # corpus = [
    #     "Hello there good man!",
    #     "It is quite dry in London",
    #     "How is the weather today?",
    #     "Hello Ruhani, How was your trip to London",
    #     "Did the weather in London is windy?"
    # ]
    # corpus = [
    #     "use scanner","inject","copy","delete","file","use"
    # ]

    # tokenized_corpus = [word_tokenize(doc) for doc in corpus]
    # #tokenized_corpus = [doc.split(" ") for doc in corpus]
    # tokenized_stemmed_corpus = list()
    # for doc in tokenized_corpus:
    #     doc_stemmed = list()
    #     for word in doc:
    #         doc_stemmed.append(stemmer.stem(word))
    #     tokenized_stemmed_corpus.append(doc_stemmed)

    # tokenized_corpus = [word_tokenize(doc) for doc in what_list]
    # print(tokenized_stemmed_corpus)
    bm25 = BM25Okapi(what_list)
    return bm25


def query(extracted_list, ontology_list, list_map, bm25_model, ttp_df, isStemmer):
    """
    The extracted list contains all the sentences extracted from the text we are analyzing. Applies bm25 to the extracted_sentences,
    considering that the bm25 has been built using the saved ontology based on the mitre tactic and techniques.
    It prints out all the ttps that have been found in the text

    Parameters:
    extracted_list (list): list of bow for each sentence (ex.)
    [
        {
            "text": ....,
            "bow": ["..", ..]
        },
        ..
    ]
    ontology_list (list): is a list containing all the content of the ontology list but lemmatized or stemmed
    list_map (dict):  {index_of_new_list -> mitre_id}
    ttp_df (dict):  {mitre_id -> {tecnique: .., tactic: ...}}

    """
    for __ in extracted_list:  # __ = { "text": ...., "bow": ["..", ..] }
        #    print(__['bow'])
        if isStemmer:
            tokenized_query = [stemmer.stem(word) for word in __["bow"]]
        else:
            tokenized_query = [lemmatizer.lemmatize(word) for word in __["bow"]]

        #    print(tokenized_query)
        doc_scores = bm25_model.get_scores(tokenized_query)
        #    print(doc_scores)

        print("Text:\n", __["text"], "\n")
        print("Extracted Information:\n", __["bow"], "\n")
        print("Mapped:\n")
        #        scores = bm25_model.get_scores(tokenized_query)
        top_index, match_ttp, score = bm25_model.get_top_n(
            tokenized_query, ontology_list, n=5
        )
        create_ttp_map(__, list_map, ttp_df, top_index, match_ttp, score)
        # for ___ in zip(top_index, match_ttp, score):
        #     try:
        #         ttp_index = ___[0]
        #         ttp_id = list_map[___[0]]
        #         ttp_technique = ttp_df[ttp_id]['TECHNIQUE']
        #         ttp_tactic = ttp_df[ttp_id]['TACTIC']
        #         ttp_ontology = ___[1]
        #         ttp_score = ___[2]
        #         # ___[2] == score
        #         # ___[1] == match_ttp
        #         if (ttp_score > 0.1):
        #             print(ttp_index, ' : ', ttp_score, ' : ', ttp_id, ' : ', ttp_technique, ' : ', ttp_tactic, ' : ', ttp_ontology)
        #     except:
        #         print('None')
        # print('\n\n')
    return


def create_ttp_map(text_dict, list_map, ttp_df, top_index, match_ttp, score):
    """

    Parameters:
    text_dict (dict): { "text": ...., "bow": ["..", ..] }
    list_map (dict):  {index_of_new_list -> mitre_id}
    ttp_df (dict):  {mitre_id -> {tecnique: .., tactic: ...}}
    top_index (list): list of index of the ontology_list for the matched techniques
    match_ttp (list):  list of dict of the matched ttps
    score (list): list of scores for the matched ttps

    Returns:
    list: list of ttps object containing the stix informations for an attack pattern for all the matched ttps.
    """
    __list__ = list()

    for ___ in zip(top_index, match_ttp, score):
        try:
            ttp_index = ___[0]
            ttp_id = list_map[___[0]]
            ttp_technique = ttp_df[ttp_id]["TECHNIQUE"]
            ttp_tactic = ttp_df[ttp_id]["TACTIC"]
            ttp_ontology = ___[1]
            ttp_score = ___[2]
            if ttp_score > 0.1:
                __dict__ = dict()
                __dict__["serial"] = "1"
                __dict__["subSerial"] = "0"
                __dict__["typeOfAction"] = "s"
                __dict__["original_sentence"] = text_dict["text"]
                __temp_dict__ = {
                    "description": "",
                    "data": text_dict["bow"],
                    "link": "",
                    "highlight": "",
                }
                __dict__["action"] = __temp_dict__
                __temp_dict__ = {
                    "description": "",
                    "data": ttp_id,
                    "link": "",
                    "highlight": "",
                }
                __dict__["techId"] = __temp_dict__
                __temp_dict__ = {
                    "description": "",
                    "data": ttp_technique,
                    "link": "",
                    "highlight": "",
                }
                __dict__["technique"] = __temp_dict__
                __temp_dict__ = {
                    "description": "",
                    "data": ttp_tactic,
                    "link": "",
                    "highlight": "",
                }
                __dict__["tactic"] = __temp_dict__

                __list__.append(__dict__)
        except:
            print("None")
    print("\n\n")
    print(__list__)
    return __list__


def read_API_doc():
    import pandas

    api_data_frame = pandas.read_csv(
        "resources/API_Description_MSDN.csv", encoding="ISO-8859-1"
    )
    return api_data_frame.to_dict("records")


if __name__ == "__main__":
    isAPI = False

    """----------------------------------------------------------------------------------------"""
    ### Building Ontology
    isStemmer = True
    what_list, list_map, ttp_df = getOntology(isStemmer)
    # what_list, list_map, ttp_df = getGhaithOntology(isStemmer)
    bm25_model = buildBM25(what_list)
    """-----------------------------------------------------------------------------------------"""

    if isAPI:
        api_dict_list = read_API_doc()
        print(
            "----------------------------------------------------------------------------------------"
        )
        for api in api_dict_list:
            # for key, val in api.items():
            print(
                "-----------------------------------"
                + api["API_NAME"]
                + "-----------------------------------"
            )
            print("API_NAME: ", api["API_NAME"])
            print("API_Description: ", api["API_Description"])
            extracted_list = getReportExtraction(
                False, isStemmer, False, api["API_Description"]
            )
            query(extracted_list, what_list, list_map, bm25_model, ttp_df, isStemmer)

            print(
                "-----------------------------------"
                + api["API_NAME"]
                + "-----------------------------------\n\n"
            )

    else:
        isFile = True

        # while(True):
        if isFile:
            report_name = "input.txt"
            # report_name = 'C:\\Users\\rrahman3\\Google Drive\\Study UNCC\\TTPDrill Handover\\Raw Threat Reports\\ThreatReport\\relevent\\Infostealer.Alina_ThreatReport.txt'
            # report_name = 'C:\\Users\\rrahman3\\Google Drive\\Study UNCC\\TTPDrill Handover\\Raw Threat Reports\\ThreatReport\\ghaith\\tested_output\\Trojan.Downexec.B_ThreatReport.txt'

        else:
            report_name = input("Enter Text:\t")
        extracted_list = getReportExtraction(isFile, isStemmer, False, report_name)
        # print(what_list)
        query(extracted_list, what_list, list_map, bm25_model, ttp_df, isStemmer)
