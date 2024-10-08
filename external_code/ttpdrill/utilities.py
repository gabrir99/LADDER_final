class FileReader:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_file(self):
        """
        Takes the content of the file and saves it in a string
        """
        with open(self.file_name, "r", encoding="utf8") as report:
            strs = report.read()
        self.text = strs
        return strs

    def get_all_sentences(self, text):
        """
        Splits everything using \n and then uses the sent_tokenize function to obtain all
        the exisiting sentences

        Parameters:
        text (str): string of text

        Returns:
        list: list of sentences
        """
        sentence_tokenize_list = list()
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            sent_tokenize = nltk.sent_tokenize(paragraph)
            for sentence in sent_tokenize:
                sentence_tokenize_list.append(sentence)
        return sentence_tokenize_list

    def get_all_valid_sentences(self, text):
        """
        Splits everything using \n and then uses the sent_tokenize function to obtain all
        the exisiting sentences. Moreover it applies some cheks to determine if the found sentence is really a valid one

        Parameters:
        text (str): string of text

        Returns:
        list: list of sentences
        """
        sentence_tokenize_list = list()
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            sent_tokenize = nltk.sent_tokenize(paragraph)
            for sentence in sent_tokenize:
                ###
                if len(sentence) == sentence.rfind(".") + 1:
                    sentence = self.remove_fullstop(sentence)
                    sentence = self.remove_filepath(sentence)
                    sentence_tokenize_list.append(sentence)
        return sentence_tokenize_list

    def get_sent_tokenize(self, text):
        sentence_tokenize_list = list()
        sent_tokenize = nltk.sent_tokenize(text)
        for sentence in sent_tokenize:
            ###
            if len(sentence) == sentence.rfind(".") + 1:
                sentence = self.remove_fullstop(sentence)
                sentence = self.remove_filepath(sentence)
                sentence_tokenize_list.append(sentence)
        return sentence_tokenize_list

    def remove_fullstop(self, text):
        """This method removes the fullstop in the middle of a sentence eg. 'services.exe' changes to 'servicesexe'."""
        """
        :parameter text str
        :return str
        """
        result_string = text.replace(".", "", text.count(".") - 1)
        return result_string

    def remove_filepath(self, text):
        # Finding Windows-style file paths:
        x = re.findall(r"([a-zA-Z]:[\\\\[a-zA-Z0-9]*]*)", text)
        # replace all found occurences with 'file path'
        for _ in x:
            text = text.replace(_, "file path", 1)

        # Finding Unix-style file paths:
        x = re.findall(r"(\/.*?\.[\w:]+)", text)
        for _ in x:
            text = text.replace(_, "file path", 1)
        return text


import re
import nltk
from nltk.corpus import stopwords
from pycorenlp import StanfordCoreNLP
import json
import os
from nltk.parse.corenlp import CoreNLPServer


# nltk.download('stopwords')


class StanfordServer:
    def __init__(self):
        pass

    def startServer(self):
        """
        Take the path to the:
        - java installation
        - stanford-corenlp zip
        After that it start the CoreNLPServer on the port 9000
        """
        java_path = "C:\\dev\\tool\\java\\bin\\java.exe"
        os.environ["JAVAHOME"] = java_path

        home = os.path.expanduser("~")
        download_path = os.path.join(home, "Downloads")
        print(download_path)
        # # The server needs to know the location of the following files:
        # #   - stanford-corenlp-X.X.X.jar
        # #   - stanford-corenlp-X.X.X-models.jar
        STANFORD = os.path.join(download_path, "stanford-corenlp-4.5.7")

        # # Create the server
        server = CoreNLPServer(
            os.path.join(STANFORD, "stanford-corenlp-4.5.7-models.jar"),
            os.path.join(STANFORD, "stanford-corenlp-4.5.7.jar"),
            os.path.join(STANFORD, "stanford-corenlp-4.5.7-models-english.jar"),
        )

        # # Start the server in the background
        server.start()
        print("Server Started")

        self.stanfordCoreNLP = StanfordCoreNLP("http://localhost:9000")

        return self.stanfordCoreNLP

    def get_stanforcorenlp(self):
        self.stanfordCoreNLP = StanfordCoreNLP("http://localhost:9000")
        return self.stanfordCoreNLP


def remove_stopwords(word_tekens):
    import nltk
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    filtered_sentence = [w for w in word_tekens if not w in stop_words]
    return filtered_sentence


# print(remove_stopwords(['ruani','a','mon','the','ruh']))
