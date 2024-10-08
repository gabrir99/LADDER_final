from abc import ABC, abstractmethod


class SentenceClassification(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        """
        initialize entity extraction model
        """
        pass

    @abstractmethod
    def get_classified_sentences(self, sentences : list):
        """

        :param text: input text
        :return: List of Entity objects
        """
        pass
