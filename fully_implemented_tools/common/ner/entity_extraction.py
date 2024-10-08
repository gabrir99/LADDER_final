from abc import ABC, abstractmethod


class EntityExtraction(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        """
        initialize entity extraction model
        """
        pass

    @abstractmethod
    def get_entities(self, text):
        """

        :param text: input text
        :return: List of Entity objects
        """
        pass
