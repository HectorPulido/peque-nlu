"""
This module contains the abstract class for intent savers.
"""
from abc import ABC, abstractmethod


class IntentSaver(ABC):
    """
    Abstract class for intent savers.
    """

    @abstractmethod
    def save(self, model, path):
        """
        Save the model to the path.

        :param model: The model to save.
        :type model: object.
        :param path: The path to save the model.
        :type path: str.
        """

    @abstractmethod
    def load(self, path) -> object:
        """
        Load the model from the path.

        :param path: The path to load the model.
        :type path: str.
        :return: The model.
        :rtype: object.
        """
