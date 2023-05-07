"""
Intent engine abstract class module.
"""
from abc import ABC, abstractmethod


class BasicIntentEngine(ABC):
    """
    Abstract class for intent engines.
    """

    @abstractmethod
    def predict(self, text) -> tuple:
        """
        Predict the intent of the input text.

        :param text: The input text.
        :type text: str.
        :return: The predicted intent and the confidence.
        :rtype: tuple.

        example: predict("hello") -> ("greet", 1)
        """

    @abstractmethod
    def fit(self, text, intent):
        """
        Fit the intent engine to train the model.

        :param text: The input text.
        :type text: str.

        :param intent: The intent of the input text.
        :type intent: str.

        """
