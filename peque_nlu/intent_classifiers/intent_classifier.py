"""
Intent classifier abstract class module.
"""
from abc import ABC, abstractmethod


class IntentClassifier(ABC):
    """
    Abstract class for intent classifiers.
    """

    @abstractmethod
    def predict(self, text):
        """
        Predict the intent of the input text.
        """

    @abstractmethod
    def multiple_predict(self, texts):
        """
        Predict the intent of multiple texts.
        """

    @abstractmethod
    def fit(self, dataset_path):
        """
        Fit the intent classifier.
        """

    @abstractmethod
    def save(self, path):
        """
        Save the model.
        """

    @staticmethod
    @abstractmethod
    def load(saver, path):
        """
        Load the model.
        """
