"""
This module contains the abstract class for feature extractors.
"""
import re
import unicodedata
from abc import ABC, abstractmethod
from peque_nlu.utils import IntentUtils


class FeatureExtractor(ABC, IntentUtils):
    """
    Abstract class for feature extractors.
    """

    stopwords = None
    entities = None

    def _strip_accents(self, text) -> str:
        """
        Strip accents from input String.

        :param text: The input string.
        :type text: str.
        :return: The processed string.
        :rtype: str.

        example: _strip_accents("àéêöhello") -> "aeeohello"
        """
        try:
            text = unicode(text, "utf-8")
        except NameError:  # unicode is a default on python 3
            pass

        text = (
            unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
        )
        return str(text)

    def preprocess_input(self, text_to_decode) -> list:
        """
        Preprocess the input text to decode.

        :param text_to_decode: The input text to decode.
        :type text_to_decode: str.
        :return: The processed text.
        :rtype: list.

        example: preprocess_input("àéêöhello") -> ["hello"]
        """

        text_to_decode = self._strip_accents(text_to_decode)
        text_to_decode = text_to_decode.lower()
        text_to_decode = re.sub(r"[^a-zA-Z\s]+", "", text_to_decode)
        text_to_decode = [
            word for word in text_to_decode.split() if word not in self.stopwords
        ]
        return text_to_decode

    @abstractmethod
    def get_features(self, text_to_decode, threshold) -> list:
        """
        Get the features from the input text.

        :param text_to_decode: The input text to decode.
        :type text_to_decode: str.
        :param threshold: The threshold to apply.
        :type threshold: float.
        :return: The features.
        :rtype: list.

        """

    def fit(self, dataset_path, stopwords=None):
        """
        Fit the feature extractor.

        :param dataset_path: The path of the dataset.
        :type dataset_path: str.

        """

        self.stopwords = stopwords
        self.entities = self.get_entities(dataset_path)
