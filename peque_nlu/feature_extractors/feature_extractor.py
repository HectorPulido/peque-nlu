import re
import unicodedata
from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    stopwords = None

    def _strip_accents(self, text):
        text = unicode(text, "utf-8")
        text = (
            unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
        )
        return str(text)

    def preprocess_input(self, text_to_decode):
        text_to_decode = self._strip_accents(text_to_decode)
        text_to_decode = text_to_decode.lower()
        text_to_decode = re.sub(r"[^a-zA-Z\s]+", "", text_to_decode)
        text_to_decode = [
            word for word in text_to_decode.split() if word not in self.stopwords
        ]
        return text_to_decode

    @abstractmethod
    def get_features(self, text_to_decode, threshold):
        pass

    @abstractmethod
    def fit(self, dataset_path):
        pass
