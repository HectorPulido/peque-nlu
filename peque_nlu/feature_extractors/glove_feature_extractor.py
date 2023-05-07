import re
import gensim.downloader as gd
from gensim.models.keyedvectors import KeyedVectors

from peque_nlu.feature_extractors import FeatureExtractor
from peque_nlu.utils import IntentUtils


class GloveFeatureExtractor(FeatureExtractor, IntentUtils):
    def __init__(self, gensim_model=None):
        self.entities = {}
        self.stopwords = None

        if isinstance(gensim_model, str):
            self.glove_vectors = gd.load(gensim_model)
        elif isinstance(gensim_model, KeyedVectors):
            self.glove_vectors = gensim_model
        else:
            raise ValueError(
                "gensim_model must be a model_name (str) or a KeyedVectors object"
            )

    def _check_examples(self, examples):
        for word in examples:
            if not self._word_is_in_glove(word):
                examples.remove(word)
        return examples

    def _word_is_in_glove(self, word):
        return word in self.glove_vectors.key_to_index

    def fit(self, dataset_path, stopwords=None):
        self.stopwords = stopwords
        self.entities = self.get_entities(dataset_path)

    def get_features(self, text_to_decode, threshold):
        text_to_decode = self.preprocess_input(text_to_decode)

        matches = []
        for word in text_to_decode:
            for entity, examples in self.entities.items():
                if word in examples:
                    matches.append({"word": word, "entity": entity, "similarities": 1})
                    continue

                if not self._word_is_in_glove(word):
                    continue

                examples = self._check_examples(examples)
                similarities = self.glove_vectors.similarity(examples, word)

                similarity = max(similarities)

                if similarity > threshold:
                    matches.append(
                        {"word": word, "entity": entity, "similarities": similarity}
                    )
        return matches
