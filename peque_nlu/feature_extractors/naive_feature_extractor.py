from peque_nlu.feature_extractors import FeatureExtractor
from peque_nlu.utils import IntentUtils


class NaiveFeatureExtractor(FeatureExtractor, IntentUtils):
    def __init__(self):
        self.entities = {}
        self.stopwords = None

    def fit(self, dataset_path, stopwords=None):
        self.stopwords = stopwords
        self.entities = self.get_entities(dataset_path)

    def get_features(self, text_to_decode, threshold):
        text_to_decode = self.preprocess_input(text_to_decode)

        matches = []
        for word in text_to_decode:
            for entity, examples in self.entities.items():
                for example in examples:
                    example = example.lower()
                    if example not in word:
                        continue
                    matches.append({"word": word, "entity": entity, "similarities": 1})
        return matches
