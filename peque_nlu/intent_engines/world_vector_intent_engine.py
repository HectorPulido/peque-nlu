import gensim.downloader as gd
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from peque_nlu.intent_engines import BasicIntentEngine


class WorldVectorIntentEngine(BasicIntentEngine):
    def __init__(self, language, gensim_model=None):
        self.stopwords = stopwords.words(language)
        self.json_dataset = None

        if isinstance(gensim_model, str):
            self.glove_vectors = gd.load(gensim_model)
        elif isinstance(gensim_model, KeyedVectors):
            self.glove_vectors = gensim_model
        else:
            raise ValueError(
                "gensim_model must be a model_name (str) or a KeyedVectors object"
            )

    def _pred(self, text):
        text = text.lower().split()

        results = []
        for intent, examples in self.json_dataset.items():
            for example in examples:
                example = example.lower().split()
                similarity = self.glove_vectors.wmdistance(text, example)
                results.append((intent, similarity))

        results = sorted(results, key=lambda x: x[1])
        return results[0]

    def predict(self, text):
        intents = []
        probabilities = []

        for text_item in text:
            result = self._pred(text_item)
            intents.append(result[0])
            probabilities.append(result[1])
        return intents, probabilities

    def fit(self, text, intent):
        self.json_dataset = {}

        for text_item, intent_item in zip(text, intent):
            if intent_item not in self.json_dataset:
                self.json_dataset[intent_item] = []

            self.json_dataset[intent_item].append(text_item)
