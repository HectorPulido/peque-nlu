"""
The WorldVectorIntentEngine class module.
"""
from nltk.corpus import stopwords
from peque_nlu.intent_engines import BasicIntentEngine
from peque_nlu.utils import glove_load


class WorldVectorIntentEngine(BasicIntentEngine):
    """
    The WorldVectorIntentEngine class.

    This class is used to create a world vector intent engine.
    This works by checking using the glove vectors, to check if there are
    similarities between the input text and the examples.
    """

    def __init__(self, language, gensim_model=None):
        """
        Initialize the WorldVectorIntentEngine.

        :param language: The language to use.
        :type language: str.

        :param gensim_model: The gensim model to use.
            It can be a model name (str) or a KeyedVectors object.
        :type gensim_model: str or KeyedVectors.
        """

        self.stopwords = stopwords.words(language)
        self.json_dataset = None
        self.glove_vectors = glove_load(gensim_model)

    def _pred(self, text) -> tuple:
        """
        Predict the intent of the input text.

        :param text: The input text.
        :type text: str.
        :return: The intent and the probability.
        :rtype: tuple.
        """
        text = text.lower().split()

        results = []
        for intent, examples in self.json_dataset.items():
            for example in examples:
                example = example.lower().split()
                similarity = self.glove_vectors.wmdistance(text, example)
                results.append((intent, similarity))

        results = sorted(results, key=lambda x: x[1])
        return results[0]

    def predict(self, text) -> tuple:
        """
        Predict the intent of the input text.

        :param text: The input text.
        :type text: str.
        :return: The intent and the probability.
        :rtype: tuple.
        """

        intents = []
        probabilities = []

        for text_item in text:
            result = self._pred(text_item)
            intents.append(result[0])
            probabilities.append(result[1])
        return intents, probabilities

    def fit(self, text, intent):
        """
        Fit the intent engine to train the model.

        :param text: The input text.
        :type text: str.
        :param intent: The intent of the input text.
        :type intent: str.
        """

        self.json_dataset = {}

        for text_item, intent_item in zip(text, intent):
            if intent_item not in self.json_dataset:
                self.json_dataset[intent_item] = []

            self.json_dataset[intent_item].append(text_item)
