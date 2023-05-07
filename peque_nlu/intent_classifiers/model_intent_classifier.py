"""
The model intent classifier module.
"""
from peque_nlu.intent_classifiers import IntentClassifier
from peque_nlu.utils import IntentUtils

from peque_nlu.intent_engines import LogisticIntentEngine


class ModelIntentClassifier(IntentClassifier, IntentUtils):
    """
    The ModelIntentClassifier class.

    This class is used to create a model intent classifier.
    """

    def __init__(
        self, language, intent_engine=None, feature_extractor=None, saver=None
    ):
        """
        Initialize the ModelIntentClassifier.

        :param language: The language to use.
        :type language: str.

        :param intent_engine: The intent engine to use.
        :type intent_engine: IntentEngine.

        :param feature_extractor: The feature extractor to use.
        :type feature_extractor: FeatureExtractor.

        :param saver: The saver to use.
        :type saver: Saver.

        """

        if intent_engine is None:
            self.intent_engine = LogisticIntentEngine(language)
        else:
            self.intent_engine = intent_engine

        self.feature_extractor = feature_extractor

        self.dataset = None
        self.categories = []

        self.saver = saver

    def save(self, path):
        """
        Save the model.

        :param path: The path to save the model.
        :type path: str.
        """

        if self.saver is None:
            raise ValueError("No saver was provided")
        self.saver.save(self, path)

    @staticmethod
    def load(saver, path) -> "ModelIntentClassifier":
        """
        Load the model.

        :param saver: The saver to use.
        :type saver: Saver.
        :param path: The path to load the model.
        :type path: str.

        :return: The loaded model.
        :rtype: ModelIntentClassifier.
        """

        return saver.load(path)

    def fit(self, dataset_path):
        """
        Fit the intent classifier.

        :param dataset_path: The path to the dataset.
        :type dataset_path: str.
        """

        self.dataset, self.categories = self.load_dataset(dataset_path)

        if self.dataset is None:
            raise ValueError("You must load the dataset first")

        text = self.dataset["text"]
        intent = self.dataset["intent"]

        if self.feature_extractor is not None:
            self.feature_extractor.fit(dataset_path, self.intent_engine.stopwords)

        self.intent_engine.fit(text, intent)

    def multiple_predict(self, texts, threshold=0.2):
        """
        Predict the intent of multiple texts.

        :param texts: The texts to predict.
        :type texts: list.
        :param threshold: The threshold to apply.
        :type threshold: float.

        :return: The predictions.
        :rtype: dict.

        example: multiple_predict(["hello", "how are you"]) ->
            {"intends": [{"text": "hello", "intent": "greet", "probability": 0.9},
                        {"text": "how are you", "intent": "greet", "probability": 0.7}]}

        Can also return the features if a feature extractor is provided:
        example: multiple_predict(["hello", "how are you"]) ->
            {"intends": [{"text": "hello", "intent": "greet", "probability": 0.9,
                        "features": [{"word": "hello", "entity": "greet", "similarities": 1}]},
                        {"text": "how are you", "intent": "greet", "probability": 0.7,
                        "features": [{"word": "how", "entity": "greet", "similarities": 1},
                                    {"word": "are", "entity": "greet", "similarities": 1},
                                    {"word": "you", "entity": "greet", "similarities": 1}]}]}

        """
        intents, probabilities = self.intent_engine.predict(texts)
        results = []
        for intent, probability, text in zip(intents, probabilities, texts):
            results.append({"text": text, "intent": intent, "probability": probability})
        final = {"intends": results}
        if self.feature_extractor is None:
            return final

        for result in final["intends"]:
            result["features"] = self.feature_extractor.get_features(
                result["text"], threshold
            )

        return final

    def predict(self, text, threshold=0.2):
        """
        Predict the intent of a text.

        :param text: The text to predict.
        :type text: str.
        :param threshold: The threshold to apply.
        :type threshold: float.

        :return: The prediction.
        :rtype: dict.
        """
        return self.multiple_predict([text], threshold)
