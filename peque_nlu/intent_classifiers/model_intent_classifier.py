from peque_nlu.intent_classifiers import IntentClassifier
from peque_nlu.utils import IntentUtils

from peque_nlu.intent_engines import LogisticIntentEngine


class ModelIntentClassifier(IntentClassifier, IntentUtils):
    def __init__(
        self, language, intent_engine=None, feature_extractor=None, saver=None
    ):
        if intent_engine is None:
            self.intent_engine = LogisticIntentEngine(language)
        else:
            self.intent_engine = intent_engine

        self.feature_extractor = feature_extractor

        self.dataset = None
        self.categories = []

        self.saver = saver

    def save(self, path):
        if self.saver is None:
            raise ValueError("No saver was provided")
        self.saver.save(self, path)

    @staticmethod
    def load(saver, path):
        return saver.load(path)

    def fit(self, dataset_path):
        self.dataset, self.categories = self.load_dataset(dataset_path)

        if self.dataset is None:
            raise ValueError("You must load the dataset first")

        text = self.dataset["text"]
        intent = self.dataset["intent"]

        if self.feature_extractor is not None:
            self.feature_extractor.fit(dataset_path, self.intent_engine.stopwords)

        self.intent_engine.fit(text, intent)

    def multiple_predict(self, texts, threshold=0.2):
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
        return self.multiple_predict([text], threshold)
