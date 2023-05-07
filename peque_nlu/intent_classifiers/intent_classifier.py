from abc import ABC, abstractmethod


class IntentClassifier(ABC):
    @abstractmethod
    def predict(self, text):
        pass

    @abstractmethod
    def multiple_predict(self, texts):
        ...

    @abstractmethod
    def fit(self, dataset_path):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load(saver, path):
        pass
