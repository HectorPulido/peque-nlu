from abc import ABC, abstractmethod


class BasicIntentEngine(ABC):
    @abstractmethod
    def predict(self, text):
        pass

    @abstractmethod
    def fit(self, text, intent):
        pass
