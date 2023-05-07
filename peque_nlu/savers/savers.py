from abc import ABC, abstractmethod


class IntentSaver(ABC):
    @abstractmethod
    def save(self, model, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
