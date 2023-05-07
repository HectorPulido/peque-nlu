import pickle
from peque_nlu.savers import IntentSaver


class PickleSaver(IntentSaver):
    def save(self, model, path):
        with open(path, "wb") as file:
            pickle.dump(model, file)

    def load(self, path):
        with open(path, "rb") as file:
            model = pickle.load(file)
        return model
