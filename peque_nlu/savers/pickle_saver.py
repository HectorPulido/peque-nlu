"""
The pickle_saver module.
"""
import pickle
from peque_nlu.savers import IntentSaver


class PickleSaver(IntentSaver):
    """
    The PickleSaver class.

    This class is used to save and load the intent engine using pickle.
    """

    def save(self, model, path):
        """
        Save the model to the path.

        :param model: The model to save.
        :type model: object.
        :param path: The path to save the model.
        :type path: str.
        """

        with open(path, "wb") as file:
            pickle.dump(model, file)

    def load(self, path) -> object:
        """
        Load the model from the path.

        :param path: The path to load the model.
        :type path: str.
        :return: The model.
        :rtype: object.
        """

        with open(path, "rb") as file:
            model = pickle.load(file)
        return model
