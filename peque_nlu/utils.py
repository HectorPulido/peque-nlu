"""
Utils module.
"""

import json
import pandas as pd
import gensim.downloader as gd
from gensim.models.keyedvectors import KeyedVectors


class IntentUtils:
    """
    The IntentUtils class.

    This class is to centralize the intent utils.
    """

    json_dataset = None

    def _load_json_dataset(self, dataset_path) -> dict:
        """
        Load the json dataset from the path.

        :param dataset_path: The path to load the json dataset.
        :type dataset_path: str.
        :return: The json dataset.
        :rtype: dict.
        """

        # if property dataset exists load it
        if hasattr(self, "json_dataset") and self.json_dataset is not None:
            return self.json_dataset

        with open(dataset_path, encoding="utf-8") as file:
            self.json_dataset = json.load(file)
        return self.json_dataset

    def _build_dataset(self, dataset) -> tuple:
        """
        Build the dataset from the json dataset.

        :param dataset: The json dataset.
        :type dataset: dict.
        :return: The pandas dataframe and the categories.
        :rtype: tuple.
        """

        # Create a pandas dataframe with the columns: text and intent
        temp_dataset = []
        for intend, examples in dataset["intents"].items():
            for example in examples:
                temp_dataset.append({"text": example, "intent": intend})

        dataframe = pd.DataFrame(temp_dataset, columns=["text", "intent"])

        categories = list(dataset["intents"].keys())
        return dataframe, categories

    def load_dataset(self, dataset_path) -> tuple:
        """
        Load the dataset from the path.

        :param dataset_path: The path to load the dataset.
        :type dataset_path: str.
        :return: The pandas dataframe and the categories.
        :rtype: tuple.
        """

        dataset = self._load_json_dataset(dataset_path)
        dataframe, categories = self._build_dataset(dataset)
        return dataframe, categories

    def get_entities(self, dataset_path):
        """
        Get the entities from the dataset.

        :param dataset_path: The path to load the dataset.
        :type dataset_path: str.
        :return: The entities.
        :rtype: list.
        """

        json_dataset = self._load_json_dataset(dataset_path)
        return json_dataset["entities"]


def glove_load(gensim_model) -> KeyedVectors:
    """
    Load the glove vectors from the gensim model.

    :param gensim_model: The gensim model.
    :type gensim_model: str.
    :return: The glove vectors.
    :rtype: KeyedVectors.
    """

    if isinstance(gensim_model, str):
        return gd.load(gensim_model)

    if isinstance(gensim_model, KeyedVectors):
        return gensim_model

    raise ValueError("gensim_model must be a model_name (str) or a KeyedVectors object")
