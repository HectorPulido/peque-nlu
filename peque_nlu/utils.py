import json
import pandas as pd


class IntentUtils:
    json_dataset = None

    def _load_json_dataset(self, dataset_path):
        # if property dataset exists load it
        if hasattr(self, "json_dataset") and self.json_dataset is not None:
            return self.json_dataset

        with open(dataset_path, encoding="utf-8") as file:
            self.json_dataset = json.load(file)
        return self.json_dataset

    def _build_dataset(self, dataset):
        # Create a pandas dataframe with the columns: text and intent
        temp_dataset = []
        for intend, examples in dataset["intents"].items():
            for example in examples:
                temp_dataset.append({"text": example, "intent": intend})

        dataframe = pd.DataFrame(temp_dataset, columns=["text", "intent"])

        categories = list(dataset["intents"].keys())
        return dataframe, categories

    def load_dataset(self, dataset_path):
        dataset = self._load_json_dataset(dataset_path)
        dataframe, categories = self._build_dataset(dataset)
        return dataframe, categories

    def get_entities(self, dataset_path):
        json_dataset = self._load_json_dataset(dataset_path)
        return json_dataset["entities"]
