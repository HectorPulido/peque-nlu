"""
The naive feature extractor module.
"""
from peque_nlu.feature_extractors import FeatureExtractor


class NaiveFeatureExtractor(FeatureExtractor):
    """
    The NaiveFeatureExtractor class.

    This class is used to create a naive feature extractor.
    This works by checking if the examples are in the input text.
    """

    def get_features(self, text_to_decode, threshold):
        """
        Get the features from the input text.

        :param text_to_decode: The input text to decode.
        :type text_to_decode: str.
        :param threshold: The threshold to apply.
        :type threshold: float.
        :return: The features.
        :rtype: list.

        example: get_features("hello", 0.5) ->
            [{"word": "hello", "entity": "greet", "similarities": 1}]
        """
        text_to_decode = self.preprocess_input(text_to_decode)

        matches = []
        for word in text_to_decode:
            for entity, examples in self.entities.items():
                for example in examples:
                    example = example.lower()
                    if example not in word:
                        continue
                    matches.append({"word": word, "entity": entity, "similarities": 1})
        return matches
