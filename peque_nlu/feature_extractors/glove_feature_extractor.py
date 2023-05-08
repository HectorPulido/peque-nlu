"""
The GloveFeatureExtractor class module.
"""
from peque_nlu.utils import glove_load
from peque_nlu.feature_extractors import FeatureExtractor


class GloveFeatureExtractor(FeatureExtractor):
    """
    The GloveFeatureExtractor class.

    This class is used to create a glove feature extractor.
    This works by checking using the glove vectors, to check if there are
    similarities between the input text and the examples.
    """

    def __init__(self, gensim_model=None):
        """
        Initialize the GloveFeatureExtractor.

        :param gensim_model: The gensim model to use.
            It can be a model name (str) or a KeyedVectors object.
        :type gensim_model: str or KeyedVectors.
        """

        self.entities = {}
        self.stopwords = None

        self.glove_vectors = glove_load(gensim_model)

    def _check_examples(self, examples) -> list:
        """
        Check if the examples are in the glove vectors.

        :param examples: The examples to check.
        :type examples: list.
        :return: The examples that are in the glove vectors.
        :rtype: list.
        """

        for word in examples:
            if not self._word_is_in_glove(word):
                examples.remove(word)
        return examples

    def _word_is_in_glove(self, word) -> bool:
        """
        Check if the word is in the glove vectors.

        :param word: The word to check.
        :type word: str.
        :return: True if the word is in the glove vectors, False otherwise.
        :rtype: bool.
        """
        return word in self.glove_vectors.key_to_index

    def get_features(self, text_to_decode, threshold):
        """
        Fit the feature extractor.

        :param dataset_path: The path of the dataset.
        :type dataset_path: str.
        :param threshold: The threshold to apply.
        :type threshold: float or dict.
        :return: The features.
        :rtype: list.

        example: get_features("hello", 0.5) ->
            [{"word": "hello", "entity": "greet", "similarities": 1}]

        example: get_features("hello", {"greet": 0.5}) ->
            [{"word": "hello", "entity": "greet", "similarities": 1}]

        """

        text_to_decode = self.preprocess_input(text_to_decode)

        matches = []
        for word in text_to_decode:
            for entity, examples in self.entities.items():
                if word in examples:
                    matches.append({"word": word, "entity": entity, "similarities": 1})
                    continue

                if not self._word_is_in_glove(word):
                    continue

                examples = self._check_examples(examples)
                similarities = self.glove_vectors.similarity(examples, word)

                similarity = max(similarities)

                temp_threshold = None
                if isinstance(threshold, dict):
                    temp_threshold = threshold.get(entity, 0.5)
                elif isinstance(threshold, float):
                    temp_threshold = threshold
                else:
                    raise ValueError(
                        "The threshold must be a float or a dict, "
                        f"not {type(threshold)}"
                    )

                if similarity > temp_threshold:
                    matches.append(
                        {"word": word, "entity": entity, "similarities": similarity}
                    )
        return matches
