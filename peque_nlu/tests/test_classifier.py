"""
Test the classifier module.
"""
from peque_nlu.utils import glove_load
from peque_nlu.intent_engines import (
    SGDIntentEngine,
    LogisticIntentEngine,
    WorldVectorIntentEngine,
)
from peque_nlu.feature_extractors import (
    NaiveFeatureExtractor,
    GloveFeatureExtractor,
)

from peque_nlu.intent_classifiers import ModelIntentClassifier


SMALL_TALK_QUERY = "Hola como te encuentras?"
DATASET_PATH = "intents_example.json"

glove_vectors = None


def get_glove_vector():
    """
    Get the glove vectors.
    """
    global glove_vectors

    if glove_vectors is not None:
        return glove_vectors

    glove_vectors = glove_load("glove-twitter-25")
    return glove_vectors


def test_sgd_intent_engine_multiple():
    """
    Test the SGDIntentEngine class multi prediction.
    """
    intent_engine = SGDIntentEngine("spanish")
    model = ModelIntentClassifier("spanish", intent_engine)
    model.fit(DATASET_PATH)

    prediction = model.multiple_predict(
        [
            SMALL_TALK_QUERY,
            "Quiero aprender sobre lo último de python",
            "describeme usando un meme",
        ]
    )

    assert len(prediction) == 3

    first_prediction = prediction[0]
    assert "intent" in first_prediction
    assert "probability" in first_prediction
    assert "text" in first_prediction
    assert "features" not in first_prediction

    assert first_prediction["intent"] == "small_talk"


def test_logistic_intent_engine_multiple():
    """
    Test the LogisticIntentEngine class multi prediction.
    """
    intent_engine = LogisticIntentEngine("spanish")
    model = ModelIntentClassifier("spanish", intent_engine)
    model.fit(DATASET_PATH)

    prediction = model.multiple_predict(
        [
            SMALL_TALK_QUERY,
            "Quiero aprender sobre lo último de python",
            "describeme usando un meme",
        ]
    )

    assert len(prediction) == 3

    first_prediction = prediction[0]
    assert "intent" in first_prediction
    assert "probability" in first_prediction
    assert "text" in first_prediction
    assert "features" not in first_prediction

    assert first_prediction["intent"] == "small_talk"


def test_sgd_intent_engine_single():
    """
    Test the SGDIntentEngine class single prediction.
    """
    intent_engine = SGDIntentEngine("spanish")
    model = ModelIntentClassifier("spanish", intent_engine)
    model.fit(DATASET_PATH)

    prediction = model.predict(SMALL_TALK_QUERY)

    assert "text" in prediction
    assert "intent" in prediction
    assert "probability" in prediction

    assert prediction["intent"] == "small_talk"


def test_logistic_intent_engine_single():
    """
    Test the LogisticIntentEngine class single prediction.
    """
    intent_engine = LogisticIntentEngine("spanish")
    model = ModelIntentClassifier("spanish", intent_engine)
    model.fit(DATASET_PATH)

    prediction = model.predict(SMALL_TALK_QUERY)

    assert "text" in prediction
    assert "intent" in prediction
    assert "probability" in prediction

    assert prediction["intent"] == "small_talk"


def test_world_vector_intent_engine():
    """
    Test the WorldVectorIntentEngine class.
    """
    intent_engine = WorldVectorIntentEngine("spanish", get_glove_vector())
    model = ModelIntentClassifier("spanish", intent_engine)
    model.fit(DATASET_PATH)

    prediction = model.multiple_predict(
        [
            SMALL_TALK_QUERY,
            "Quiero aprender sobre lo último de python",
            "describeme usando un meme",
        ]
    )

    assert len(prediction) == 3

    first_prediction = prediction[0]
    assert "intent" in first_prediction
    assert "probability" in first_prediction
    assert "text" in first_prediction
    assert "features" not in first_prediction

    assert first_prediction["intent"] == "small_talk"


def test_glove_feature_extractor_multiple():
    """
    Test the GloveFeatureExtractor class, multiple prediction.
    """
    intent_engine = SGDIntentEngine("spanish")
    feature_extractor = GloveFeatureExtractor(get_glove_vector())
    model = ModelIntentClassifier("spanish", intent_engine, feature_extractor)
    model.fit(DATASET_PATH)

    prediction = model.multiple_predict(
        [
            SMALL_TALK_QUERY,
            "Quiero aprender sobre lo último de python",
            "sabes como puedo aprender sobre angular?",
        ],
        threshold={
            "timing": 0.5,
            "technology": 0.2,
        },
    )

    assert len(prediction) == 3

    first_prediction = prediction[0]
    assert first_prediction["intent"] == "small_talk"
    assert "features" in first_prediction
    assert len(first_prediction["features"]) == 0

    second_prediction = prediction[1]
    assert second_prediction["intent"] == "search"
    assert "features" in first_prediction
    assert len(second_prediction["features"]) == 2

    feature_one = second_prediction["features"][0]
    assert "word" in feature_one
    assert "entity" in feature_one
    assert "similarities" in feature_one
    assert feature_one["word"] == "ultimo"
    assert feature_one["entity"] == "timing"
    assert feature_one["similarities"] > 0.5

    feature_two = second_prediction["features"][1]
    assert "word" in feature_two
    assert "entity" in feature_two
    assert "similarities" in feature_two
    assert feature_two["word"] == "python"
    assert feature_two["entity"] == "technology"
    assert feature_two["similarities"] == 1


def test_glove_feature_extractor_single():
    """
    Test the GloveFeatureExtractor class, single prediction.
    """
    intent_engine = SGDIntentEngine("spanish")
    feature_extractor = GloveFeatureExtractor(get_glove_vector())
    model = ModelIntentClassifier("spanish", intent_engine, feature_extractor)
    model.fit(DATASET_PATH)

    prediction = model.predict(
        "quiero conocer el ultimo blogpost de unity",
        threshold={
            "timing": 0.5,
            "technology": 0.2,
        },
    )

    assert "features" in prediction

    """
    This feature extractor performs better than the naive one, 
    recognize the word "unity" as technology.
    """
    assert len(prediction["features"]) == 2

    feature_one = prediction["features"][0]
    assert "word" in feature_one
    assert "entity" in feature_one
    assert "similarities" in feature_one
    assert feature_one["word"] == "ultimo"
    assert feature_one["entity"] == "timing"
    assert feature_one["similarities"] > 0.5

    feature_two = prediction["features"][1]
    assert "word" in feature_two
    assert "entity" in feature_two
    assert "similarities" in feature_two
    assert feature_two["word"] == "unity"
    assert feature_two["entity"] == "technology"
    assert feature_two["similarities"] > 0.2


def test_naive_feature_extractor_multiple():
    """
    Test the NaiveFeatureExtractor class, multiple prediction.
    """
    intent_engine = SGDIntentEngine("spanish")
    feature_extractor = NaiveFeatureExtractor()
    model = ModelIntentClassifier("spanish", intent_engine, feature_extractor)
    model.fit(DATASET_PATH)

    prediction = model.multiple_predict(
        [
            SMALL_TALK_QUERY,
            "Quiero aprender sobre lo último de python",
            "sabes como puedo aprender sobre angular?",
        ]
    )

    assert len(prediction) == 3

    first_prediction = prediction[0]
    assert first_prediction["intent"] == "small_talk"
    assert "features" in first_prediction

    second_prediction = prediction[1]
    assert second_prediction["intent"] == "search"
    assert "features" in first_prediction
    assert len(second_prediction["features"]) == 2
    feature_one = second_prediction["features"][0]
    assert "word" in feature_one
    assert "entity" in feature_one
    assert "similarities" in feature_one

    assert feature_one["word"] == "ultimo"
    assert feature_one["entity"] == "timing"
    assert feature_one["similarities"] == 1


def test_naive_feature_extractor_single():
    """
    Test the NaiveFeatureExtractor single class, single prediction.
    """
    intent_engine = SGDIntentEngine("spanish")
    feature_extractor = NaiveFeatureExtractor()
    model = ModelIntentClassifier("spanish", intent_engine, feature_extractor)
    model.fit(DATASET_PATH)

    prediction = model.predict("quiero conocer el ultimo blogpost de unity")
    assert "features" in prediction
    assert len(prediction["features"]) == 1

    feature_one = prediction["features"][0]
    assert "word" in feature_one
    assert "entity" in feature_one
    assert "similarities" in feature_one
    assert feature_one["word"] == "ultimo"
    assert feature_one["entity"] == "timing"
    assert feature_one["similarities"] == 1
