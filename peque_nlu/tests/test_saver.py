"""
Test the saver module.
"""
import os
from peque_nlu.intent_engines import SGDIntentEngine
from peque_nlu.savers import PickleSaver

PICKLE_PATH = "test.pkl"


def test_save_pickle():
    """
    Test the save pickle function.
    """
    intent_engine = SGDIntentEngine("spanish")
    saver = PickleSaver()
    saver.save(intent_engine, PICKLE_PATH)
    assert os.path.exists(PICKLE_PATH)
    os.remove(PICKLE_PATH)


def test_load_pickle():
    """
    Test the load pickle function.
    """
    intent_engine = SGDIntentEngine("spanish")
    saver = PickleSaver()
    saver.save(intent_engine, PICKLE_PATH)
    intent_engine = saver.load(PICKLE_PATH)
    assert intent_engine is not None
    os.remove(PICKLE_PATH)
