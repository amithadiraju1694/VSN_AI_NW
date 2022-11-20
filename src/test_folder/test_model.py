import sys
from pathlib import Path

# root path
root_path = str(
    Path(__file__).parent.parent.parent
)

#Adding root path to where modules are searched for
sys.path.insert(0, root_path )

from src.model.pred_nw import (
    predict_next_word,
    pre_process_data,
    post_process
)

from src.helpers.model_helps import load_model
from src.helpers.load_assets import load_vocab, load_dp_meta

import pytest
import tensorflow as tf
import numpy as np


model_instance = load_model()
w2i, _ = load_vocab()
dp_meta = load_dp_meta()


valid_sent_to_test = ["īśvaro vikramī", "prasann-ātmā viśva-sṛṭ",
    "mahī-bhartā", "agraṇīr", "asaṅkhyeyo 'pramey-ātmā viśiṣṭaḥ",
    "āgrāhyaḥ śāśvataḥ kṛṣṇo", "padma-nābho 'ravindākṣaḥ padma-garbhaḥ śarīra-bhṛt",
    "aja– sarveśvaras siddhas siddhis sarvādir acyutaḥ"]

inval_sent_to_test = ["Hey Mike", "", " ", ":", "@", "123", "abc", "abc.com", "Hi, I am Amith"]

@pytest.mark.asyncio
async def test_model_func():
    
    pred_strings = await predict_next_word(sentences=valid_sent_to_test,
     model = model_instance)
    
    EOS_cnt = 0
    for op in pred_strings:
        if op == 'EOS': EOS_cnt +=1
    
    assert EOS_cnt <= len(valid_sent_to_test)//2,\
     " Next word for more than half of inputs sentences were\
         predicted as 'End of Sentence'. It's likely that model is not generalizing well.\
            Please check training pipeline. "

def test_prepro_typ():
    acceptable_types = [tf.Tensor,
     tf.SparseTensor, tf.IndexedSlices]

    prepro_out = pre_process_data(
        sentences=valid_sent_to_test)
    
    typ_prepro = type(prepro_out)

    assert not typ_prepro in acceptable_types, f"""
    Pre-Processing function returned a type of object which is not
    supported for model training: {typ_prepro}. Check the helper functions
    or inputs used in it.
    """


def test_prepro_def_func():
    """
    Function to test if pre-processing was given all invalid texts
    does it respond with correct outputs.
    """

    pre_out = pre_process_data(
        sentences=inval_sent_to_test).numpy()

    one_def_row = [w2i["UNK"]] * dp_meta["max_length"]

    def_pad_docs = np.repeat([one_def_row], repeats = len(inval_sent_to_test) ,
     axis = 0)
    
    
    assert np.array_equal(pre_out, def_pad_docs)
    
    
