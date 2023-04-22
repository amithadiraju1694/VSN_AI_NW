import sys
from pathlib import Path

# root path
root_path = str(
    Path(__file__).parent.parent.parent
)

#Adding root path to where modules are searched for
sys.path.insert(0, root_path )

import numpy as np

from src.helpers.model_helps import (
    encode_data, 
    get_maxnum_words,
    validate_inp_text,
    min_word_len
)

from src.test_folder.test_model import (
    valid_sent_to_test,
    inval_sent_to_test,
    dp_meta,
    w2i
)

import pytest

def generate_mx_sents():

    max_dum = ["DUMMY"] * dp_meta["max_length"]

    dum_st = " ".join( max_dum )
    mid_ind = dp_meta["max_length"] // 2
    if dp_meta["max_length"] % 2 != 0:
        mid_ind+=1

    return ["123 456 "+dum_st,
        " ".join( max_dum[: mid_ind] ) + " 123 456 " + " ".join(max_dum[mid_ind: ]),
        dum_st + " 123 456"]

mx_sents = generate_mx_sents()


def test_enc_out():
    
    """
    Function to test if pre-process has at least one
    default padding row, when invalid sentences are given to pre-process
    """
    # output of predict_nextword, should have list with index of "UNK"
    # in those specific indices
    val_half = np.random.choice(valid_sent_to_test, size = 3,
    replace = True).tolist()
    inv_half = np.random.choice(inval_sent_to_test,size=3,
     replace = False ).tolist()

    val_half.extend(inv_half)

    encoded_docs = encode_data(sentences=val_half, vocab = 
    w2i)

    for i in range(3,6):
        assert len(encoded_docs[i]) == 0, f"""
        An ivalid document was given to encode in this index,
        which should have returned empty list, got {len(encoded_docs[i])}
        encoded items in this list instead.
        """

def test_maxnum_words():
    more_mx_len = valid_sent_to_test[-3: ]

    for sent in more_mx_len:
        max_words = get_maxnum_words(words = sent.replace("'", " ")\
            .split(" ") )
        
        assert len(max_words) <= dp_meta["max_length"],"""
        Some sentences are exceeding allowed max length of model.
        Check `get_maxnum_words` of helpers and data preparation process.
        """

def test_maxnum_mix_words():

    for ind, s in enumerate(mx_sents):
        sent_words = s.split(" ")
        max_words = get_maxnum_words(words = sent_words)

        assert len(max_words) == dp_meta["max_length"],"""  
         """

        alnum_st = sent_words.index("123")
        alnum_end = sent_words.index("456")

        if ind == 0:
            assert max_words[0].isalpha() and max_words[1
            ].isalpha(),""" Helper 
            `get_maxnum_words` did not replace invalid characters from input string.
            Check the filtering logic.  """

            assert max_words[0] == sent_words[alnum_end+1], """"""

        elif ind == 1:

            assert max_words[alnum_st].isalpha() and max_words[alnum_end].isalpha(),"""
               """
        
            assert max_words[alnum_st] == sent_words[alnum_end+1], """ """

            assert max_words[alnum_st-1] == sent_words[alnum_st-1], """ """
        
        else: 

            assert max_words[-1] == sent_words[alnum_st-1], """"""
            assert max_words[-2] == sent_words[alnum_st-2], """"""

@pytest.mark.asyncio
async def test_val_inp():
    sents = inval_sent_to_test[1:5]

    # All these sentences must return empty strings
    for s in sents:
        lis = await validate_inp_text(
            usr_typ_str=s
        )

        assert len(lis) == 0, """
        Invalid sentences are also considered valid.
        Check `validate_inp_text` from model helper functions."""
