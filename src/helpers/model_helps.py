import tensorflow as tf
from typing import List
from pathlib import Path

def encode_data(sentences: List[str], vocab: List[str]):
    encoded_docs = [ ]
    for d in sentences:
        encoded_docs.append(
            [ float(
                vocab.index(w)
             ) for w in d.split(" ") 
        ]
        )
    return encoded_docs

def pad_data(encoded_data: List[int],
 vocab: List[str]):
    return tf.cast( 
        tf.keras.preprocessing\
        .sequence.pad_sequences(
            encoded_data, maxlen=8,
            padding='post' , value = vocab.index('UNK')
        )
    , dtype = tf.float32
    )


def load_model():

    src_path = Path(__file__).parent.parent
    model_path = str(src_path) + "/model/VSN_NW"

    return tf.saved_model\
        .load(model_path)