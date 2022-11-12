import tensorflow as tf

from helpers.load_assets import load_vocab
from helpers.model_helps import encode_data, pad_data


from typing import List

w2i, i2w = load_vocab()

def pre_process_data(sentences: List[str]):
    encoded_docs = encode_data(sentences, w2i)
    padded_docs = pad_data(encoded_docs, w2i)

    return padded_docs


def predict_next_word(sentences: List[str], model):
    padded_docs = pre_process_data(sentences)
    predictions = model(padded_docs)

    return post_process( 
        predictions
    )

def post_process(logits_tensor) -> List[str]:
    word_indices = tf.math.argmax(logits_tensor, 1).numpy()
    # TODO: get key of value, not you're using value, which is index as key
    word_preds = [ i2w.get(ind) for ind in word_indices ]

    return word_preds

