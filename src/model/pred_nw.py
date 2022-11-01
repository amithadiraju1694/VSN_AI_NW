import tensorflow as tf
from ..helpers.load_assets import load_vocab
from ..helpers.model_helps import encode_data, pad_data
from typing import List

vocab = load_vocab()

def predict_next_word(sentences: List[str], model):
    encoded_docs = encode_data(sentences, vocab)
    padded_docs = pad_data(encoded_docs, vocab)
    predictions = model(padded_docs)

    return predictions

def post_process(logits_tensor) -> List[str]:
    word_indices = tf.math.argmax(logits_tensor, 1).numpy()
    word_preds = [ vocab[ind] for ind in word_indices ]

    return word_preds

