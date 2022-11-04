import tensorflow as tf

from helpers.load_assets import load_vocab
from helpers.model_helps import encode_data, pad_data


from typing import List

vocab = load_vocab()

def pre_process_data(sentences: List[str]):
    encoded_docs = encode_data(sentences, vocab)
    padded_docs = pad_data(encoded_docs, vocab)

    return padded_docs


def predict_next_word(sentences: List[str], model):
    padded_docs = pre_process_data(sentences)
    predictions = model(padded_docs)

    return post_process( 
        predictions
    )

def post_process(logits_tensor) -> List[str]:
    word_indices = tf.math.argmax(logits_tensor, 1).numpy()
    word_preds = [ vocab[ind] for ind in word_indices ]

    return word_preds

