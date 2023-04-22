from typing import List, Union

import numpy as np
import tensorflow as tf
from typeguard import typechecked

from src.helpers.load_assets import load_dp_meta, load_vocab
from src.helpers.model_helps import encode_data, pad_data

w2i, i2w = load_vocab()
dp_meta = load_dp_meta()


@typechecked
def pre_process_data(
    sentences: List[str],
) -> Union[tf.Tensor, tf.SparseTensor, tf.IndexedSlices]:
    """Pre-processes data by encoding string to indicies from vocabulary
    and pads them to a fixed length.

    Args:
        sentences: List of strings containing input text.

    Returns:
        padded_docs: tf.Tensor/SparseTensor containing padding indices up-to max-length
        for each sentence.
    """
    encoded_docs = encode_data(sentences, w2i)
    padded_docs = pad_data(encoded_docs, w2i)

    return padded_docs


@typechecked
async def predict_next_word(sentences: List[str], model):
    padded_docs = pre_process_data(sentences)
    _, ncols = padded_docs.shape

    all_unk = np.array([w2i["UNK"]] * ncols, dtype=np.int32)

    predictions = []
    for doc in padded_docs:
        if np.array_equal(doc, all_unk):
            predictions.append([w2i["UNK"]])
        else:
            predictions.append(model(np.array(doc).reshape(-1, dp_meta["max_length"])))
    postpro_predictions = await post_process(predictions)

    return postpro_predictions


@typechecked
async def post_process(logits_tensor: List[Union[List, tf.Tensor]]) -> List[str]:

    word_indices = np.zeros(shape=(len(logits_tensor),))
    for ind, tensor in enumerate(logits_tensor):
        if isinstance(tensor, list):
            word_indices[ind] = tensor[0]
        else:
            # TODO: Check for shapes of this single tensor
            # and validate argmax after that
            word_indices[ind] = tf.math.argmax(tensor, axis=1).numpy()[0]

    word_preds = [i2w.get(ind, None) for ind in word_indices]

    return word_preds
