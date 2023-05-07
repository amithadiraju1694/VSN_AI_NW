from typing import List, Union

import numpy as np
from typeguard import typechecked

from vsn_nw.helpers.load_assets import load_dp_meta, load_vocab
from vsn_nw.helpers.model_helps import encode_data, pad_data

w2i, i2w = load_vocab()
dp_meta = load_dp_meta()


@typechecked
def pre_process_data(
    sentences: List[str],
):
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
    
    # Get numpy array of arrays with padding for each doc
    padded_docs = pre_process_data(sentences)
    _, ncols = padded_docs.shape

    all_unk = np.array([w2i["UNK"]] * ncols, dtype=np.int32)

    predictions = []

    # Iterate through each padded document and run inference
    for doc in padded_docs:
        if np.array_equal(doc, all_unk):
            predictions.append([w2i["UNK"]])
        else:
            predictions.append(
                model.run(['dense'],
                       {"input": doc.reshape(-1,dp_meta["max_length"])}
                     )[0]
                )
    postpro_predictions = await post_process(predictions)

    return postpro_predictions


@typechecked
async def post_process(logits_tensor: List[np.ndarray] ) -> List[str]:

    word_indices = np.zeros(shape=(len(logits_tensor),))

    for ind, tensor in enumerate(logits_tensor):
        if isinstance(tensor, list):
            word_indices[ind] = tensor[0]
        else:
            word_indices[ind] = np.argmax(tensor, axis=1)[0]

    word_preds = [i2w.get(ind, None) for ind in word_indices]

    return word_preds
