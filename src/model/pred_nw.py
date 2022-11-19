import tensorflow as tf

from helpers.load_assets import load_vocab
from helpers.model_helps import encode_data, pad_data


from typing import List

w2i, i2w = load_vocab()

async def pre_process_data(sentences: List[str]):
    encoded_docs = encode_data(sentences, w2i)
    padded_docs = pad_data(encoded_docs, w2i)

    return padded_docs


async def predict_next_word(sentences: List[str], model):
    padded_docs = await pre_process_data(sentences)
    predictions = model(padded_docs)

    return await post_process( 
        predictions
    )

async def post_process(logits_tensor) -> List[str]:
    word_indices = tf.math.argmax(logits_tensor, 1).numpy()
    
    word_preds = [ i2w.get(ind) for ind in word_indices ]

    return word_preds

