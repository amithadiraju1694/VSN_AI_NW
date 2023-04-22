import re
import unicodedata
from pathlib import Path
from typing import List, Union

import tensorflow as tf
from typeguard import typechecked
from unidecode import unidecode

from src.helpers.load_assets import load_dp_meta, load_vocab

train_metadata = load_dp_meta()
Encoded_Data = List[List[float]]
w2i, _ = load_vocab()


def find_min_word_len(w2i: dict[str, int]) -> int:
    cm = 2**31 - 1
    for k in w2i:
        if k != " " and len(k) > 0:
            cm = min(cm, len(k))
    return cm


min_word_len = find_min_word_len(w2i=w2i)


@typechecked
def normalize_unicode(st: str) -> str:
    """Normalizes input string ( removes dot notation at top or bottom )
    without any strange characters."""
    return unicodedata.normalize("NFD", st)


@typechecked
def isascii(s: str) -> bool:
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())


def log_encoded_docs(encoded_docs: List[List[float]]):
    print("\n No. of encoded documents: ", len(encoded_docs))

    for d in encoded_docs:
        print("\n No.of words encoded in this doc: ", len(d))

        print("\n Encoded document: ", d)

    return


@typechecked
async def validate_inp_text(usr_typ_str: str) -> List[Union[None, str]]:
    usr_typ_str = usr_typ_str.strip()

    if "|" in usr_typ_str:
        return [
            st.strip()
            for st in usr_typ_str.split("|")
            if len(st.strip()) >= min_word_len
        ]

    if "" == usr_typ_str or " " == usr_typ_str or len(usr_typ_str) < min_word_len:
        return []

    return [usr_typ_str]


def get_maxnum_words(words: List[str]) -> List[str]:

    max_words = []
    ci = 0
    spec_chars = {
        "": 0,
        " ": 0,
        "\n": 0,
        "@": 0,
        ".": 0,
        "\t": 0,
        "-": 0,
        "_": 0,
        "+": 0,
        "*": 0,
        "%": 0,
        "&": 0,
        "#": 0,
        "!": 0,
        "?": 0,
        "/": 0,
    }

    wl = len(words)

    while (len(max_words) < train_metadata.get("max_length")) and ci < wl:

        if len(re.findall("[0-9]+", words[ci])) > 0:
            ci += 1
            continue

        # Identify if current word can be put in end list
        if words[ci].isalpha() or (spec_chars.get(words[ci], None) == None):
            max_words.append(words[ci])

        ci += 1
    return max_words


@typechecked
def encode_data(sentences: List[str], vocab: dict[str, int]) -> Encoded_Data:
    encoded_docs = []
    for d in sentences:
        d = normalize_unicode(unidecode(d).replace("'", " "))

        words = d.split(" ")

        max_num_words = get_maxnum_words(words)

        encoded_docs.append(
            [
                float(vocab.get(w))
                for w in max_num_words
                if w != " " and (vocab.get(w) != None)
            ]
        )

    log_encoded_docs(encoded_docs)

    return encoded_docs


@typechecked
def pad_data(encoded_data: Encoded_Data, vocab: dict[str, int]):
    return tf.cast(
        tf.keras.preprocessing.sequence.pad_sequences(
            encoded_data,
            maxlen=train_metadata.get("max_length"),
            padding="post",
            value=vocab.get("UNK"),
        ),
        dtype=tf.float32,
    )


def load_model():

    src_path = Path(__file__).parent.parent
    model_path = str(src_path) + "/assets/VSN_NW"

    return tf.saved_model.load(model_path)
