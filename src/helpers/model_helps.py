import tensorflow as tf
from typing import List
from pathlib import Path
import unicodedata

def normalize_unicode(st: str) -> str:
    return unicodedata.normalize('NFD', st)

def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())

def log_encoded_docs(encoded_docs):
    print("\n No. of encoded documents: ", len(encoded_docs))

    for d in encoded_docs:
        print("\n No.of words encoded in this doc: ", len(d))

        print("\n Encoded document: ", d)
    
    return

def validate_inp_text(usr_typ_str: str) -> List[str]:

    if "" == usr_typ_str or " " == usr_typ_str: return []

    if "|" in usr_typ_str:
        return [st.strip() for st in usr_typ_str.split("|")]
        
    
    return [usr_typ_str.strip()]



def encode_data(sentences: List[str], vocab: List[str]):
    encoded_docs = [ ]
    for d in sentences:
        cur_enc = [ ]
        for w in d.split(" "):
            if isascii(w):
                pass
            else:
                w = normalize_unicode(w)
            cur_enc.append(
            float( vocab.index( w ) ) 
            )
        
        encoded_docs.append(cur_enc)

    log_encoded_docs(encoded_docs)

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