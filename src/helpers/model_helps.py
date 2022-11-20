import tensorflow as tf
from typing import List
from pathlib import Path
from unidecode import unidecode
import unicodedata
from src.helpers.load_assets import load_dp_meta

train_metadata = load_dp_meta()
Encoded_Data = List[List[float]]


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

async def validate_inp_text(usr_typ_str: str) -> List[str]:

    if "" == usr_typ_str or " " == usr_typ_str: return []

    if "|" in usr_typ_str:
        return [st.strip() for st in usr_typ_str.split("|")]
        
    
    return [usr_typ_str.strip()]

def get_maxnum_words(words: List[str]) -> List[str]:
    
    max_words = [ ]; ci = 0; 
    spec_chars = {"":0, " ":0 , "\n": 0,"@":0, ".":0,
    "\t": 0, "-": 0, "_": 0, "+": 0, "*": 0, "%":0,
    "&":0, "#":0,"!":0, "?":0, "/": 0}

    wl = len(words)

    while (len(max_words) < train_metadata.get("max_length")
    ) and ci < wl:

        # Identify if current word can be put in end list
        if (words[ci].isalpha()) or (spec_chars.get(words[ci], None) == None):
            max_words.append(words[ci])
        
        ci +=1
    return max_words



def encode_data(sentences: List[str], vocab: dict[str]) ->Encoded_Data:
    encoded_docs = [ ]
    for d in sentences:
        d = normalize_unicode( unidecode(d).replace("'", " ") )

        words = d.split(" ")

        #TODO: Check for empty strings and not run full eval for it
        max_num_words = get_maxnum_words(words)

        encoded_docs.append(
            [ float(vocab.get(w))\
             for w in max_num_words if w !=" " and (
                vocab.get(w) != None) ]
        )

    log_encoded_docs(encoded_docs)

    return encoded_docs

def pad_data(encoded_data: List[int],
 vocab: dict[str]):
    return tf.cast( 
        tf.keras.preprocessing\
        .sequence.pad_sequences(
            encoded_data, maxlen=train_metadata.get("max_length"),
            padding='post' , value = vocab.get('UNK')
        )
    , dtype = tf.float32
    )

def load_model():

    src_path = Path(__file__).parent.parent
    model_path = str(src_path) + "/assets/VSN_NW"

    return tf.saved_model\
        .load(model_path)