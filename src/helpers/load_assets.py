import os
import re
from pathlib import Path
import json


# src_path
src_path = str(
    Path(__file__).parent.parent
)

def load_vocab():
    
    vocab_path = src_path+ "/assets/vocabEOS_True.txt"
    
    file = open(vocab_path)
    vocab_list = [ re.sub("\n", "", line).strip(

    ) for line in file.readlines()]
    
    w2i , i2w = {}, {}
    for ind, word in enumerate(vocab_list):
        w2i[word] = ind
        i2w[ind] = word
    
    return (w2i, i2w)


def load_dp_meta():

    metadata_path = src_path+"/assets/dataprep_metadata.json"
    print(metadata_path)
    with open(metadata_path, 'r') as mdf:
        jobj = json.load(mdf)
    
    return jobj
    


