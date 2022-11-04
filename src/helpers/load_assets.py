import numpy as np
import os
import re

here = os.path.dirname(os.path.abspath(__file__))

def load_vocab():
    file_name = os.path.join(here, "vocabEOS_True.txt")
    
    file = open(file_name)
    return [ re.sub("\n", "", line).strip() for line in file.readlines(
        
    )]

