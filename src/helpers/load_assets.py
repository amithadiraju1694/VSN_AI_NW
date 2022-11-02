import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))

def load_vocab():
    #/Users/aadir00/Desktop/Amith/VSN_AI_NW/src/helpers/vocabEOS_True.txt
    file_name = os.path.join(here, "vocabEOS_True.txt")
    
    file = open(file_name)
    
    return [line.rstrip('\n'
    ) for line in file
    ]

