
import sys
from pathlib import Path

# root path
root_path = str(
    Path(__file__).parent.parent.parent
)

#Adding root path to where modules are searched for
sys.path.insert(0, root_path )


from src.helpers.model_helps import (
    encode_data, 
    load_dp_meta,
    get_maxnum_words,
    validate_inp_text
)

import pytest