import os
import random

import numpy as np
import tensorflow as tf

from .spec_model import SpecModel
from .spec_prep import SpecPrep
from .spec_vi import SpecVI


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    tf.compat.v2.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
