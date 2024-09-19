import os
import random

import numpy as np
import tensorflow as tf
from ..logging import logger


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    tf.compat.v2.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.debug(f"Seed set to {seed}")
