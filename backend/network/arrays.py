"""Types des tableaux NumPy manipulés par le réseau."""

import numpy as np
from numpy.typing import NDArray

type Matrix = NDArray[np.float64]
type Labels = NDArray[np.int64]
