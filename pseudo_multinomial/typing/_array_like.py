import typing
from typing import *

import numpy as np

__all__ = [
    'Real',
    'VectorLike',
    'MatrixLike',
]
__all__.extend(typing.__all__)

Real = Union[int, float]
sr1 = Sequence[Real]
sr2 = Sequence[sr1]

VectorLike = Union[np.ndarray, sr1]
MatrixLike = Union[np.ndarray, sr2]
