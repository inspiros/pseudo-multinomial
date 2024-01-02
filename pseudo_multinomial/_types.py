from typing import Sequence, Union

import numpy as np

__all__ = [
    'Real',
    'VectorLike',
    'MatrixLike',
]

Real = Union[int, float]
sr1 = Sequence[Real]
sr2 = Sequence[sr1]

VectorLike = Union[np.ndarray, sr1]
MatrixLike = Union[np.ndarray, sr2]
