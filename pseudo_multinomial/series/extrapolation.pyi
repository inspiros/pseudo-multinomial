from typing import Sequence, Optional

import numpy as np

__all__ = [
    'wynn_eps',
]


def wynn_eps(sn: Sequence[float],
             r: Optional[int] = None,
             randomized: bool = False) -> np.ndarray:
    ...
