from typing import List, Sequence, Optional, Union, Tuple

import numpy as np

from .chains import Chain
from .typing import *

__all__ = [
    'MasterChain',
]


class MasterChain:
    chains: List[Chain]
    n_chains: int
    S: np.ndarray

    def __init__(self,
                 chains: Sequence[Chain],
                 chain_transition_matrix: MatrixLike,
                 random_state: Optional[np.random.RandomState] = None):
        pass

    def set_seed(self, seed: Optional[int] = None) -> None:
        pass

    def state(self) -> Tuple[int, int]:
        pass

    def set_state(self, state: int, chain_state: int):
        pass

    def random_init(self, state: int = -1, max_chain_state: int = 1000):
        pass

    def next_states(self, n: int = 1) -> np.ndarray:
        pass

    def n_states(self) -> np.ndarray:
        pass

    def entrance_stationaries(self) -> np.ndarray:
        pass

    def expectations(self) -> np.ndarray:
        pass

    def probs(self) -> np.ndarray:
        pass

    def is_finite(self) -> bool:
        pass

    def chain_transition_matrix(self) -> np.ndarray:
        pass

    def transition_matrix(self, n: int = 0) -> np.ndarray:
        pass

    def __len__(self) -> int:
        pass

    def __next__(self) -> int:
        pass

    def __getitem__(self, item) -> Chain:
        pass

    def __repr__(self) -> str:
        pass

    def pseudo_multinomial(self, n: int = 1, random_init: bool = True) -> Union[int, np.ndarray]:
        pass

    @staticmethod
    def from_pvals(chains: Sequence[Chain],
                   pvals: Optional[Union[VectorLike, float]] = None,
                   repeat: Union[bool, Sequence[bool], np.ndarray] = True,
                   random_state: Optional[np.random.RandomState] = None) -> 'MasterChain':
        pass
