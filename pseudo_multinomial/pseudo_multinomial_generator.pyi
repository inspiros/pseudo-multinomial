from typing import List, Sequence, Optional, Union, Tuple

import numpy as np

from ._types import VectorLike, MatrixLike
from .chains import Chain

__all__ = [
    'PseudoMultinomialGenerator',
]


class PseudoMultinomialGenerator:
    chains: List[Chain]
    n_chains: int
    S: np.ndarray

    def __init__(self,
                 chains: Sequence[Chain],
                 chain_transition_matrix: MatrixLike,
                 random_state: Optional[Union[np.random.RandomState, int]] = None):
        ...

    def set_random_state(self, random_state: Optional[Union[np.random.RandomState, int]] = None) -> None:
        ...

    def state(self) -> Tuple[int, int]:
        ...

    def set_state(self, state: int, chain_state: int):
        ...

    def random_init(self, state: int = -1, max_chain_state: int = 1000):
        ...

    def next_states(self, n: int = 1) -> np.ndarray:
        ...

    def n_states(self) -> np.ndarray:
        ...

    def entrance_stationaries(self) -> np.ndarray:
        ...

    def expectations(self) -> np.ndarray:
        ...

    def probs(self) -> np.ndarray:
        ...

    def is_finite(self) -> bool:
        ...

    def chain_transition_matrix(self) -> np.ndarray:
        ...

    def transition_matrix(self, n: int = 0) -> np.ndarray:
        ...

    def __len__(self) -> int:
        ...

    def __next__(self) -> int:
        ...

    def __getitem__(self, item) -> Chain:
        ...

    def __repr__(self) -> str:
        ...

    def generate(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        ...

    @staticmethod
    def from_pvals(chains: Sequence[Chain],
                   pvals: Optional[VectorLike] = None,
                   repeat: Union[bool, Sequence[bool], np.ndarray] = True,
                   random_state: Optional[Union[np.random.RandomState, int]] = None) -> 'PseudoMultinomialGenerator':
        ...


def pseudo_multinomial(chain: Union[PseudoMultinomialGenerator, Sequence[Chain], Chain],
                       size: Optional[int] = None,
                       random_init: bool = True,
                       **kwargs) -> Union[int, np.ndarray]:
    ...
