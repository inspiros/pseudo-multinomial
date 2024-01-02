import numpy as np

__all__ = ['Chain']


class Chain:
    def exit_probability(self, k: int) -> float:
        ...

    def expectation(self) -> float:
        ...

    def n_states(self) -> float:
        ...

    def is_finite(self) -> bool:
        ...

    def transition_matrix(self, n: int = 0) -> np.ndarray:
        ...

    def __len__(self):
        ...

    def __getattr__(self, item):
        ...
