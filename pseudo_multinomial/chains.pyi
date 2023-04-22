import numpy as np

__all__ = ['Chain']


class Chain:
    def exit_probability(self, k: int) -> float:
        pass

    def expectation(self) -> float:
        pass

    def n_states(self) -> float:
        pass

    def is_finite(self) -> bool:
        pass

    def transition_matrix(self, n: int = 0) -> np.ndarray:
        pass

    def __len__(self):
        pass

    def __getattr__(self, item):
        pass
