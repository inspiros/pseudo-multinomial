import cyroot
import numpy as np

from .typing import Callable, Optional, Union, VectorLike

__all__ = ['RootFindingSolver']

ObjectiveFuncType = Union[
    Callable[[], float], Callable[[float], float],
    Callable[[], VectorLike], Callable[[VectorLike], VectorLike]
]


class RootFindingSolver:
    def __init__(self,
                 objective_fn: ObjectiveFuncType,
                 objective_val: Union[float, VectorLike] = 0.,
                 update_param_fn: Optional[Callable] = None):
        if not isinstance(objective_val, float):
            objective_val = np.asarray(objective_val, dtype=np.float64).reshape(-1)

        self.objective_fn = objective_fn
        self.update_param_fn = update_param_fn
        self.objective_val = objective_val

    def _f(self, param):
        if self.update_param_fn is not None:
            self.update_param_fn(param)
            return self.objective_fn() - self.objective_val
        return self.objective_fn(param) - self.objective_val

    def solve(self, method: str, *args, **kwargs):
        try:
            if method in cyroot.SCALAR_ROOT_FINDING_METHODS:
                res = cyroot.find_scalar_root(method, self._f, *args, **kwargs)
            elif method in cyroot.VECTOR_ROOT_FINDING_METHODS:
                res = cyroot.find_vector_root(method, self._f, *args, **kwargs)
            else:
                raise ValueError(f'Unknown method {method}.')
            return res.root
        except:
            return None
