import numpy as np

from ._types import MatrixLike

__all__ = [
    'eig',
    'brute',
    'lstsq',
    'qr_solve',
    'compute_stationary',
]


# noinspection DuplicatedCode
def eig(P):
    v = np.linalg.eig(P.T)[1].real
    return v[:, 0] / v[:, 0].sum()


# noinspection DuplicatedCode
def lstsq(P):
    A = np.eye(P.shape[0], dtype=P.dtype) - P.T
    A = np.concatenate((A, np.ones((1, P.shape[1]), dtype=P.dtype)))
    b = np.zeros((P.shape[0] + 1, 1))
    b[-1] = 1
    return np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)


# noinspection DuplicatedCode
def qr_solve(P):
    A = np.eye(P.shape[0], dtype=P.dtype) - P.T
    A = np.concatenate((A, np.ones((1, P.shape[1]), dtype=P.dtype)))
    b = np.zeros((P.shape[0] + 1, 1))
    b[-1] = 1
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b).reshape(-1)


# noinspection DuplicatedCode
def brute(P, check_freq=5, max_iter=100, rtol=1e-5, atol=1e-8):
    S = np.eye(P.shape[0], dtype=P.dtype)
    prev = S[0]
    for i in range(max_iter):
        S = S @ P
        if i % check_freq == 0:
            if np.allclose(S[0], prev, rtol=rtol, atol=atol):
                break
            prev = S[0]
    return S[0]


ALGOS = {
    'eig': eig,
    'lstsq': lstsq,
    'qr_solve': qr_solve,
    'brute': brute,
}


def compute_stationary(P: MatrixLike,
                       method: str = 'auto',
                       **kwargs) -> np.ndarray:
    r"""
    Compute stationary distribution given the transition matrix.

    Args:
        P: Transition matrix of shape (N, N) of a Markov Chain with N states.
        method: Algorithm to use, available algorithms are 'eig', 'lstsq', 'qr_solve', and 'brute'.
            Defaults to 'auto', which choose automatically.
        **kwargs: Extra keyword arguments for the solver.

    Returns:
        s: Stationary distribution of shape (N,).
    """
    if method is None:
        method = 'auto'
    assert method in list(ALGOS.keys()) + ['auto']
    if method == 'auto':
        # automatically chose best algo based on shape of transition matrix
        n = P.shape[0]
        if n <= 128:
            # for small matrices, eigen decomposition is fast enough
            # probably the fastest
            method = 'eig'
        else:
            # for larger matrices, the two other algorithms are much better
            # while lstsq is more stable, qr_solver is generally faster
            method = 'qr_solve'
    return ALGOS[method](P, **kwargs)
