import numpy as np

from .consecutive_count import consecutive_bincount
from ..pseudo_multinomial_generator import PseudoMultinomialGenerator

__all__ = ['validate_generator']


def validate_generator(g: PseudoMultinomialGenerator, n_rolls: int = 1000):
    r"""
    Check analytical probabilities and expectations of a pseudo-random multinomial distribution
    generator.

    Args:
        g (PseudoMultinomialGenerator): An instance of :class:`PseudoMultinomialGenerator`.
        n_rolls (int): Number of rolls.
    """
    if not isinstance(g, PseudoMultinomialGenerator):
        raise ValueError('g is not an instance of PseudoMultinomialGenerator.')
    if n_rolls < 1:
        raise ValueError(f'n_rolls must be positive. Got {n_rolls}.')

    print(f'[Checking Generator] n_rolls={n_rolls}')
    print(g)
    print('-' * 50)
    print('Chain transition matrix:')
    print(g.S)
    print('-' * 50)

    state, chain_state = g.state()
    rolls = g.next_states(n_rolls)
    g.set_state(state, chain_state)

    print('analytical probs:', g.probs())
    print('simulated probs :', np.bincount(rolls, minlength=len(g)) / n_rolls)

    print('\nanalytical expectations:', g.expectations())
    print('simulated expectations :', consecutive_bincount(rolls))
    print()
