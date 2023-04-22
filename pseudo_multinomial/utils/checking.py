import numpy as np

from .consecutive_count import consecutive_bincount
from ..master_chain import MasterChain

__all__ = ['check_chain']


def check_chain(chain: MasterChain, n_rolls: int = 1000):
    """Check analytical probabilities and expectations of chain.

    Args:
        chain (MasterChain): Chain.
        n_rolls (int): Number of rolls.
    """
    if n_rolls < 1:
        raise ValueError(f'n_rolls must be positive. Got {n_rolls}.')
    print(f'[Testing] n_rolls={n_rolls}')
    print(chain)
    print('-' * 50)
    print('Chain transition matrix:')
    print(chain.S)
    print('-' * 50)

    state, chain_state = chain.state()
    rolls = chain.next_states(n_rolls)
    chain.set_state(state, chain_state)

    print('\nanalytical probs:', chain.probs())
    print('simulated probs :', np.bincount(rolls, minlength=len(chain)) / n_rolls)

    print('\nanalytical expectations:', chain.expectations())
    print('simulated expectations :', consecutive_bincount(rolls))
    print()
