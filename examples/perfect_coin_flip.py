import numpy as np

from pseudo_multinomial import PseudoMultinomialGenerator, ForwardingChain
from pseudo_multinomial.utils import consecutive_bincount


def main():
    g = PseudoMultinomialGenerator.from_pvals(
        chains=[ForwardingChain(),
                ForwardingChain()],
        repeat=True)

    print(g.pseudo_multinomial(100))
    print('probs:', g.probs())
    print('expectations:', g.expectations())
    print('numpy simulated expectations:',
          consecutive_bincount(np.random.randint(0, 2, 100000)))


if __name__ == '__main__':
    main()
