import math

from pseudo_multinomial import *
from pseudo_multinomial.chains import InfiniteChain
from pseudo_multinomial.utils import validate_generator


class QuadraticChain(FiniteChain):
    def __init__(self, c=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert c >= 0
        self.c = c

    def exit_probability(self, k: int):
        return self.c * k ** 2

    def n_states(self):
        return math.floor(math.sqrt(1 / self.c))

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'


class AlgebraicChain(InfiniteChain):
    def __init__(self, c=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert c >= 0
        self.c = c
        self._c_sqrt = math.sqrt(self.c)

    def exit_probability(self, k: int):
        return self._c_sqrt * k / math.sqrt(1 + self.c * k ** 2)

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'


def main():
    g = PseudoMultinomialGenerator.from_pvals([
        LinearChain(c=0.4),
        GeometricChain(a=1, r=.2),
        LambdaChain(lambda k: k / math.sqrt(1 + k ** 2)),
        QuadraticChain(c=.15),
        AlgebraicChain(c=7),
    ], repeat=False)

    validate_generator(g, n_rolls=100000)


if __name__ == '__main__':
    main()
