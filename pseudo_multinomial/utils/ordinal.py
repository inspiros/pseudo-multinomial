from transfinite import Ordinal as Ordinal
from transfinite.ordinal import is_ordinal
from transfinite.util import is_finite_ordinal

__all__ = [
    'Ordinal',
    'is_ordinal',
    'is_finite_ordinal',
    'is_transfinite_ordinal',
    'omega',
]


def is_transfinite_ordinal(n):
    """Check if given value is a transfinite ordinal."""
    return isinstance(n, Ordinal) and n.coefficient > 0


def omega():
    """Return the first transfinite ordinal w."""
    return Ordinal()
