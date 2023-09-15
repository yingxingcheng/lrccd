import numpy as np
from functools import reduce

import pytest
from numpy.polynomial.legendre import leggauss
from horton import periodic


def get_xc_kernal_type(hxc):
    hxc = str(hxc).lower()
    if hxc == 'hxc':
        use_ks = False
        use_rpa = False
    elif hxc in ['rpa', 'h']:
        use_ks = False
        use_rpa = True
    elif hxc in ['ks', 'bare']:
        use_ks = True
        use_rpa = False
    else:
        raise TypeError('Unknown type {}! It should be "hxc", "rpa" or "ks"'.format(hxc))
    return use_ks, use_rpa


def freq_grid(n, L=0.3):
    """
    Gaussian-Legendre quadrature points and weights.
    Args:
        n: the number of points generated.
        L: scale, in libmbd, this value is 0.6 default, but here, we take 0.3.
    Returns:
        A tuple that contains points and weights.
    """
    x, w = leggauss(n)
    w = 2 * L / (1 - x) ** 2 * w
    x = L * (1 + x) / (1 - x)
    return x, w

