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


def get_gauss_legendre_points_lambda(nw=16):
    pts, weights = leggauss(nw)
    pts = pts.real
    new_pts = 1 / 2. * pts + 1 / 2.
    return new_pts, weights / 2.


def get_label_from_z(z_list):
    ele_dict = {}
    label_lis = []
    for z in z_list:
        ele = periodic[int(z)].symbol
        if ele in ele_dict.keys():
            ele_dict[ele] += 1
        else:
            ele_dict[ele] = 1
        label_lis.append('{}{}'.format(ele, ele_dict[ele]))
    return label_lis

