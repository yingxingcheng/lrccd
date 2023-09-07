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
    # return np.hstack(([0], x[::-1])), np.hstack(([0], w[::-1]))
    return x, w


def get_contract_mat(large_mat, natom):
    res = np.zeros((natom, natom))
    length = int(large_mat.shape[0] / natom)
    for i in range(length):
        for j in range(length):
            res += large_mat[i * natom: (i + 1) * natom, j * natom: (j + 1) * natom]
    return res


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


def compute_response(hardness, response0, natom):
    assert hardness.shape == response0.shape
    N = hardness.shape[0]

    matrix = np.block([
        [hardness, np.zeros((N, 1)), np.identity(N), np.zeros((N, 1))],
        [np.zeros((1, N)), np.zeros((1, 1)), np.zeros((1, N)), np.zeros((1, 1))],
        [np.identity(N), np.zeros((N, 1)), response0, np.zeros((N, 1))],
        [np.zeros((1, N)), np.zeros((1, 1)), np.zeros((1, N)), np.zeros((1, 1))]
    ])
    matrix[:natom, N] = 1
    matrix[N, :natom] = 1
    matrix[N + 1: N + 1 + natom:, -1] = 1
    matrix[-1, N + 1: N + 1 + natom:] = 1
    print(matrix)
    matrix_inv = np.linalg.inv(matrix)
    response = - matrix_inv[:N, :N]
    return response


def compute_response_svd(hardness, chi0, natom):
    ndim = hardness.shape[0]
    O = np.identity(ndim)
    D = np.zeros((ndim,))
    D[:natom] = 1.0

    P = np.zeros((2 * ndim + 1, 2 * ndim + 1))
    P[:ndim, :ndim] = -hardness
    P[ndim:2 * ndim, :ndim] = P[:ndim, ndim:2 * ndim] = O
    P[:ndim, -1] = P[-1, :ndim] = D
    P[ndim:2 * ndim, ndim:2 * ndim] = -chi0

    P_inv = np.linalg.pinv(P)
    chi = P_inv[:ndim, :ndim]
    return chi
