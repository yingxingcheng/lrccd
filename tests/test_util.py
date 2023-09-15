from __future__ import print_function

import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import json

from lrccd import ACKS2w
from lrccd.util import *

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


@pytest.mark.parametrize('nfreq', [10, 12, 13, 20])
def test_gauss_legendre_points_lambda(nfreq):
    omegas, weights = get_gauss_legendre_points_lambda(nfreq)
    # \int_0^1 x dx == 0.5
    assert np.dot(omegas, weights) == pytest.approx(0.5, 1e-8)
    # \int_0^1 e^x dx == e - 1
    assert np.dot(np.exp(omegas), weights) == pytest.approx(np.e - 1, 1e-8)


def test_get_label_from_z():
    z_list = [1, 2, 3, 4]
    symbol_list = get_label_from_z(z_list)
    assert symbol_list == ['H1', 'He1', 'Li1', 'Be1']

    z_list = [1, 1, 3, 4]
    symbol_list = get_label_from_z(z_list)
    assert symbol_list == ['H1', 'H2', 'Li1', 'Be1']

    z_list = [1, 1, 1, 4]
    symbol_list = get_label_from_z(z_list)
    assert symbol_list == ['H1', 'H2', 'H3', 'Be1']

    z_list = [1, 1, 1, 4, 6, 8]
    symbol_list = get_label_from_z(z_list)
    assert symbol_list == ['H1', 'H2', 'H3', 'Be1', 'C1', 'O1']
