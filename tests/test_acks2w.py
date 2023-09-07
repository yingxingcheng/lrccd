from __future__ import print_function

import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import json
from lrccd import ACKS2w, freq_grid
import shutil

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


@pytest.mark.parametrize('mol', ['h2o', 'c2h2', 'h2'])
@pytest.mark.parametrize('use_imag_freq', [True, False])
def test_acks2w(mol, use_imag_freq):
    mol = mol.lower()
    dirname = os.path.join(DATA_PATH, 'acks2w', 'aTZ')
    output_dir = 'out'
    omega_list = [2.0, 0.0, 1.0]
    cache_fname = mol + '_cache.h5' 

    acks2w = ACKS2w(dirname=dirname, output_dir=output_dir, freqs=omega_list,
                        part_method='mbis', cache_fname=cache_fname, use_imag_freq=use_imag_freq,
                        fchk_fname='{}.fchk'.format(mol), verbose=True)
    acks2w.run()

    if use_imag_freq:
        assert 'polar_Re_0.000_Im_0.000' in acks2w.cache
        assert 'polar_ks_Re_0.000_Im_0.000' in acks2w.cache
        assert 'polar_rpa_Re_0.000_Im_0.000' in acks2w.cache

        assert 'polar_Re_0.000_Im_1.000' in acks2w.cache
        assert 'polar_ks_Re_0.000_Im_1.000' in acks2w.cache
        assert 'polar_rpa_Re_0.000_Im_1.000' in acks2w.cache

        assert 'polar_Re_0.000_Im_2.000' in acks2w.cache
        assert 'polar_ks_Re_0.000_Im_2.000' in acks2w.cache
        assert 'polar_rpa_Re_0.000_Im_2.000' in acks2w.cache

        assert 'hardness_hartree_Re_0.000_Im_0.000' in acks2w.cache
        assert 'hardness_xc_Re_0.000_Im_0.000' in acks2w.cache
        assert 'is_ok' in acks2w.cache and acks2w.cache['is_ok']

    shutil.rmtree(output_dir)

@pytest.mark.parametrize('mol', ['h2o'])
@pytest.mark.parametrize('use_imag_freq', [True])
def test_acks2w_polar(mol, use_imag_freq):
    mol = mol.lower()
    dirname = os.path.join(DATA_PATH, 'acks2w', 'water')
    output_dir = 'out'
    omega_list = [0.0]
    cache_fname = mol + '_cache.h5' 

    acks2w = ACKS2w(dirname=dirname, output_dir=output_dir, freqs=omega_list,
                        part_method='mbis', cache_fname=cache_fname, use_imag_freq=use_imag_freq,
                        fchk_fname='{}.fchk'.format(mol), verbose=True, use_gga=True, xc='pbe')
    acks2w.run()
    assert 'is_ok' in acks2w.cache and acks2w.cache['is_ok']
    polar = acks2w.load_cache('polar', 0.0)
    polar_ref = np.array(
        [[7.23806684E+00, 6.06514838E-11, 1.42180046E-10],
        [6.06514838E-11,  8.04213953E+00, 1.20021770E-10],
        [1.42180046E-10,  1.20021770E-10, 6.89222663E+00]]
    )
    assert polar == pytest.approx(polar_ref, abs=0.3)
    shutil.rmtree(output_dir)

