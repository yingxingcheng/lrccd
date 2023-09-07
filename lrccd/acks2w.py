#!/usr/bin/env python

from __future__ import print_function
import os

import numpy as np
import scipy.linalg
from progress.bar import Bar
from horton import IOData, RLibXCWrapper, BeckeMolGrid, get_npure_cumul, \
    fill_pure_polynomials, ProAtomDB, HirshfeldIWPart, solve_poisson_becke, \
    electronvolt, load_h5, periodic, HirshfeldWPart, JustOnceClass, Cache, \
    just_once, dump_h5, MBISWPart, log

from lrccd.util import *

np.set_printoptions(precision=5, suppress=True)
log.set_level(1)


class ACKS2w(JustOnceClass):
    CACHE_FNAME = 'cache.h5'
    ATOM_FNAME = 'atoms.h5'
    FCHK_FNAME = 'gaussian.fchk'

    def __init__(self,
                 dirname,
                 lmax=1,
                 output_dir=None,
                 freqs=None,
                 use_imag_freq=False,
                 from_cache=True,
                 refresh_cache=False,
                 cache_fname=None,
                 atoms_db_fname=None,
                 use_gga=False,
                 fchk_fname=None,
                 part_method='mbis',
                 eta=0.001,
                 verbose=False,
                 use_analytic_lda=False,
                 xc='lda',
                 agspec='medium',
                 ):
        r"""ACKS2w model

        Parameters
        ----------
        dirname : string
            Path of ab-initio calculations.
        lmax : int, default=1
            The max of angular momentum index.
        output_dir : string, default=None
            The directory name for output.
        freqs : array_like, default=None
            A list of frequencies.
        use_imag_freq : bool, default=False
            Whether use pure imaginary frequencies.
        from_cache : bool, default=True
            Whether load cache from file.
        refresh_cache : bool, default=False
            Whether update cache.
        cache_fname : string, default=None
            The filename of cache file.
        atoms_db_fname : string, default=None
            The filename of atomic database.
        use_gga : bool, default=False
            Whether use GGA.
        fchk_fname : string, default=None
            The filename of Gaussian checkpoint file with a suffix ``fchk``.
        part_method : {'hirshfeldi', 'mbis'}, default='mbis'
            The partitioning method.
        eta: float, default=1e-3
            Small value :math:`\eta` for removing divergence when real frequencies used.
        verbose: bool, default=False
            Whether output more details.
        use_analytic_lda : bool, default=False
            Whether use an analytic LDA expression to calculate dnesity gradient.

        """
        # Init base class
        JustOnceClass.__init__(self)

        # Input files and cache files
        self.dirname = dirname
        self.lmax = lmax
        self.from_cache = from_cache
        self.cache_fname = cache_fname or ACKS2w.CACHE_FNAME
        self.atoms_db_fname = atoms_db_fname or os.path.join(self.dirname, ACKS2w.ATOM_FNAME)
        self.use_analytic_lda = use_analytic_lda

        # Parameter for calculations
        self.freqs = [0.0] if freqs is None else list(freqs)
        # static case is required for other frequencies
        if not np.isclose(min(self.freqs), 0.0):
            self.freqs.append(0.0)
        self.freqs = sorted(self.freqs)
        self.nfreqs = len(self.freqs)

        self.use_imag_freq = use_imag_freq
        self.eta = eta

        self.fchk_fname = fchk_fname or ACKS2w.FCHK_FNAME
        self.output_dir = output_dir or self.dirname
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.verbose = verbose
        self.gga = use_gga
        assert xc in ['lda', 'ldax', 'svwn', 'svwn5', 'svwn1', 'svwn2', 'svwn3', 'svwn4',
                           'svwn5_rpa', 'pbe']
        self.xc = xc.lower()
        if use_gga: assert self.xc == 'pbe'
        self.agspec = agspec
        self._setup_grid()
        self._setup_xc()
        self.read_cache(from_cache, refresh_cache)
        self.part_method = str(part_method).lower()


    def read_cache(self, from_cache, refresh_cache):
        # restore cache array
        if from_cache:
            self.cache = self._restore_from_output(self.cache_fname)
        else:
            self.cache = Cache()

        if 'is_ok' not in self.cache:
            # this must be a new cache, thus we need to restore cache
            self.cache.dump('is_ok', 0)
            self.refresh_cache = True
        else:
            # first check if all freqs are finished.
            for omega in self.freqs:
                job_key = 'job_' + self.get_key(omega)
                if self.cache.load(job_key, default=0) != 1:
                    if self.verbose:
                        print('find new {}!'.format(job_key))
                    self.cache.dump('is_ok', 0)
                    break
                else:
                    if self.verbose:
                        print('{} done!'.format(job_key))

            if refresh_cache or self.cache.load('is_ok') == 0:
                self.refresh_cache = True
            else:
                self.refresh_cache = False

    def _restore_from_output(self, fname):
        """Load cache from output file named `fname`."""
        full_filename = os.path.join(self.output_dir, fname)
        if os.path.exists(full_filename):
            obj = load_h5(full_filename)
        else:
            return Cache()
        cache = Cache()
        for k, v in obj.iteritems():
            cache.dump(k, v)
        return cache

    def _setup_grid(self):
        """Molecular grid setup."""
        fchk_fname = os.path.join(self.dirname, self.fchk_fname)
        self.mol = IOData.from_file(fchk_fname)
        self.exp = self.mol.exp_alpha

        # define the integration grid
        self.grid = BeckeMolGrid(self.mol.coordinates, self.mol.numbers,
                                 self.mol.pseudo_numbers, mode='keep', agspec=self.agspec)

    def _setup_xc(self):
        """Exchange-correlation functionals setup."""
        if self.gga or self.xc == 'pbe':
            print('PBE fxc loaded!')
            self.xc_wrappers = [RLibXCWrapper('gga_x_pbe'), RLibXCWrapper('gga_c_pbe')]
        else:
            # Use LDA
            print('{} fxc loaded!'.format(self.xc.upper()))
            if self.xc in ['lda', 'svwn', 'svwn5']:
                self.xc_wrappers = [RLibXCWrapper('lda_x'), RLibXCWrapper('lda_c_vwn')]
            elif self.xc in ['svwn1', 'svwn2', 'svwn3', 'svwn4']:
                self.xc_wrappers = [
                    RLibXCWrapper('lda_x'),
                    RLibXCWrapper('lda_c_vwn_{}'.format(self.xc[-1]))
                ]
            elif self.xc in ['svwn5_rpa']:
                self.xc_wrappers = [RLibXCWrapper('lda_x'), RLibXCWrapper('lda_c_vwn_rpa')]
            elif self.xc in ['ldax']:
                self.xc_wrappers = [RLibXCWrapper('lda_x')]
            else:
                raise RuntimeError('Unknown lda functional: {}'.format(self.xc))

    def load_cache(self, key, omega):
        """Load object from cache.

        Parameters
        ----------
        key : string
            The keyword of object loaded.
        omega :float
            The corresponding frequency.

        Returns
        -------
        np.array
            Object stored in cache, or ``None`` on empty.

        """
        str_omega = self.get_key(omega)
        if not self.use_complex_func(omega):
            k = key + '_' + str_omega
            if k in self.cache:
                obj = self.cache.load(k)
                return obj
            return None
        else:
            k_re = key + '_Re_' + str_omega
            k_im = key + '_Im_' + str_omega

            if k_re in self.cache and k_im in self.cache:
                obj_re = self.cache.load(k_re)
                obj_im = self.cache.load(k_im)
                obj = np.zeros(obj_re.shape, dtype=np.complex)
                obj.real = obj_re[:]
                obj.imag = obj_im[:]
                return obj
            return None

    def dump_cache(self, key, omega, obj):
        """Dump cache.

        Parameters
        ----------
        key : string
            The key used for indexing.
        omega : float
            Frequency.
        obj
            The object needed to be dumped.

        """
        tag = self.get_key(omega)
        if not self.use_complex_func(omega):
            k = key + '_' + tag
            self.cache.dump(k, obj)
        else:
            k_re = key + '_Re_' + tag
            k_im = key + '_Im_' + tag
            self.cache.dump(k_re, obj.real)
            self.cache.dump(k_im, obj.imag)

    def run(self):
        """Run ACKS2w model."""
        if not self.refresh_cache and self.cache.load('is_ok') == 1:
            return

        bar = Bar('Calculating', max=len(self.freqs))
        for idx, omega in enumerate(self.freqs):
            # record the state of each job.
            job_key = 'job_' + self.get_key(omega)
            if not self.refresh_cache and self.cache.load(job_key, default=0) == 1:
                continue

            # construct potential functions g(r) and compute non-interacting response function
            k1 = 'response_ks'
            response_ks = self.load_cache(k1, omega)
            k2 = 'hardness_hartree'
            hardness_hartree = self.load_cache(k2, 0.0)
            k3 = 'hardness_xc'
            hardness_xc = self.load_cache(k3, 0.0)

            if response_ks is None or hardness_hartree is None or hardness_xc is None:
                if self.is_static(omega):
                    response_ks, hardness_hartree, hardness_xc = self.calc_parameters(omega)
                    for k, o in zip([k1, k2, k3], [response_ks, hardness_hartree, hardness_xc]):
                        self.dump_cache(k, omega, o)
                else:
                    response_ks, _, _ = self.calc_parameters(omega)
                    self.dump_cache(k1, omega, response_ks)

            k4 = 'response'
            response = self.load_cache(k4, omega)
            if response is None:
                response = self.calc_response(response_ks=response_ks,
                                              hardness_hartree=hardness_hartree,
                                              hardness_xc=hardness_xc, use_rpa=False)
                self.dump_cache(k4, omega, response)

            k5 = 'response_rpa'
            response_rpa = self.load_cache(k5, omega)
            if response_rpa is None:
                response_rpa = self.calc_response(response_ks=response_ks,
                                                  hardness_hartree=hardness_hartree,
                                                  hardness_xc=hardness_xc, use_rpa=True)
                self.dump_cache(k5, omega, response_rpa)

            self.calc_polarizability(omega, response_ks, key='polar_ks')
            self.calc_polarizability(omega, response, key='polar')
            self.calc_polarizability(omega, response_rpa, key='polar_rpa')

            # record the state of each job.
            self.cache.dump(job_key, 1)
            bar.next()
            print()
        bar.finish()

        # dump all cache
        if self.refresh_cache:
            self.cache.dump('is_ok', 1)
            new_dict = {k: v for k, v in self.cache.iteritems()}
            dump_h5(os.path.join(self.output_dir, self.cache_fname), new_dict)

    @just_once
    def calc_potential_function(self):
        """Compute potential functions"""
        # compute the total overlap operator
        if self.verbose:
            print('overlap')

        self.do_partition()
        # compute overlap operators
        self.olp = self.mol.obasis.compute_overlap(self.mol.lf)
        self.olp_operators = self.compute_overlap_operators()

    def calc_parameters(self, omega):
        """Calculate all force-field parameters of ACKS2w model, including KS response matrix,
        Hartree contirbution to hardness and xc contribution to hardness.

        Parameters
        ----------
        omega : float
            The frequency.

        Returns
        -------
        tuple
            A tuple of KS response matrix, Hartree hardness and xc hardness.

        """
        self.calc_potential_function()
        operators = self.olp_operators

        # Compute the density response functions
        if self.verbose:
            print('Computing non-interacting response functions')

        self.compute_fukui_av()
        dmds = self.compute_dmds(omega) + [self.dmd_fukui]
        if not self.use_complex_func(omega):
            dmds = self.compute_dmds(omega) + [self.dmd_fukui]
            # Construct the cross-grammian of the operators and the dmds, Eq. (25)
            cg = np.zeros((self.nop, len(dmds)))
            for i0 in range(self.nop):
                for i1 in range(len(dmds)):
                    cg[i0, i1] = operators[i0].contract_two('ab,ab', dmds[i1])
        else:
            real_fukui_av = self.dmd_fukui
            complx_fukui_av = np.zeros(self.dm_full.shape, dtype=np.complex)
            complx_fukui_av.real[:] = real_fukui_av._array[:]
            dmds = self.compute_dmds(omega) + [complx_fukui_av]

            # Construct the cross-grammian of the operators and the dmds, Eq. (25)
            cg = np.zeros((self.nop, len(dmds)), dtype=np.complex)
            for i0 in range(self.nop):
                for i1 in range(len(dmds)):
                    tmp_op = operators[i0]._array
                    cg[i0, i1] = np.einsum('ab,ab', tmp_op, dmds[i1])

        # We get the ks-response for free, Eq. (25)
        response_ks = cg[:, :self.nop]
        response_ks = 0.5 * (response_ks + response_ks.T)
        if self.verbose:
            print('response_ks evals')
            print(np.linalg.eigvalsh(response_ks))

        if self.is_static(omega):
            # construct a bi-orthogonal density basis, Eq. (16) and Eq. (27)
            pinv = np.linalg.pinv(cg, rcond=1e-8)
            dmds_orth = []
            for i0 in range(self.nop):
                dmd = self.mol.lf.create_two_index()
                for i1 in range(self.nop + 1):
                    dmd._array[:] += pinv[i1, i0] * dmds[i1]._array
                dmds_orth.append(dmd)

            # double check: Eq. (16)
            cg_check = np.zeros((self.nop, self.nop))
            for i0 in range(self.nop):
                for i1 in range(self.nop):
                    cg_check[i0, i1] = operators[i0].contract_two('ab,ab', dmds_orth[i1])

            if self.verbose:
                print('Orthogonality test', abs(cg_check - np.identity(self.nop)).max())
            dmds_orth = dmds_orth

            # compute densities of dmds on grid
            rho_list = []
            for iop in range(self.nop):
                if self.verbose:
                    print('rhos', iop)
                rhos = np.zeros(self.grid.size)
                self.mol.obasis.compute_grid_density_dm(dmds_orth[iop], self.grid.points, rhos)
                rho_list.append(rhos)

            if self.gga:
                # compute density gradients of dmds on grid
                grad_list = []
                for iop in range(self.nop):
                    if self.verbose:
                        print('grads', iop)
                    grads = np.zeros((self.grid.size, 3))
                    self.mol.obasis.compute_grid_gradient_dm(dmds_orth[iop], self.grid.points,
                                                             grads)
                    grad_list.append(grads)
            else:
                grad_list = []

            hardness_hartree = self.calc_hardness_hartree(rho_list)
            hardness_xc = self.calc_hardness_xc(rho_list, grad_list)
        else:
            hardness_hartree = hardness_xc = None
        return response_ks, hardness_hartree, hardness_xc

    def calc_response(self, response_ks, hardness_hartree, hardness_xc, use_rpa=False):
        """General method to calculate response matrix

        Parameters
        ----------
        response_ks : 2D np.array
            KS response matrix.
        hardness_hartree : 2D np.array
            Hartree hardness.
        hardness_xc : 2D np.array)
            Exchange-correlation hardness.
        use_rpa : bool, default=False
            Whether use RPA.

        Returns
        -------
        2D np.array
            Response matrix.

        """
        if use_rpa:
            hardness = hardness_hartree
        else:
            hardness = hardness_hartree + hardness_xc
        response = ACKS2w.compute_response(hardness, response_ks, self.mol.natom)
        return response


    def calc_polarizability(self, omega, response, key):
        """General method to calculate polarizability.

        Parameters
        ----------
        omega : float
            The frequency.
        response : np.array
            Response matrix, 2D array containing data with `float` type.
        key : {'polar', 'polar_rpa', 'polar_ks'}
            The type of response matrix, which could be 'polar', 'polar_rpa' and
            'polar_ks', representing full, RPA, and KS response, respectively.

        Returns
        -------
        np.array:
            Dipole polarizability, 2D array containing data with `float` type.

        """
        assert key in ['polar', 'polar_rpa', 'polar_ks']

        field = self.field
        polar = self.load_cache(key, omega)
        if polar is None:
            polar = -np.dot(field.T.conj(), np.dot(response, field))
            self.dump_cache(key, omega, polar)

        if self.verbose:
            print('The type of response matrix is: {}'.format(key))
            print('The polarizability is: ')
            print(polar)

    @just_once
    def do_partition(self, lmax=3):
        """Do partition up to `lmax`"""
        self.padb = ProAtomDB.from_file(self.atoms_db_fname)
        if self.part_method == 'mbis':
            self.part = MBISWPart(self.mol.coordinates, self.mol.numbers, self.mol.pseudo_numbers,
                                  self.grid, self.rho_gs, lmax=lmax)
        elif self.part_method == 'hirshfeldi':
            self.part = HirshfeldIWPart(self.mol.coordinates, self.mol.numbers,
                                        self.mol.pseudo_numbers, self.grid, self.rho_gs, self.padb,
                                        local=True, lmax=lmax)
        else:
            raise TypeError('Unsupported partitioning scheme {} found'.format(self.part_method))
        self.part.do_partitioning()

    def compute_overlap_operators(self):
        """Compute overlap operators"""
        overlap_operators = {}
        for iatom in range(self.part.natom):
            if self.verbose:
                print('Computing AIM multipole operators on grid for atom', iatom)
            # Prepare solid harmonics on grids.
            grid = self.part.get_grid(iatom)
            if self.lmax > 0:
                work = np.zeros((grid.size, self.npure - 1), float)
                work[:, 0] = grid.points[:, 2] - self.part.coordinates[iatom, 2]
                work[:, 1] = grid.points[:, 0] - self.part.coordinates[iatom, 0]
                work[:, 2] = grid.points[:, 1] - self.part.coordinates[iatom, 1]
                if self.lmax > 1:
                    fill_pure_polynomials(work, self.lmax)
            else:
                work = None

            at_weights = self.part.cache.load('at_weights', iatom)
            # Convert the weight functions to AIM overlap operators.
            for ipure in range(self.npure):
                if self.verbose:
                    print('Computing AIM multipole overlap operators', iatom, ipure)
                op = self.mol.lf.create_two_index()
                if ipure > 0:
                    tmp = at_weights * work[:, ipure - 1]
                else:
                    tmp = at_weights
                # convert weight functions to matrix based on basis sets
                self.mol.obasis.compute_grid_density_fock(grid.points, grid.weights, tmp, op)
                overlap_operators[(iatom, ipure)] = op

        # Correct the s-type overlap operators such that the sum is exactly
        # equal to the total overlap.
        if self.verbose:
            print('Correcting sum of s-type operators')
        error_overlap = self.mol.lf.create_two_index()
        for iatom in range(self.part.natom):
            atom_overlap = overlap_operators[(iatom, 0)]
            error_overlap.iadd(atom_overlap)
        error_overlap.iadd(self.olp, -1)
        error_overlap.iscale(1.0 / self.part.natom)
        for iatom in range(self.part.natom):
            atom_overlap = overlap_operators[(iatom, 0)]
            atom_overlap.iadd(error_overlap, -1)

        # sort the operators
        result = []
        # sort the response function basis
        for ipure in range(self.npure):
            for iatom in range(self.part.natom):
                result.append(overlap_operators[(iatom, ipure)])
        return result

    @just_once
    def compute_fukui_av(self):
        """Compute fukui density matrix from average of the HOMO and LUMO density."""
        # Add the average fukui dm to the list of dmds
        exp = self.mol.exp_alpha
        nocc = int(exp.occupations.sum())
        self.dmd_fukui = self.mol.lf.create_two_index()
        # fukui dm is approximated as average of the HOMO and LUMO density
        self.dmd_fukui._array[:] = (np.outer(exp.coeffs[:, nocc], exp.coeffs[:, nocc]) +
                                    np.outer(exp.coeffs[:, nocc - 1], exp.coeffs[:, nocc - 1])) / 2

    def compute_dmds(self, omega):
        """General method to compute half KS response matrix.

        Parameters
        ----------
        omega : float
            The frequency used in Gaussian-Legendre quadrature.

        Returns
        -------
        list:
            A list of half KS response matrix, of which element represents a unique integral
            corresponding potential operator.

        """
        self.calc_potential_function()
        operators = self.olp_operators

        exp = self.mol.exp_alpha
        eta = self.eta
        nop = len(operators)
        norb = exp.nfn

        if self.is_static(omega):
            work = []

            # Fill the work array with the operators transformed to the orbital basis.
            for iop in range(nop):
                tmp_obj = np.dot(exp.coeffs.T, np.dot(operators[iop]._array, exp.coeffs))
                work.append(tmp_obj)

            # Compute the ratio of (n1-n2)/(epsilon1-epsilon2) for every pair of orbitals.
            numerator = np.subtract.outer(exp.occupations, exp.occupations)
            denominator = np.subtract.outer(exp.energies, exp.energies)
            with np.errstate(invalid='ignore', divide='ignore'):
                prefacs = 1 / denominator
            prefacs[numerator == 0] = 0.0
            prefacs *= numerator
            assert np.isfinite(prefacs).all()

            # Set the diagonal elements of prefaxs to zero.
            for iorb in range(norb):
                prefacs[iorb, iorb] = 0.0

            # piecewise multiplication of the transformed operators with the prefactors
            for iop in range(nop):
                work[iop] *= prefacs

            # Transform the derivatives of the density matrix back to the GO basis
            # Dot each operator with the orbitals
            for iop in range(nop):
                work[iop] = np.dot(exp.coeffs, np.dot(work[iop], exp.coeffs.T))

            dmds = []
            for iop in range(nop):
                dmd = self.mol.lf.create_two_index()
                # The prefactor two represents cc.
                dmd._array[:] = work[iop] * 2
                dmds.append(dmd)
            return dmds

        if self.use_complex_func(omega):
            # real omega but omega is not 0.0
            work, work2 = [], []

            # Fill the work array with the operators transformed to the orbital basis.
            for iop in range(nop):
                tmp_obj1 = np.dot(exp.coeffs.T.conj(), np.dot(operators[iop]._array, exp.coeffs))
                tmp_obj2 = np.dot(exp.coeffs.T.conj(), np.dot(operators[iop]._array, exp.coeffs))
                work.append(np.asarray(tmp_obj1, dtype=np.complex))
                work2.append(np.asarray(tmp_obj2, dtype=np.complex))

            # Compute the ratio of (n1-n2)/(epsilon1-epsilon2) for every pair of orbitals.
            with np.errstate(invalid='ignore'):
                pos_prefacs = np.subtract.outer(exp.occupations, exp.occupations) / (
                        np.subtract.outer(exp.energies, exp.energies) + omega + eta * 1j)
                neg_prefacs = np.subtract.outer(exp.occupations, exp.occupations) / (
                        -np.subtract.outer(exp.energies, exp.energies) + omega + eta * 1j)

            # piecewise multiplication of the transformed operators with the prefactors
            for iop in range(nop):
                work[iop] *= np.asarray(pos_prefacs, dtype=np.complex)
                work2[iop] *= np.asarray(neg_prefacs, dtype=np.complex)

            # Transform the derivatives of the density matrix back to the GO basis
            # Dot each operator with the orbitals
            for iop in range(nop):
                work[iop] = np.dot(exp.coeffs, np.dot(work[iop], exp.coeffs.T.conj()))
                work2[iop] = np.dot(exp.coeffs, np.dot(work2[iop], exp.coeffs.T.conj()))

            dmds = []
            for iop in range(nop):
                dmd = np.zeros(work[iop].shape, dtype=np.complex)
                dmd[:, :] = work[iop] - work2[iop]
                dmds.append(dmd)
            return dmds
        else:
            # pure imaginary frequency
            work = []

            # Fill the work array with the operators transformed to the orbital basis.
            for iop in range(nop):
                tmp_obj = np.dot(exp.coeffs.T, np.dot(operators[iop]._array, exp.coeffs))
                work.append(np.asarray(tmp_obj))

            # Compute the ratio of (n1-n2)/(epsilon1-epsilon2) for every pair of orbitals.
            with np.errstate(invalid='ignore'):
                diff_occ = np.subtract.outer(exp.occupations, exp.occupations)
                diff_ene = np.subtract.outer(exp.energies, exp.energies)
                prefacs = diff_occ * 2 * diff_ene / (diff_ene ** 2 + omega ** 2)

            # piecewise multiplication of the transformed operators with the prefactors
            for iop in range(nop):
                work[iop] *= np.asarray(prefacs)

            # Transform the derivatives of the density matrix back to the GO basis
            # Dot each operator with the orbitals
            dmds = []
            for iop in range(nop):
                dmd = self.mol.lf.create_two_index()
                # here, prefactor two represents cc.
                work[iop] = np.dot(exp.coeffs, np.dot(work[iop], exp.coeffs.T))
                dmd._array[:] = work[iop]
                dmds.append(dmd)
            return dmds

    def calc_hardness_hartree(self, rho_list):
        """Compute Hartree hardness.

        Parameters
        ----------
        rho_list: list
            A list of density functions.

        Returns
        -------
        np.array
            Hartree hardness, 2D array containing data with `np.complex` type if complex type
            functions are used or `float` for real type functions.

        Raises
        ------
        RuntimeError
            If the static Hartree hardness does not exist.

        """
        # Construct the classical hardness matrix numerically fast, Eq. (24)
        hardness_hartree = self.load_cache('hardness_hartree', 0.0)
        if hardness_hartree is not None:
            return hardness_hartree

        hardness_hartree = np.zeros((self.nop, self.nop))
        pot_list = []
        for iop in range(self.nop):
            if self.verbose:
                print('pots', iop)
            pots = np.zeros(self.grid.size)
            begin = 0
            for i in range(self.mol.natom):
                atgrid = self.grid.subgrids[i]
                end = begin + atgrid.size
                becke_weights = self.grid.becke_weights[begin:end]
                density_decomposition = atgrid.get_spherical_decomposition(
                    rho_list[iop][begin:end], becke_weights, lmax=4)
                hartree_decomposition = solve_poisson_becke(density_decomposition)
                self.grid.eval_decomposition(hartree_decomposition, atgrid.center, pots)
                begin = end
            pot_list.append(pots)

        for iop0 in range(self.nop):
            for iop1 in range(self.nop):
                hardness_hartree[iop0, iop1] = self.grid.integrate(rho_list[iop0],
                                                                   pot_list[iop1])
        hardness_hartree[:] = 0.5 * (hardness_hartree + hardness_hartree.T)
        if self.verbose:
            print('hardness_hartree')
            print(hardness_hartree[:self.mol.natom, :self.mol.natom])
        return hardness_hartree

    def calc_hardness_xc(self, rho_list, grad_list):
        """Calculate exchange-correlation hardness.

        Parameters
        ----------
        rho_list : list
            A list of density basis functions.
        grad_list : list
            A list of gradient of density basis functions.

        Returns
        -------
        np.array
            Exchange-correlation hardness, 2D array containing data with `float` type for real
            functions or `np.complex` type for complex functions.

        """
        # XC contributions to hardness
        hardness_xc = self.load_cache('hardness_xc', 0.0)
        if hardness_xc is not None:
            return hardness_xc

        # for static or pure imaginary frequencies, one can use GGA functional with
        # finite-difference method to calculate xc contribution to hardness.
        eps = 1e-4
        hardness_xc = np.zeros((self.nop, self.nop))

        if self.gga:
            # compute density gradient of gs
            if self.verbose:
                print('grad_gs')
            grad_gs = np.zeros((self.grid.size, 3))
            self.mol.obasis.compute_grid_gradient_dm(self.dm_full, self.grid.points, grad_gs)
        else:
            # LDA case
            grad_gs = None

        for xc_wrapper in self.xc_wrappers:
            # actual computation with finite diffs, Eq. (28)
            for iop0 in range(self.nop):
                if self.verbose:
                    print('xc', xc_wrapper, iop0)

                if self.gga or not self.use_analytic_lda:
                    potp = np.zeros(self.grid.size)
                    potm = np.zeros(self.grid.size)
                    rhop = self.rho_gs + eps * rho_list[iop0]
                    rhom = self.rho_gs - eps * rho_list[iop0]
                else:
                    potp = np.zeros(self.grid.size)

                if self.gga:
                    # for GGA functional, the default method to calculate xc contribution to
                    # hardness using finite-difference method.
                    spotp = np.zeros(self.grid.size)
                    spotm = np.zeros(self.grid.size)
                    gradp = grad_gs + eps * grad_list[iop0]
                    gradm = grad_gs - eps * grad_list[iop0]
                    sigmap = (gradp * gradp).sum(axis=1)
                    sigmam = (gradm * gradm).sum(axis=1)
                    xc_wrapper.compute_gga_vxc(rhop, sigmap, potp, spotp)
                    xc_wrapper.compute_gga_vxc(rhom, sigmam, potm, spotm)
                    gpotp = 2 * (gradp * spotp.reshape(-1, 1))
                    gpotm = 2 * (gradm * spotm.reshape(-1, 1))
                    dgpot = (gpotp - gpotm) / (2 * eps)
                else:
                    # LDA case, one can use finite-difference to calculate fxc while exact
                    # analytic expressions are also available.
                    if self.use_analytic_lda:
                        xc_wrapper.compute_lda_fxc(self.rho_gs, potp)
                    else:
                        xc_wrapper.compute_lda_vxc(rhop, potp)
                        xc_wrapper.compute_lda_vxc(rhom, potm)
                    dgpot = None

                if self.gga or not self.use_analytic_lda:
                    dpot = (potp - potm) / (2 * eps)

                for iop1 in range(self.nop):
                    if self.use_analytic_lda:
                        hardness_xc[iop0, iop1] += np.asarray(
                            potp * rho_list[iop0] * rho_list[iop1].conj()
                        ).dot(self.grid.weights)
                    else:
                        hardness_xc[iop0, iop1] += self.grid.integrate(dpot, rho_list[iop1])

                    if self.gga:
                        hardness_xc[iop0, iop1] += self.grid.integrate(dgpot[:, 0],
                                                                       grad_list[iop1][:, 0])
                        hardness_xc[iop0, iop1] += self.grid.integrate(dgpot[:, 1],
                                                                       grad_list[iop1][:, 1])
                        hardness_xc[iop0, iop1] += self.grid.integrate(dgpot[:, 2],
                                                                       grad_list[iop1][:, 2])
        hardness_xc = 0.5 * (hardness_xc + hardness_xc.T)
        if self.verbose:
            print('hardness_xc')
            print(hardness_xc[:self.mol.natom, :self.mol.natom])
        return hardness_xc

    # Helper function
    def use_complex_func(self, omega):
        """Whether use complex version function.

        Parameters
        ----------
        omega : float
            The frequency.

        Returns
        -------
        bool
            True for real `omega` or False for pure imaginary `omega` and static `omega`.

        """
        if self.is_static(omega):
            return False
        return not self.use_imag_freq

    def is_static(self, omega):
        r"""Whether a static `omega` is used, i.e., :math:`\omega=0.0`."""
        return np.isclose(float(omega), 0.0)

    @property
    def dm_full(self):
        """The ground-state density matrix"""
        return self.mol.get_dm_full()

    @property
    def rho_gs(self):
        """The ground-state density"""
        if self.verbose:
            print('rho_gs')
        _rho_gs = np.zeros((self.grid.size,))
        self.mol.obasis.compute_grid_density_dm(self.dm_full, self.grid.points, _rho_gs)
        return _rho_gs

    @property
    def npure(self):
        """The number of pure functions up to a given angular momentum, `lmax`."""
        return get_npure_cumul(self.lmax)

    @property
    def nop(self):
        """The total number of operator according to the `lmax` and the number of atoms."""
        return self.npure * self.mol.natom

    @property
    def field(self):
        """The field corresponding distributed polarizabilities."""
        _field, new = self.cache.load('field', alloc=(self.nop, 3))
        if new:
            _field[:self.mol.natom] = self.mol.coordinates
            if self.npure >= 4:
                _field[self.mol.natom: self.mol.natom * 2, 2] = 1
                _field[self.mol.natom * 2:self.mol.natom * 3, 0] = 1
                _field[self.mol.natom * 3:self.mol.natom * 4, 1] = 1
        return _field

    def get_key(self, omega):
        """Generate the key for saving cache.

        If `omega` is a real number, e.g., 0.1 the key will be 'Re_0.100_Im_0.000', while the key
        will be 'Re_0.000_Im_0.200' is the `omega` is  a pure imaginary number, e.g., ' 0.2j'.

        Parameters
        ----------
        omega : float
            The frequency

        Returns
        -------
        string
            The frequency-dependent key.

        """
        key_tmp = 'Re_{:.3f}_Im_{:.3f}'
        if self.use_complex_func(omega):
            return key_tmp.format(omega, 0.0)
        else:
            return key_tmp.format(0.0, omega)

    @property
    def atomic_symbol(self):
        """All atomic symbols of the molecule."""
        return [periodic[int(i)].symbol for i in self.mol.pseudo_numbers]

    @property
    def atomic_Z(self):
        """All atomic numbers of the molecule."""
        return [int(i) for i in self.mol.pseudo_numbers]

    @staticmethod
    def compute_response(hardness, chi0, natom):
        dtype = chi0.dtype
        ndim = hardness.shape[0]
        O = np.identity(ndim, dtype=dtype)
        D = np.zeros((ndim,), dtype=dtype)
        D[:natom] = 1.0
        P = np.zeros((2 * ndim + 1, 2 * ndim + 1), dtype=dtype)
        P[:ndim, :ndim] = -hardness
        P[ndim:2 * ndim, :ndim] = P[:ndim, ndim:2 * ndim] = O
        P[:ndim, -1] = P[-1, :ndim] = D
        P[ndim:2 * ndim, ndim:2 * ndim] = -chi0
        P_inv = np.linalg.pinv(P, rcond=1e-8)
        chi = P_inv[:ndim, :ndim]
        return chi


    def get_response_tensor(self, omega, hxc):
        r"""Get response matrix.

        The different response matrix is calculated from corresponding response specified by
        `hxc`, which could be 'ks', 'rpa', 'hxc'.

        Parameters
        ----------
        omega : float
            The frequency :math:`\omega`.
        hxc : {'ks', 'rpa', 'hxc'}
            The type of response that could be 'ks', 'rpa' and 'hxc', representing KS
            (only Coulomb interaction), RPA (Coulomb + exchange interaction) and the full
            exchange-correlation interaction response, respectively.

        Returns
        -------
            Response matrix, a 2D array.

        """
        use_response_ks, use_rpa = get_xc_kernal_type(hxc)

        if use_response_ks:
            response = self.load_cache('response_ks', omega)
        elif use_rpa:
            response = self.load_cache('response_rpa', omega)
        else:
            response = self.load_cache('response', omega)
        if response is None:
            raise RuntimeError('The cache of response is None!')
        return response
