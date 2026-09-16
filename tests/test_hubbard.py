"""Tests for the exact-Hubbard feature.

The solver itself is a port that carries its own validation suite (Mathieu-function
band edges, brute-force special-function sums, real-space Wannier overlaps, and a
pixel comparison against the figures published with the original Mathematica
package).  What is new *here* is the unit layer, and that is what these tests are
aimed at: a wrong conversion factor produces perfectly plausible numbers, so it has
to be pinned by an invariance rather than by a golden value.

The load-bearing test is `test_u_over_recoil_is_unit_system_invariant`: the same
physical problem, stated in four different (lattice constant, alpha) conventions,
must give the same dimensionless U/E_rec.
"""

from __future__ import annotations

import numpy as np
import pytest

from bloch_schrodinger.hubbard import (V_REC, UnitConverter, a1d_inv_from_a3d,
                                       hopping, hubbard_u_quasi1d,
                                       hubbard_u_quasi2d,
                                       hubbard_u_wannier_1d,
                                       log_a2d_inv_from_a3d,
                                       matched_square_depth,
                                       site_trap_frequency)
from bloch_schrodinger.hubbard.bands import e_k0_sigma, eig_h0_sigma

# (lattice constant, alpha) conventions that all describe the same physics
CONVENTIONS = [(1.0, 0.5), (np.pi, 1.0), (4 * np.pi / 3, 1.0), (2.5, 3.0)]


# --------------------------------------------------------------- unit layer
@pytest.mark.parametrize("d_c,alpha", CONVENTIONS)
def test_recoil_maps_to_V_REC(d_c, alpha):
    """The defining identity: a lattice's own recoil is pi^2/2 in solver units."""
    c = UnitConverter(d_c, alpha)
    assert float(c.energy_to_adlong(c.recoil_energy)) == pytest.approx(V_REC, rel=1e-13)


@pytest.mark.parametrize("d_c,alpha", CONVENTIONS)
def test_converter_round_trips(d_c, alpha):
    c = UnitConverter(d_c, alpha)
    for v in (3.7, -0.25, 1e3):
        assert float(c.energy_from_adlong(c.energy_to_adlong(v))) == pytest.approx(v)
        assert float(c.length_from_adlong(c.length_to_adlong(v))) == pytest.approx(v)


def test_converter_rejects_bad_input():
    for bad in [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0), (np.nan, 1.0)]:
        with pytest.raises(ValueError):
            UnitConverter(*bad)


def test_u_over_recoil_is_unit_system_invariant():
    """Same physics in four unit systems -> identical U/E_rec.

    This is the test that would catch a wrong conversion factor; a golden value
    would not, because a consistent factor error rescales every number together.
    """
    depth = 12.0
    w_over_erec = 40.0 / V_REC        # hbar*omega_perp, in recoils
    a1d_over_d = -0.1

    ratios, t_ratios = [], []
    for d_c, alpha in CONVENTIONS:
        c = UnitConverter(d_c, alpha)
        res = hubbard_u_quasi1d(
            depth=depth, omega_perp=w_over_erec * c.recoil_energy,
            lattice_constant=d_c, alpha=alpha, conv_param=2,
        )
        u = float(res(1.0 / (a1d_over_d * d_c)))
        ratios.append(u / c.recoil_energy)
        t_ratios.append(res.t / c.recoil_energy)

    assert np.ptp(ratios) / abs(np.mean(ratios)) < 1e-10
    assert np.ptp(t_ratios) / abs(np.mean(t_ratios)) < 1e-10


def test_quasi2d_u_over_recoil_is_unit_system_invariant():
    depth, w_over_erec = 12.0, 40.0 / V_REC
    ratios = []
    for d_c, alpha in CONVENTIONS[:3]:
        c = UnitConverter(d_c, alpha)
        res = hubbard_u_quasi2d(
            depth=depth, omega_z=w_over_erec * c.recoil_energy,
            lattice_constant=d_c, alpha=alpha, conv_param=1,
        )
        # log(1/a_2D) with a_2D in caller units; fix a_2D/d and convert
        log_a2d_inv = -np.log(0.05 * d_c)
        ratios.append(float(res(log_a2d_inv)) / c.recoil_energy)
    # Looser than the 1D case (1e-10) on purpose. 1/U in 2D is a difference of two
    # nearly equal terms (effMass*Re[1/T] and Re[Pi_Hubbard], ~5.8 each for a
    # result of ~0.08), so it amplifies inputs by ~70x, and the renormalisation
    # counter term amplifies its own argument by several more orders. The ~1e-16
    # difference in omega_z between two unit systems therefore surfaces at ~1e-9;
    # that is the algorithm's conditioning, not a conversion error.
    assert np.ptp(ratios) / abs(np.mean(ratios)) < 1e-7


@pytest.mark.parametrize("d_c,alpha", CONVENTIONS)
def test_scattering_length_relations_are_invariant(d_c, alpha):
    """a_1D/d and a_2D/d depend only on a_3D/d and omega/E_rec."""
    c = UnitConverter(d_c, alpha)
    w = (40.0 / V_REC) * c.recoil_energy
    a3d_over_d = 0.3
    # 1/a_1D in units of 1/d
    got = float(a1d_inv_from_a3d(a3d_over_d * d_c, w, d_c, alpha)) * d_c
    assert got == pytest.approx(12.5094971611, rel=1e-9)
    # log(d/a_2D)
    got2 = float(log_a2d_inv_from_a3d(a3d_over_d * d_c, w, d_c, alpha)) + np.log(d_c)
    assert got2 == pytest.approx(1.8827191661, rel=1e-9)


# ------------------------------------------------------------------ physics
def test_hopping_matches_quarter_bandwidth():
    """4t equals the lowest-band width in a deep lattice, in any unit system."""
    for d_c, alpha in CONVENTIONS[:3]:
        c = UnitConverter(d_c, alpha)
        v = 20.0 * V_REC
        width = e_k0_sigma(v, 40, np.pi) - e_k0_sigma(v, 40, 0.0)
        width = float(c.energy_from_adlong(width))
        t = hopping(20.0, d_c, alpha)
        assert abs(4 * t - width) / width < 1e-3


def test_exact_u_approaches_wannier_u_at_weak_coupling():
    """The Wannier overlap is the leading Born term of the exact U.

    Checked as a limit, not at a point: the gap must shrink as the coupling does.
    """
    d_c, alpha, depth = np.pi, 1.0, 12.0
    c = UnitConverter(d_c, alpha)
    w_perp = (40.0 / V_REC) * c.recoil_energy
    exact = hubbard_u_quasi1d(depth=depth, omega_perp=w_perp,
                              lattice_constant=d_c, alpha=alpha, conv_param=2)
    naive, quartic, norm = hubbard_u_wannier_1d(depth, w_perp, d_c, alpha)
    assert norm == pytest.approx(1.0, abs=1e-6)

    devs = []
    for a1d_inv in (1e-2 / d_c, 1e-3 / d_c):
        u_e, u_n = float(exact(a1d_inv)), float(naive(a1d_inv))
        devs.append(abs(u_e - u_n) / abs(u_n))
    # both small, and shrinking towards zero coupling
    assert devs[0] < 0.05
    assert devs[1] <= devs[0]


def test_deeper_lattice_narrows_the_exact_wannier_gap():
    """A tighter on-site wavefunction makes the single-band estimate better."""
    d_c, alpha = np.pi, 1.0
    c = UnitConverter(d_c, alpha)
    w_perp = (40.0 / V_REC) * c.recoil_energy
    devs = []
    for depth in (8.0, 16.0, 25.0):
        exact = hubbard_u_quasi1d(depth=depth, omega_perp=w_perp,
                                  lattice_constant=d_c, alpha=alpha, conv_param=2)
        naive, _, _ = hubbard_u_wannier_1d(depth, w_perp, d_c, alpha)
        a1d_inv = 1e-2 / d_c
        devs.append(abs(float(exact(a1d_inv)) - float(naive(a1d_inv)))
                    / abs(float(naive(a1d_inv))))
    assert devs[0] > devs[1] > devs[2]


def test_u_vanishes_at_zero_coupling():
    res = hubbard_u_quasi1d(depth=12, omega_perp=8.0, lattice_constant=np.pi,
                            conv_param=1)
    assert abs(float(res(0.0))) < 1e-12


def test_site_trap_frequency_matches_band_gap_in_deep_lattice():
    """hbar*omega = 2 sqrt(v E_rec) is the harmonic 0->1 gap of a sinusoidal well.

    The true gap sits one recoil below it -- the standard leading anharmonic
    shift, ``E_1 - E_0 ~ 2 sqrt(s) E_rec - E_rec`` -- so the harmonic value is an
    overestimate that converges from above.  Both facts are asserted, since the
    function is used to match a separable reference lattice to a real one and a
    sign error in the correction would bias every matched depth.
    """
    d_c, alpha = np.pi, 1.0
    c = UnitConverter(d_c, alpha)
    ratios = []
    for depth in (10.0, 30.0, 100.0):
        w, _ = eig_h0_sigma(depth * V_REC, 50, 0.0)
        gap = float(c.energy_from_adlong(w[1] - w[0]))
        harmonic = site_trap_frequency(depth, d_c, alpha)
        assert harmonic > gap                                   # overestimate
        assert harmonic - c.recoil_energy == pytest.approx(gap, rel=0.06)
        ratios.append(gap / harmonic)
    assert ratios[0] < ratios[-1]                               # converges to 1
    assert ratios[-1] > 0.94


@pytest.mark.parametrize("d_c,alpha", CONVENTIONS)
def test_matched_square_depth_inverts_site_trap_frequency(d_c, alpha):
    for depth in (5.0, 12.0, 30.0):
        w = site_trap_frequency(depth, d_c, alpha)
        assert float(matched_square_depth(w, d_c, alpha)) == pytest.approx(depth)


# ------------------------------------------------- agreement with the source port
def test_matches_standalone_reference_implementation():
    """alpha = 1/2, d = 1 IS the solver's own unit system, so nothing should move."""
    ref = pytest.importorskip("hubbard_lattice")
    r = ref.setup_hubbard_u_quasi1d(12 * V_REC, 12 * V_REC, 0.05, 40.0, 2)
    n = hubbard_u_quasi1d(depth=12, omega_perp=40.0, lattice_constant=1.0,
                          alpha=0.5, conv_param=2)
    probe = np.array([-10.0, -3.0, 1.0, 7.0])
    assert np.allclose(np.asarray(r(probe), float), np.asarray(n(probe), float),
                       rtol=1e-12)
