"""
Functions for setting up a harmonic oscillator system.
"""

# Add source directory to path.
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
)

# Import standard modules.
from typing import cast

# Import external modules.
import numpy as np

# Import local modules.
import simulation as sim


def hamiltonian_standard(domain: sim.HilbertSpace1D) -> sim.RMatrix:
    """
    Generates the Hamiltonian of a standard harmonic oscillator in natural
    units. This function uses a fourth-order central difference approximation
    to construct the kinetic energy operator in position space. This function
    also enforces periodic boundaries.

    Parameters
    ----------
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.

    Returns
    -------
    sim.RMatrix
        The Hamiltonian matrix.
    """

    # Construct the kinetic energy operator.
    ke_coeff: float = -1 / (2 * 12 * (domain.x_dx**2))

    ke_coeff_diag_0: float = -30 * ke_coeff
    ke_coeff_diag_1: float = 16 * ke_coeff
    ke_coeff_diag_2: float = -ke_coeff

    ke_diag_0: sim.RVector = np.full(
        domain.num_points, ke_coeff_diag_0, dtype=np.float64
    )
    ke_diag_1: sim.RVector = np.full(
        (domain.num_points - 1), ke_coeff_diag_1, dtype=np.float64
    )
    ke_diag_2: sim.RVector = np.full(
        (domain.num_points - 2), ke_coeff_diag_2, dtype=np.float64
    )

    ke_operator: sim.RMatrix = (
        np.diag(ke_diag_0, k=0)
        + np.diag(ke_diag_1, k=1)
        + np.diag(ke_diag_2, k=2)
        + np.diag(ke_diag_1, k=-1)
        + np.diag(ke_diag_2, k=-2)
    )

    # Enforce periodic boundaries.
    ke_operator[0, -1] = ke_coeff_diag_1
    ke_operator[-1, 0] = ke_coeff_diag_1

    ke_operator[0, -2] = ke_coeff_diag_2
    ke_operator[-2, 0] = ke_coeff_diag_2
    ke_operator[-1, 1] = ke_coeff_diag_2
    ke_operator[1, -1] = ke_coeff_diag_2

    # Construct the potential energy operator.
    pe_diag: sim.RVector = 0.5 * (domain.x_axis**2)
    pe_operator: sim.RMatrix = np.diag(pe_diag, k=0)

    return ke_operator + pe_operator


def hamiltonian_driven(
    ket: sim.GVector, domain: sim.HilbertSpace1D, time: float
) -> sim.CVector:
    """
    Calculates the action of the Hamiltonian of a driven harmonic oscillator on
    a given state, in natural units. This function uses the momentum space
    (Fourier) to calculate the action of the kinetic energy operator.

    Parameters
    ----------
    ket: GVector
        The state (e.g. wavefunction) to act on.
    domain: HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    time: float
        The time at which to evaluate the Hamiltonian.

    Returns
    -------
    CVector
        The result of acting the Hamiltonian on the given state.
    """

    # Construct and apply the kinetic energy operator.
    p_axis: sim.RVector = cast(
        sim.RVector, 2 * np.pi * np.fft.fftfreq(domain.num_points, d=domain.x_dx)
    )
    ke_operator: sim.RVector = 0.5 * (p_axis**2)
    ke_ket: sim.CVector = np.fft.ifft(ke_operator * np.fft.fft(ket))

    # Construct and apply the potential energy operator.
    pe_operator: sim.RVector = 0.5 * (domain.x_axis**2) + (domain.x_axis * np.cos(time))
    pe_ket: sim.GVector = pe_operator * ket

    return -1j * (ke_ket + pe_ket)


def hamiltonian_driven_diff(
    ket: sim.GVector, domain: sim.HilbertSpace1D, time_1: float, time_2: float
) -> sim.CVector:
    """
    Calculates the difference in the action of the Hamiltonian of a driven
    harmonic oscillator on a given state, in natural units, at two different
    times.

    Parameters
    ----------
    ket: GVector
        The state (e.g. wavefunction) to act on.
    domain: HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    time_1: float
        The first time at which to evaluate the Hamiltonian.
    time_2: float
        The second time at which to evaluate the Hamiltonian.

    Returns
    -------
    CVector
        The result of the difference in acting the Hamiltonian on the given
        state, at the given two different times.
    """

    # Construct the potential energy operator.
    pe_operator_diff: sim.RVector = domain.x_axis * (np.cos(time_1) - np.cos(time_2))
    pe_ket_diff: sim.GVector = pe_operator_diff * ket

    return cast(sim.CVector, -1j * pe_ket_diff)
