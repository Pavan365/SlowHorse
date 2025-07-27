"""
Functions for setting up a harmonic oscillator system.
"""

# Add source directory to path.
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
)

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

    Paramaeters
    -----------
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

    # Enfore periodic boundaries.
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


def hamiltonian_driven(domain: sim.HilbertSpace1D, time: float) -> sim.RMatrix:
    """
    Generates the time-dependent Hamiltonian term of a driven harmonic
    oscillator in natural units.

    Parameters
    ----------
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    time: float
        The time at which to generate the time-dependent Hamiltonian term.

    Returns
    -------
    sim.RMatrix
        The time-dependent Hamiltonian term (matrix).
    """

    return np.diag(domain.x_axis, k=0) * np.cos(time)
