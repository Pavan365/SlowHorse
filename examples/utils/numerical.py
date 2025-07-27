"""
Numerical functions for setting up simulations and investigating results.
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


def norm_wavefunction(
    wavefunction: sim.GVector, domain: sim.HilbertSpace1D
) -> sim.GVector:
    """
    Normalises a wavefunction over a given domain. This function uses the
    integral definition of the norm.

    Parameters
    ----------
    wavefunction: simulation.GVector
        The wavefunction to normalise.
    domain: simulation.GVector
        The discretised Hilbert space (domain) of the system.

    Returns
    -------
    simulation.GVector
        The normalised wavefunction.
    """

    return wavefunction / np.sqrt(
        np.trapezoid(np.abs(wavefunction) ** 2, dx=domain.x_dx)
    )


def wavefunctions_norms(
    wavefunctions: sim.GVectors, domain: sim.HilbertSpace1D
) -> sim.RVector:
    """
    Calculates the norms of a set of wavefunctions over a given domain. This
    function uses the integral definition of the norm.

    Parameters
    ----------
    wavefunctions: simulation.GVectors
        The wavefunctions to calculate the norms of. These should be passed
        with shape (num_wavefunctions, num_points).
    domain: simulation.GVector
        The discretised Hilbert space (domain) of the system.

    Returns
    -------
    simulation.RVector
        The norms of the wavefunctions.
    """

    return np.sqrt(np.trapezoid((np.abs(wavefunctions) ** 2), dx=domain.x_dx, axis=1))


def wavefunctions_energies(
    wavefunctions: sim.GVectors,
    domain: sim.HilbertSpace1D,
    system: sim.TDSE1D,
    time_domain: sim.TimeGrid,
) -> sim.RVector:
    """
    Calculates the energies of a set of wavefunctions for a given system. The
    energies are calculated using the expectation value of the Hamiltonian
    operator.

    Parameters
    ----------
    wavefunctions: simulation.GVectors
        The wavefunctions to calculate the energies of. These should be passed
        with shape (num_wavefunctions, num_points), where "num_wavefunctions"
        should equal the number of time points.
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    system: simulation.TDSE1D
        The time-dependent Schrödinger equation (TDSE) of the system.
    time_domain: simulation.TimeGrid
        The time domain (grid) over which the wavefunctions are defined.

    Returns
    -------
    energies: simulation.RVector
        The energies of the wavefunctions.
    """

    energies: sim.RVector = np.zeros(wavefunctions.shape[0], dtype=np.float64)

    if system.hamiltonian_ti is None and system.hamiltonian_td is None:
        raise ValueError("The system must have a Hamiltonian term.")

    # Time-independent energies.
    if system.hamiltonian_td is None:
        assert system.hamiltonian_ti is not None
        hamiltonian: sim.GMatrix = system.hamiltonian_ti(domain)

        for i in range(wavefunctions.shape[0]):
            energies[i] = np.vdot(wavefunctions[i], hamiltonian @ wavefunctions[i]).real

    # Time-dependent energies,
    else:
        if wavefunctions.shape[0] != time_domain.num_points:
            raise ValueError("mismatch between number of wavefunctions and time points")

        assert system.hamiltonian_ti is not None
        hamiltonian_ti: sim.GMatrix = system.hamiltonian_ti(domain)

        for i, (wavefunction, time) in enumerate(
            zip(wavefunctions, time_domain.t_axis)
        ):
            hamiltonian: sim.GMatrix = hamiltonian_ti + system.hamiltonian_td(
                domain, time
            )
            energies[i] = np.vdot(wavefunction, hamiltonian @ wavefunction).real

    return energies
