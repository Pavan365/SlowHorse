"""
Example use case of the Semi-Global propagation scheme using a driven harmonic
oscillator system.
"""

# Add source directory to path.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import standard modules.
import time
from typing import cast

# Import external modules.
import numpy as np
from numpy.typing import NDArray

# Import local modules.
import utils.harmonic_oscillator as ho
import utils.numerical as num
import utils.visualisation as vis

# Import local modules (src).
import simulation as sim
import propagator as prop


def standard_solution(
    domain: sim.HilbertSpace1D,
    system: sim.TDSE1D,
    wavefunction: sim.GVector,
    time_domain: sim.TimeGrid,
) -> sim.CVectors:
    """
    Calculates the exact wavefunctions for a pure eigenstate of the standard
    harmonic oscillator (no driving) over a given time domain. This function
    is used to ensure that the Semi-Global propagator works in the case that
    the Hamiltonian is time-independent.

    Parameters
    ----------
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    system: simulation.TDSE1D
        The time-dependent Schrödinger equation (TDSE) of the system.
    wavefunction: simulation.GVector
        The initial wavefunction of the system.
    time_domain: simulation.TimeGrid
        The time domain (grid) over which to calculate the exact wavefunctions.

    Returns
    -------
    wavefunctions: simulation.CVectors
        The exact wavefunctions.
    """

    assert system.hamiltonian_ti is not None

    # Calculate the energy of the eigenstate.
    energy: float = cast(
        float,
        np.vdot(wavefunction, system.hamiltonian_ti(domain) @ wavefunction).real
        * domain.x_dx,
    )

    # Calculate the exact solutions.
    wavefunctions: sim.CVectors = np.zeros(
        (time_domain.num_points, domain.num_points), dtype=np.complex128
    )
    wavefunctions[0] = wavefunction.copy()

    for i in range(1, time_domain.num_points):
        wavefunctions[i] = wavefunction * np.exp(-1j * energy * time_domain.t_axis[i])

    return wavefunctions


def main():
    # Set up the spatial domain.
    x_lim: float = 10.0
    x_num_points: int = 512

    domain: sim.HilbertSpace1D = sim.HilbertSpace1D(-x_lim, x_lim, x_num_points)

    # Set up the system.
    system: sim.TDSE1D = sim.TDSE1D(
        ho.hamiltonian_standard, ho.hamiltonian_driven, None
    )

    # Set up the wavefunction.
    assert system.hamiltonian_ti is not None

    hamiltonian: sim.GMatrix = system.hamiltonian_ti(domain)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    indexes: NDArray[np.int32] = np.argsort(eigenvalues).astype(np.int32)
    eigenvalues, eigenvectors = eigenvalues[indexes], eigenvectors[:, indexes]

    state: int = 0
    wavefunction: sim.GVector = num.norm_wavefunction(eigenvectors[:, state], domain)

    # Set up the time domain.
    t_min: float = 0.0
    t_max: float = 1.0

    t_num_points: int = 1000
    t_num_points += 1

    time_domain: sim.TimeGrid = sim.TimeGrid(t_min, t_max, t_num_points)

    # Set up the simulation.
    order_m: int = 5
    order_f: int = 10

    tolerance: float = 1e-16

    # Propagate the wavefunction (timed).
    time_start: float = time.time()
    wavefunctions: sim.CVectors = prop.propagate(
        domain, system, wavefunction, time_domain, order_m, order_f, tolerance
    )
    time_final: float = time.time()

    # Print the runtime.
    runtime: float = time_final - time_start
    print(f"Runtime: {runtime:.2f} seconds")

    # Save the wavefunctions.
    filename: str = "example_harmonic"
    np.savetxt(f"data/{filename}.txt", wavefunctions)

    # Calculate the norms and energies.
    norms: sim.RVector = num.wavefunctions_norms(wavefunctions, domain)
    energies: sim.RVector = num.wavefunctions_energies(
        wavefunctions, domain, system, time_domain
    )

    # Print the max deviation in the norms and energies.
    print(f"Max Norm Deviation: {np.max(np.abs(norms[0] - norms)):.2e}")
    print(f"Max Energy Deviation: {np.max(np.abs(energies[0] - energies)):.2e}")

    # Calculate the error from the exact solution.
    expectation_x: sim.RVector = cast(
        sim.RVector,
        np.trapezoid(
            ((np.abs(wavefunctions) ** 2) * domain.x_axis), dx=domain.x_dx, axis=1
        ),
    )
    expectation_x_exact: sim.RVector = (
        -0.5 * np.sin(time_domain.t_axis) * time_domain.t_axis
    )
    error: sim.RVector = expectation_x - expectation_x_exact

    # Print the max error from the exact solution.
    print(f"Max Error: {np.abs(np.max(error)):.2e}")

    # Generate figures and animation.
    vis.plot_wavefunctions(
        wavefunctions[[0, -1]], domain, ["Initial", "Final"], f"figures/{filename}.png"
    )
    vis.animate_wavefunctions(
        wavefunctions, domain, time_domain, f"animations/{filename}.mp4"
    )


if __name__ == "__main__":
    main()
