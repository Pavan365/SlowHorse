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


def main():
    # Set up the spatial domain.
    x_lim: float = 10.0
    x_num_points: int = 512

    domain: sim.HilbertSpace1D = sim.HilbertSpace1D(-x_lim, x_lim, x_num_points)

    # Set up the wavefunction.
    hamiltonian: sim.GMatrix = ho.hamiltonian_standard(domain)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    indexes: NDArray[np.int32] = np.argsort(eigenvalues).astype(np.int32)
    eigenvalues, eigenvectors = eigenvalues[indexes], eigenvectors[:, indexes]

    state: int = 0
    wavefunction: sim.GVector = num.norm_wavefunction(eigenvectors[:, state], domain)

    # Set up the system.
    eigenvalue_min: float = eigenvalues[0]
    eigenvalue_max: float = eigenvalues[-1]

    def operator(
        ket: sim.GVector, domain: sim.HilbertSpace1D, time: float
    ) -> sim.CVector:
        return -1j * ho.hamiltonian_driven(ket, domain, time)

    def operator_diff(
        ket: sim.GVector, domain: sim.HilbertSpace1D, time_1: float, time_2: float
    ) -> sim.CVector:
        return -1j * ho.hamiltonian_driven_diff(ket, domain, time_1, time_2).astype(
            np.complex128
        )

    system: sim.TDSE1D = sim.TDSE1D(
        operator,
        operator_diff,
        eigenvalue_min,
        eigenvalue_max,
    )

    # Set up the time domain.
    t_min: float = 0.0
    t_max: float = 10.0

    t_num_points: int = 10000
    t_num_points += 1

    time_domain: sim.TimeGrid = sim.TimeGrid(t_min, t_max, t_num_points)

    # Set up the simulation.
    order_m: int = 10
    order_f: int = 10

    tolerance: float = 1e-5
    approximation: prop.ApproximationBasis = prop.ApproximationBasis.NEWTONIAN

    # Propagate the wavefunction (timed).
    time_start: float = time.time()
    wavefunctions: sim.CVectors = prop.propagate(
        domain,
        system,
        wavefunction,
        time_domain,
        order_m,
        order_f,
        tolerance,
        approximation,
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
