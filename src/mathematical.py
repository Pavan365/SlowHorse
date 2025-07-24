"""
Core mathematical functions for implementing the Semi-Global propagation scheme
for the time-dependent Schrödinger equation.

Abbreviations
-------------
+ ch : Chebyshev

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import external modules.
import numpy as np

# Import local modules.
import simulation as sim


def rescale_matrix(
    matrix: sim.GMatrix, a: float, b: float
) -> tuple[sim.GMatrix, float, float]:
    """
    Rescales the eigenvalue domain of a Hermitian matrix to the interval [a, b]
    using an affine transformation. This function also returns the factors used
    to perform the affine transformation.

    Parameters
    ----------
    matrix: simulation.GMatrix
        The Hermitian matrix to rescale.
    a: float
        The lower bound of the target interval.
    b: float
        The upper bound of the target interval.

    Returns
    -------
    matrix_rs: simulation.GMatrix
        The rescaled Hermitian matrix.
    scale: float
        The scale factor used in the affine transformation.
    shift: float
        The shift factor used in the affine transformation.
    """

    # Get the eigenvalues.
    eigenvalues: sim.RVector = np.sort(np.linalg.eigvalsh(matrix)).astype(np.float64)
    eigenvalues_min: float = eigenvalues[0]
    eigenvalues_max: float = eigenvalues[-1]

    # Calculate the scale and shift factors for the affine transformation.
    scale: float = (b - a) / (eigenvalues_max - eigenvalues_min)
    shift: float = ((a * eigenvalues_max) - (b * eigenvalues_min)) / (
        eigenvalues_max - eigenvalues_min
    )

    # Rescale the matrix.
    matrix_rs: sim.GMatrix = (scale * matrix) + (
        shift * np.identity(matrix.shape[0], dtype=matrix.dtype)
    )

    return matrix_rs, scale, shift
