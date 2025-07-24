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
from scipy.fft import dct

# Import local modules.
import simulation as sim


def ch_coefficients(
    function_values: sim.GVector | sim.GVectors,
    type: int,
) -> sim.GVector | sim.GVectors:
    """
    Calculates the coefficients for a Chebyshev expansion of a function through
    the discrete cosine transform (DCT). The function being expanded should be
    evaluated on either Chebyshev-Gauss or Chebyshev-Lobatto nodes.

    + DCT-I     : Chebyshev-Lobatto
    + DCT-II    : Chebyshev-Gauss

    Parameters
    ----------
    function_values: simulation.GVector | simulation.GVectors
        The values of the function evaluated on either Chebyshev-Gauss or
        Chebyshev-Lobatto nodes. If the function is multi-dimensional, the
        expansion is taken to be along the zeroth axis.
    type: int
        The type of discrete cosine transform (DCT) to use. DCT-I should be
        used for functions evaluated on Chebyshev-Lobatto nodes, and DCT-II
        for functions evaluated on Chebyshev-Gauss nodes.

    Returns
    -------
    coefficients: simulation.GVector | simulation.GVector
        The Chebyshev expansion coefficients.
    """

    if type not in [1, 2]:
        raise ValueError("invalid DCT type")

    # Store the number of expansion terms.
    order: int = function_values.shape[0]

    # Perform the discrete cosine transform (DCT).
    coefficients: sim.GVector | sim.GVector = np.asarray(
        dct(function_values, type=type, axis=0, norm=None)
    )

    # Normalisation for DCT-I.
    if type == 1:
        coefficients /= order - 1
        coefficients[0] /= 2
        coefficients[-1] /= 2

    # Normalisation for DCT-II.
    elif type == 2:
        coefficients /= order
        coefficients[0] /= 2

    return coefficients


def ch_expansion(
    operator: sim.GMatrix, ket: sim.GVector, coefficients: sim.GVector
) -> sim.GVector:
    """
    Calculates the Chebyshev expansion of an operator through the recursion
    relation for the Chebyshev polynomials of the first kind. The number of
    expansion terms is taken to be the number of coefficients.

    Parameters
    ----------
    operator: simulation.GMatrix
        The operator being expanded (e.g. Hamiltonian). The eigenvalue domain
        of the operator should be in the interval [-1, 1].
    ket: simulation.GVector
        The ket (e.g. wavefunction) being acted upon by the operator.
    coefficients: simulation.GVector
        The Chebyshev expansion coefficients. The coefficients are expected to
        be the cosine transformed values of values generated from evaluating a
        function of the operator on Chebyshev-Gauss or Chebyshev-Lobatto nodes.
    """

    # Store the number of expansion terms.
    order: int = coefficients.shape[0]

    # Calculate the first two Chebyshev expansion polynomials.
    polynomial_minus_2: sim.GVector = ket.copy()
    polynomial_minus_1: sim.GVector = operator @ ket

    # Construct the starting expansion term.
    expansion: sim.GVector = (coefficients[0] * polynomial_minus_2) + (
        coefficients[1] * polynomial_minus_1
    )

    # Construct the complete expansion.
    for i in range(2, order):
        polynomial_n: sim.GVector = (
            2 * (operator @ polynomial_minus_1)
        ) - polynomial_minus_2
        expansion += coefficients[i] * polynomial_n

        polynomial_minus_2: sim.GVector = polynomial_minus_1
        polynomial_minus_1: sim.GVector = polynomial_n

    return expansion


def ch_gauss_nodes(num_nodes: int) -> sim.RVector:
    """
    Calculates the Chebyshev-Gauss nodes on the interval [-1, 1]. These are the
    root of a Chebyshev polynomial, excluding the endpoints -1 and 1. The nodes
    are calculated in ascending order.

    Parameters
    ----------
    num_nodes: int
        The number of Chebyshev-Gauss nodes to calculate.

    Returns
    -------
    nodes: simulation.RVector
        The Chebyshev-Gauss nodes.
    """

    # Generate the Chebyshev-Gauss nodes.
    nodes: sim.RVector = -np.cos(
        (np.pi * (np.arange(num_nodes, dtype=np.float64) + 0.5)) / num_nodes
    )

    return nodes


def ch_lobatto_nodes(num_nodes: int) -> sim.RVector:
    """
    Calculates the Chebyshev-Lobatto nodes on the interval [-1, 1]. These are
    the extrema of a Chebyshev polynomial, including the endpoints -1 and 1. The
    nodes are calculated in ascending order.

    Parameters
    ----------
    num_nodes: int
        The number of Chebyshev-Lobatto nodes to calculate.

    Returns
    -------
    nodes: simulation.RVector
        The Chebyshev-Lobatto nodes.
    """

    # Generate the Chebyshev-Lobatto nodes.
    nodes: sim.RVector = -np.cos(
        (np.pi * np.arange(num_nodes, dtype=np.float64)) / (num_nodes - 1)
    )

    return nodes


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


def rescale_vector(
    vector: sim.RVector, a: float, b: float
) -> tuple[sim.RVector, float, float]:
    """
    Rescales the domain of a real vector to the interval [a, b] using an affine
    transformation. This function also returns the factors used to perform the
    affine transformation.

    Parameters
    ----------
    vector: simulation.RVector
        The real vector to rescale.
    a: float
        The lower bound of the target interval.
    b: float
        The upper bound of the target interval.

    Returns
    -------
    vector_rs: simulation.RVector
        The rescaled real vector.
    scale: float
        The scale factor used in the affine transformation.
    shift: float
        The shift factor used in the affine transformation.
    """

    # Get the domain of the vector.
    vector_min: float = np.min(vector)
    vector_max: float = np.max(vector)

    # Calculate the scale and shift factors for the affine transformation.
    scale: float = (b - a) / (vector_max - vector_min)
    shift: float = ((a * vector_max) - (b * vector_min)) / (vector_max - vector_min)

    # Rescale the vector.
    vector_rs: sim.RVector = (scale * vector) + shift

    return vector_rs, scale, shift
