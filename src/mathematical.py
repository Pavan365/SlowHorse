"""
Core mathematical functions for implementing the Semi-Global propagation scheme
for the time-dependent Schrödinger equation.

Abbreviations
-------------
+ ch : Chebyshev
+ ne : Newtonian

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import standard modules.
from typing import cast

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
    coefficients: sim.GVector | sim.GVectors = np.asarray(
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
    Calculates the Chebyshev expansion of an operator acting on a ket through
    the recursion relation for the Chebyshev polynomials of the first kind. The
    number of expansion terms is taken to be the number of coefficients.

    Parameters
    ----------
    operator: simulation.GMatrix
        The operator (e.g. Hamiltonian) acting on the ket. The eigenvalue
        domain of the operator should be in the interval [-1, 1].
    ket: simulation.GVector
        The ket (e.g. wavefunction) being acted upon by the operator.
    coefficients: simulation.GVector
        The Chebyshev expansion coefficients. The coefficients are expected to
        be the cosine transformed values of values generated from evaluating a
        function of the operator on Chebyshev-Gauss or Chebyshev-Lobatto nodes.

    Returns
    -------
    expansion: simulation.GVector
        The expansion term resulting from the Chebyshev expansion of the
        operator acting on the ket.
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


def ch_ta_conversion(order: int, time_min: float, time_max: float) -> sim.RMatrix:
    """
    Calculates the square (lower triangular) conversion matrix for converting
    Chebyshev expansion coefficients to Taylor-like derivatives, across a time
    interval. The matrix is intended for use with coefficients resulting from
    sampling a function on Chebyshev-Lobatto nodes.

    Parameters
    ----------
    order: int
        The size of the conversion matrix, which corresponds to the highest
        Taylor-like derivative produced from the matrix.
    time_min: float
        The lower bound of the time interval.
    time_max: float
        The upper bound of the time interval.

    Returns
    -------
    conversion: simulation.RMatrix
        The conversion matrix.
    """

    # Calculate time interval information.
    time_sum: float = time_min + time_max
    time_dt: float = time_max - time_min

    # Calculate recurring coefficients.
    a: float = (2 * time_sum) / time_dt
    b: float = 4 / time_dt

    # Set up the conversion matrix.
    conversion: sim.RMatrix = np.zeros((order, order), dtype=np.float64)
    conversion[0, 0] = 1.0

    conversion[1, 0] = -time_sum / time_dt
    conversion[1, 1] = 2 / time_dt

    # Construct the complete matrix.
    for i in range(2, order):
        # Calculate the m = 0 term (Semi-Global Appendix C.2).
        conversion[i, 0] = -(a * conversion[i - 1, 0]) - conversion[i - 2, 0]

        # Calculate the 1 <= m <= n - 2 terms (Semi-Global Appendix C.2).
        for j in range(1, i - 1):
            conversion[i, j] = (
                (b * j * conversion[i - 1, j - 1])
                - (a * conversion[i - 1, j])
                - conversion[i - 2, j]
            )

        # Calculate the m = n - 1 term.
        conversion[i, i - 1] = (b * (i - 1) * conversion[i - 1, i - 2]) - (
            a * conversion[i - 1, i - 1]
        )

        # Calculate the m = n term.
        conversion[i, i] = b * i * conversion[i - 1, i - 1]

    return conversion


def ne_coefficients(
    nodes: sim.RVector,
    function_values: sim.GVector,
) -> sim.GVector:
    """
    Calculates the coefficients for a Newtonian interpolation expansion of a
    function through building a divided differences table, which is upper
    triangular and contains the coefficients on the main diagonal. The function
    being expanded should be evaluated on nodes in the target domain.

    Parameters
    ----------
    nodes: sim.RVector
        The nodes in the target domain that the function being expanded is
        evaluated on.
    function_values: sim.GVector
        The values of the function being expanded evaluated on the nodes in
        the target domain.

    Returns
    -------
    coefficients: sim.GVector
        The Newtonian interpolation coefficients.
    """

    # Store the number of expansion terms.
    order: int = nodes.shape[0]

    # Set up the divided differences table.
    table: sim.GMatrix = cast(
        sim.GMatrix, np.zeros((order, order), dtype=function_values.dtype)
    )
    table[0, :] = function_values

    # Construct the divided differences table (upper triangular).
    for i in range(1, order):
        for j in range(i, order):
            table[i, j] = (table[i - 1, j] - table[i - 1, j - 1]) / (
                nodes[j] - nodes[j - i]
            )

    # Store the Newtonian interpolation coefficients.
    coefficients: sim.GVector = table[
        np.arange(order, dtype=np.int32), np.arange(order, dtype=np.int32)
    ]

    return coefficients


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
