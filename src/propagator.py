"""
Implementation of the Semi-Global propagation scheme for the time-dependent
Schrödinger equation.

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import external modules.
import numpy as np
from scipy.special import factorial

# Import local modules.
import simulation as sim


def inhomogeneous_operator(
    nodes: sim.RVector, time: float, order: int, threshold: float, tolerance: float
) -> sim.CVector:
    """
    Evaluates the function of the inhomogeneous operator on a set of given
    nodes, at a specified time point. This function represents the residual of
    the time evolution operator, minus its truncated Taylor expansion.

    Parameters
    ----------
    nodes: simulation.RVector
        The real valued nodes to evaluate the function of the inhomogeneous
        operator on. In general, these should be either Chebyshev-Gauss or
        Chebyshev-Lobatto nodes (Chebyshev expansion).
    time: float
        The time point at which to evaluate the function of the inhomogeneous
        operator on.
    order: int
        The order of the truncated Taylor expansion of the inhomogeneous
        operator.
    threshold: float
        The threshold above which to use the exact definition of the function
        of the inhomogeneous operator. A stable definition is used if any of
        the (nodes x time) points fall below the threshold.
    tolerance: float
        The tolerance for the stable function definition of the inhomogeneous
        operator. This is used to truncate a Taylor series when convergence is
        reached.

    Returns
    -------
    simulation.CVector
        The evaluated inhomogeneous operator function values.
    """

    # Stable definition (Taylor).
    if np.any(np.abs(nodes * time) < threshold):
        term_sta_1: float = time**order

        term_sta_2: sim.CVector = np.zeros(nodes.shape[0], dtype=np.complex128)
        max_expansion: int = 100

        for i in range(max_expansion):
            term: sim.CVector = ((-1j * nodes * time) ** i) / factorial(i + order)
            term_sta_2 += term

            if np.all(np.abs(term) < tolerance):
                break

        return term_sta_1 * term_sta_2

    # Standard definition.
    else:
        term_std_1: sim.CVector = (-1j * nodes).astype(np.complex128) ** -order
        term_std_2: sim.CVector = np.exp(-1j * nodes * time)

        term_std_3: sim.CVector = np.zeros(nodes.shape[0], dtype=np.complex128)

        for i in range(order):
            term_std_3 += ((-1j * nodes * time) ** i) / factorial(i)

        return term_std_1 * (term_std_2 - term_std_3)


def inhomogeneous_kets(
    operator: sim.GMatrix, ket: sim.GVector, derivatives: sim.GVectors, order: int
) -> sim.GVectors:
    """
    Calculates the inhomogeneous kets which represent time-evolved states with
    corrections from time derivatives of an inhomogeneous term. The derivatives
    should be calculated through converting Chebyshev expansion coefficients or
    Newtonian interpolation coefficients to Taylor-like derivative terms.

    + Semi-Global Section 2.4.2

    Parameters
    ----------
    operator: simulation.GMatrix
        The operator (e.g. Hamiltonian) acting on the ket.
    ket: simulation.GVector
        The ket (e.g. wavefunction) being acted upon by the operator.
    derivatives: simulation.GVectors
        The Taylor-like derivatives of an inhomogeneous term. The zeroth axis
        is taken to be the axis of increasing derivative order.
    order: int
        The number of inhomogeneous kets to calculate (i.e. expansion order).

    Returns
    -------
    kets: simulation.GVectors
        The inhomogeneous kets.
    """

    # Set up the inhomogeneous kets.
    kets: sim.GVectors = np.zeros(
        (order, ket.shape[0]), dtype=np.result_type(operator, ket, derivatives)
    )
    kets[0] = ket.copy()

    # Calculate the inhomogeneous kets.
    for i in range(1, order):
        kets[i] = (operator @ kets[i - 1]) + derivatives[i - 1]

    return kets
