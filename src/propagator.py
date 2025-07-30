"""
Implementation of the Semi-Global propagation scheme for the time-dependent
Schrödinger equation.

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import standard modules.
from enum import Enum
from typing import cast

# Import external modules.
import numpy as np
from scipy.special import factorial

# Import local modules.
import mathematical as math
import simulation as sim


class ApproximationBasis(Enum):
    """
    Enumeration of the available approximation bases for the inhomogeneous
    term in the Semi-Global propagation scheme.

    Members
    -------
    CHEBYSHEV: str
        Represents a Chebyshev expansion of the inhomogeneous term.
    NEWTONIAN: str
        Represents a Newtonian interpolation expansion of the inhomogeneous
        term.
    """

    CHEBYSHEV = "ch"
    NEWTONIAN = "ne"


def inhomogeneous_operator(
    nodes: sim.RVector, time: float, order: int, threshold: float, tolerance: float
) -> sim.CVector:
    """
    Evaluates the function of the inhomogeneous operator on a set of given
    nodes, at a specified time point. This function represents the residual of
    the time evolution operator, minus its truncated Taylor expansion.

    + Semi-Global Section 2.3.2

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

        return factorial(order) * term_sta_1 * term_sta_2

    # Standard definition.
    else:
        term_std_1: sim.CVector = (-1j * nodes).astype(np.complex128) ** -order
        term_std_2: sim.CVector = np.exp(-1j * nodes * time)

        term_std_3: sim.CVector = np.zeros(nodes.shape[0], dtype=np.complex128)

        for i in range(order):
            term_std_3 += ((-1j * nodes * time) ** i) / factorial(i)

        return factorial(order) * term_std_1 * (term_std_2 - term_std_3)


def inhomogeneous_kets(
    operator: sim.Operator,
    domain: sim.HilbertSpace1D,
    time: float,
    ket: sim.GVector,
    derivatives: sim.GVectors,
    order: int,
) -> sim.CVectors:
    """
    Calculates the inhomogeneous kets which represent time-evolved states with
    corrections from time derivatives of an inhomogeneous term. The derivatives
    should be calculated through converting Chebyshev expansion coefficients or
    Newtonian interpolation coefficients to Taylor-like derivative terms.

    + Semi-Global Section 2.4.2

    Parameters
    ----------
    operator: simulation.Operator
        A function that returns the action of an operator (e.g. Hamiltonian) on
        a state.
    domain: HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    time: float
        The time at which to evaluate the operator.
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
    kets: sim.CVectors = np.zeros((order, ket.shape[0]), dtype=np.complex128)
    kets[0] = ket.copy()

    # Calculate the inhomogeneous kets.
    for i in range(1, order):
        kets[i] = (operator(kets[i - 1], domain, time) + derivatives[i - 1]) / i

    return kets


def propagate(
    domain: sim.HilbertSpace1D,
    system: sim.TDSE1D,
    wavefunction: sim.GVector,
    time_domain: sim.TimeGrid,
    order_m: int,
    order_f: int,
    tolerance: float,
    approximation: ApproximationBasis,
) -> sim.CVectors:
    """
    Propagates a wavefunction with respect to the time-dependent Schrödinger
    equation (TDSE) using the Semi-Global propagation scheme. This propagation
    scheme is intended to be used for systems with time-dependent Hamiltonians.

    Parameters
    ----------
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    system: simulation.TDSE1D
        The time-dependent Schrödinger equation (TDSE) of the system.
    wavefunction: simulation.GVector
        The wavefunction to propagate with respect to the time-dependent
        Schrödinger equation (TDSE).
    time_domain: simulation.TimeGrid
        The time domain (grid) over which to propagate the wavefunction.
    order_m: int
        The order of the truncated Taylor expansion, which approximates the
        homogeneous propagation term with corrections from the time derivatives
        of an inhomogeneous term.
    order_f: int
        The order of the Chebyshev expansion of the inhomogeneous operator
        function, which represents a correction (residual) to the Taylor
        expansion.
    tolerance: float
        The tolerance of the propagator. This is used to define the convergence
        criterion during iterative time ordering.
    approximation: ApproximationBasis
        The approximation basis to use for the inhomogeneous term. This can
        either be a Chebyshev expansion or a Newtonian interpolation expansion.

    Returns
    -------
    wavefunctions: simulation.CVectors
        The propagated wavefunctions.
    """

    ## NOTE: GLOBAL PRE-COMPUTATIONS
    # Create a vector to store the wavefunctions.
    wavefunctions: sim.CVectors = np.zeros(
        (time_domain.num_points, domain.num_points), dtype=np.complex128
    )
    wavefunctions[0] = wavefunction.copy()

    # Set up the Chebyshev-Gauss and Chebyshev-Lobatto nodes.
    cg_nodes: sim.RVector = math.ch_gauss_nodes(order_f)
    cl_nodes: sim.RVector = math.ch_lobatto_nodes(order_m)

    # Set up the entire time grid of Chebyshev-Lobatto nodes.
    cl_nodes_rs: sim.RVector = math.rescale_vector(
        cl_nodes, time_domain.t_axis[0], time_domain.t_axis[1]
    )[0]

    t_nodes: sim.RVector = np.zeros(
        (((order_m - 1) * (time_domain.num_points - 1)) + 1), dtype=np.float64
    )
    t_nodes[0] = time_domain.t_axis[0]
    t_nodes[1:] = (
        cl_nodes_rs[1::]
        + (np.arange(time_domain.num_points - 1) * time_domain.t_dt).reshape(-1, 1)
    ).flatten(order="C")

    # Set up the Chebyshev-Gauss nodes for expanding the inhomogeneous operator.
    f_nodes: sim.RVector = math.rescale_vector(
        cg_nodes, system.eigenvalue_min, system.eigenvalue_max
    )[0]

    # Evaluate the inhomogeneous operator on the first two time intervals.
    # These can be reused because the effective delta-time is constant.
    function_values: sim.CVectors = np.zeros((order_m, order_f), dtype=np.complex128)
    function_values_next: sim.CVectors = np.zeros(
        (order_m, order_f), dtype=np.complex128
    )

    for i in range(order_m):
        function_values[i] = inhomogeneous_operator(
            f_nodes, t_nodes[i], order_m, threshold=1e-2, tolerance=1e-16
        )
        function_values_next[i] = inhomogeneous_operator(
            f_nodes,
            t_nodes[i + (order_m - 1)],
            order_m,
            threshold=1e-2,
            tolerance=1e-16,
        )

    # Calculate the Chebyshev expansion coefficients of the inhomogeneous operator.
    function_coefficients: sim.CVectors = (
        math.ch_coefficients(function_values.T[::-1], dct_type=2)
        .astype(np.complex128)
        .T
    )
    function_coefficients_next: sim.CVectors = (
        math.ch_coefficients(function_values_next.T[::-1], dct_type=2)
        .astype(np.complex128)
        .T
    )

    # Generate the conversion matrix.
    # To convert expansion coefficients to Taylor-like derivatives.
    conversion_matrix: sim.RMatrix = np.zeros((order_m, order_m), dtype=np.float64)

    # Chebyshev expansion conversion.
    if approximation == ApproximationBasis.CHEBYSHEV:
        conversion_matrix = math.ch_ta_conversion(
            order_m, time_domain.t_axis[0], time_domain.t_axis[1]
        )

    # Newtonian expansion conversion.
    else:
        conversion_matrix = math.ne_ta_conversion(cl_nodes_rs)

    ## NOTE: STEP 1
    # Set the guess wavefunctions for the first time step.
    wf_guesses: sim.CVector = np.zeros(
        (order_m, domain.num_points), dtype=np.complex128
    )
    wf_guesses[:] = wavefunction.copy()

    ## NOTE: STEP 2
    # Propagate the wavefunction.
    for i in range(time_domain.num_points - 1):
        ## NOTE: STEP 2.A
        # Store the time interval information.
        t_start_idx: int = i * (order_m - 1)
        t_final_idx: int = (order_m - 1) + (i * (order_m - 1))

        ## NOTE: LOCAL PRE-COMPUTES (TIME-INTERVAL SPECIFIC)
        # Store the Chebyshev-Lobatto nodes.
        t_nodes_current: sim.RVector = t_nodes[t_start_idx : (t_final_idx + 1)]
        t_mid = t_nodes_current[order_m // 2]

        # Set up the inhomogeneous kets (variable scoping).
        lambdas: sim.CVectors = np.zeros(
            (order_m, domain.num_points), dtype=np.complex128
        )

        # Set the starting convergence.
        convergence: float = np.inf

        # Define a counter and set the maximum number of iterations.
        count: int = 0
        max_iters: int = 20

        ## NOTE: STEP 2.C
        # Propagate the wavefunction until convergence is reached.
        while convergence > tolerance and count < max_iters:
            ## NOTE: STEP 2.C.I
            # Set up the inhomogeneous terms.
            inhomogeneous_values: sim.CVectors = np.zeros(
                (order_m, domain.num_points), dtype=np.complex128
            )
            for j in range(order_m):
                inhomogeneous_values[j] = system.hamiltonian_diff(
                    wf_guesses[j], domain, t_nodes_current[j], t_mid
                )

            ## NOTE: STEP 2.C.II
            # Calculate the expansion coefficients of the inhomogeneous terms.
            inhomogeneous_coefficients: sim.CVectors = np.zeros(
                (order_m, domain.num_points), dtype=np.complex128
            )

            # Chebyshev expansion coefficients.
            if approximation == ApproximationBasis.CHEBYSHEV:
                inhomogeneous_coefficients = math.ch_coefficients(
                    inhomogeneous_values[::-1], dct_type=1
                ).astype(np.complex128)

            # Newtonian expansion coefficients.
            else:
                # Rescale to a length four domain.
                t_nodes_current_d4: sim.RVector = (
                    4.0 / time_domain.t_dt
                ) * t_nodes_current

                for j in range(domain.num_points):
                    inhomogeneous_coefficients[:, j] = math.ne_coefficients(
                        t_nodes_current_d4, inhomogeneous_values[:, j]
                    )

            ## NOTE: STEP 2.C.III
            # Calculate the Taylor-like derivative terms.
            taylor_derivatives: sim.CVectors = (
                conversion_matrix.T @ inhomogeneous_coefficients
            )

            ## NOTE: STEP 2.C.IV
            # Calculate the inhomogeneous kets (lambdas).
            lambdas: sim.CVectors = inhomogeneous_kets(
                system.hamiltonian,
                domain,
                t_mid,
                wf_guesses[0],
                taylor_derivatives,
                (order_m + 1),
            ).astype(np.complex128)

            ## NOTE: STEP 2.C.V
            # Store the previous guess wavefunction.
            wf_guess_old: sim.CVector = wf_guesses[-1].copy()

            ## NOTE: STEP 2.C.VI
            # Build the next set of guess wavefunctions.
            for j in range(1, order_m):
                # Store the time interval information.
                t_dt_m = t_nodes[j]

                operator_term: sim.CVector = math.ch_expansion(
                    system.hamiltonian_rs,
                    domain,
                    t_mid,
                    lambdas[-1],
                    function_coefficients[j],
                ).astype(np.complex128)

                # Calculate the truncated Taylor expansion.
                taylor_term: sim.CVector = np.zeros(
                    domain.num_points, dtype=np.complex128
                )
                for k in range(order_m):
                    taylor_term += (t_dt_m**k) * lambdas[k]

                # Store the new guess wavefunction.
                wf_guesses[j] = operator_term + taylor_term

            ## NOTE: STEP 2.C.VII
            # Calculate the convergence.
            wf_guess_new: sim.CVector = wf_guesses[-1].copy()
            convergence: float = cast(
                float,
                np.linalg.norm(wf_guess_new - wf_guess_old)
                / np.linalg.norm(wf_guess_old),
            )

            # Update the number of iterations.
            count += 1

            # NOTE: DIAGNOSTIC
            # Print diagnostic information.
            print(f"Time Index: {i} \t Iteration: {count}")
            print(f"Convergence: {convergence:.5e}")

        # If convergence failed, raise an error.
        if count >= max_iters:
            raise ValueError("convergence failed")

        ## NOTE: STEPS 2.D & 2.E
        # Store the propagated wavefunction.
        wavefunctions[i + 1] = wf_guesses[-1].copy()

        ## NOTE: STEP 2.F
        # Calculate the guess wavefunctions for the next time interval.
        if i < time_domain.num_points - 2:
            # Set the first guess wavefunction.
            wf_guesses[0] = wf_guesses[-1].copy()

            # Build the next set of guess wavefunctions.
            for j in range(1, order_m):
                # Store the time interval information.
                t_dt_next_m = t_nodes[j] + time_domain.t_dt

                operator_term_next: sim.CVector = math.ch_expansion(
                    system.hamiltonian_rs,
                    domain,
                    t_mid,
                    lambdas[-1],
                    function_coefficients_next[j],
                ).astype(np.complex128)

                # Calculate the truncated Taylor expansion.
                taylor_term_next: sim.CVector = np.zeros(
                    domain.num_points, dtype=np.complex128
                )
                for k in range(order_m):
                    taylor_term_next += (t_dt_next_m**k) * lambdas[k]

                # Store the new guess wavefunction.
                wf_guesses[j] = operator_term_next + taylor_term_next

    return wavefunctions
