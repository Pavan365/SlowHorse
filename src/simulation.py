"""
Classes for setting up simulations.
"""

# Import standard modules.
from typing import Callable, TypeAlias

# Import external modules.
import numpy as np
from numpy.typing import NDArray

# Generalised type aliases for matrices, vectors and collections of matrices/vectors.
# G -> General, R -> Real, C -> Complex.
GMatrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RMatrix: TypeAlias = NDArray[np.float64]
CMatrix: TypeAlias = NDArray[np.complex128]

GMatrices: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RMatrices: TypeAlias = NDArray[np.float64]
CMatrices: TypeAlias = NDArray[np.complex128]

GVector: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RVector: TypeAlias = NDArray[np.float64]
CVector: TypeAlias = NDArray[np.complex128]

GVectors: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RVectors: TypeAlias = NDArray[np.float64]
CVectors: TypeAlias = NDArray[np.complex128]


class HilbertSpace1D:
    """
    Represents a discretised Hilbert space in 1 dimension.

    Attributes
    ----------
    x_min: float
        The minimum x-axis value.
    x_max: float
        The maximum x-axis value.
    num_points: int
        The number of sampling points for the discretised x-axis grid.
    x_axis: RVector
        The discretised x-axis grid.
    x_dx: float
        The discretised x-axis grid spacing.
    """

    def __init__(self, x_min: float, x_max: float, num_points: int) -> None:
        """
        Initialises an instance of the HilbertSpace1D class.

        Parameters
        ----------
        x_min: float
            The minimum x-axis value.
        x_max: float
            The maximum x-axis value.
        num_points: int
            The number of sampling points for the discretised x-axis grid.
        """

        # Assign attributes.
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.num_points: int = num_points

        # Define the discretised x-axis grid.
        self.x_axis: RVector = np.linspace(x_min, x_max, num_points, dtype=np.float64)
        self.x_dx: float = self.x_axis[1] - self.x_axis[0]


# Type aliases for time-dependent Schrödinger equation terms.
Operator: TypeAlias = Callable[[GVector, HilbertSpace1D, float], GVector]
OperatorRs: TypeAlias = Callable[[GVector, HilbertSpace1D, float], GVector]
OperatorDiff: TypeAlias = Callable[
    [
        GVector,
        HilbertSpace1D,
        float,
        float,
    ],
    GVector,
]


class TDSE1D:
    """
    Represents a time-dependent Schrödinger equation in 1 dimension.

    Attributes
    ----------
    hamiltonian: Operator
        A function that returns the action of a Hamiltonian on a state.
    hamiltonian_rs: OperatorRs
        A function that returns the action of a Hamiltonian on a state,
        rescaled to the domain [-1, 1].
    hamiltonian_diff: OperatorDiff
        A function that returns the difference in the action of a Hamiltonian
        on a state, at two different times.
    eigenvalue_min: float
        The minimum eigenvalue of the Hamiltonian (approximate).
    eigenvalue_max: float
        The maximum eigenvalue of the Hamiltonian (approximate).
    """

    def __init__(
        self,
        hamiltonian: Operator,
        hamiltonian_diff: OperatorDiff,
        eigenvalue_min: float,
        eigenvalue_max: float,
    ) -> None:
        """
        Initialises an instance of the TDSE1D class.

        Parameters
        ----------
        hamiltonian: Hamiltonian
            A function that returns the action of a Hamiltonian on a state.
        hamiltonian_diff: HamiltonianDiff
            A function that returns the difference in the action of a
            Hamiltonian on a state, at two different times.
        eigenvalue_min: float
            The minimum eigenvalue of the Hamiltonian (approximate).
        eigenvalue_max: float
            The maximum eigenvalue of the Hamiltonian (approximate).
        """

        # Assign attributes.
        self.hamiltonian: Operator = hamiltonian
        self.hamiltonian_diff: OperatorDiff = hamiltonian_diff

        self.eigenvalue_min: float = eigenvalue_min
        self.eigenvalue_max: float = eigenvalue_max

    def hamiltonian_rs(
        self, ket: GVector, domain: HilbertSpace1D, time: float
    ) -> GVector:
        """
        Calculates the action of the Hamiltonian on a state, rescaled to the
        domain [-1, 1]. This is useful when performing a Chebyshev expansion of
        the Hamiltonian operator.

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
        GVector
            The result of acting the Hamiltonian on the given state, rescaled
            to the domain [-1, 1].
        """

        return (
            (2 * self.hamiltonian(ket, domain, time))
            - ((self.eigenvalue_min + self.eigenvalue_max) * ket)
        ) / (self.eigenvalue_max - self.eigenvalue_min)


class TimeGrid:
    """
    Represents a discretised time interval.

    Attributes
    ----------
    t_min: float
        The minimum time-axis value.
    t_max: float
        The maximum time-axis value.
    num_points: int
        The number of sampling points for the discretised time-axis grid.
    t_axis: RVector
        The discretised time-axis grid.
    t_dt: float
        The discretised time-axis grid spacing.
    """

    def __init__(self, t_min: float, t_max: float, num_points: int) -> None:
        """
        Initialises an instance of the TimeGrid class.

        Parameters
        ----------
        t_min: float
            The minimum time-axis value.
        t_max: float
            The maximum time-axis value.
        num_points: int
            The number of sampling points for the discretised time-axis grid.
        """

        # Assign attributes.
        self.t_min: float = t_min
        self.t_max: float = t_max
        self.num_points: int = num_points

        # Define the discretised time-axis grid.
        self.t_axis: RVector = np.linspace(t_min, t_max, num_points, dtype=np.float64)
        self.t_dt: float = self.t_axis[1] - self.t_axis[0]
