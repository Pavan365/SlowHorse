"""
Classes for setting up simulations.
"""

# Import standard modules.
from enum import Enum
from typing import Callable, Optional, TypeAlias

# Import external modules.
import numpy as np
from numpy.typing import NDArray

# Generalised type aliases for matrices, vectors and collections of vectors.
# G -> General, R -> Real, C -> Complex.
GMatrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RMatrix: TypeAlias = NDArray[np.float64]
CMatrix: TypeAlias = NDArray[np.complex128]

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
HamiltonianTI: TypeAlias = Callable[[HilbertSpace1D], GMatrix]
HamiltonianTD: TypeAlias = Callable[[HilbertSpace1D, float], GMatrix]
InhomogeneousTerm: TypeAlias = Callable[[HilbertSpace1D, float], GVector]


class TDSE1D:
    """
    Represents a time-dependent Schrödinger equation in 1 dimension.

    Attributes
    ----------
    hamiltonian_ti: Optional[HamiltonianTI]
        A function that returns a Hamiltonian term that is time-independent.
    hamiltonian_td: Optional[HamiltonianTD]
        A function that returns a Hamiltonian term that is time-dependent.
    inhomogeneous_term: Optional[InhomogeneousTerm]
        A function that returns an inhomogeneous term.
    """

    def __init__(
        self,
        hamiltonian_ti: Optional[HamiltonianTI] = None,
        hamiltonian_td: Optional[HamiltonianTD] = None,
        inhomogeneous_term: Optional[InhomogeneousTerm] = None,
    ) -> None:
        """
        Initialises an instance of the TDSE1D class.

        Parameters
        ----------
        hamiltonian_ti: Optional[HamiltonianTI]
            A function that returns a Hamiltonian term that is time-independent.
        hamiltonian_td: Optional[HamiltonianTD]
            A function that returns a Hamiltonian term that is time-dependent.
        inhomogeneous_term: Optional[InhomogeneousTerm]
            A function that returns an inhomogeneous term.
        """

        if hamiltonian_ti is None and hamiltonian_td is None:
            raise ValueError("a Hamiltonian function must be provided")

        self.hamiltonian_ti: Optional[HamiltonianTI] = hamiltonian_ti
        self.hamiltonian_td: Optional[HamiltonianTD] = hamiltonian_td
        self.inhomogeneous_term: Optional[InhomogeneousTerm] = inhomogeneous_term
