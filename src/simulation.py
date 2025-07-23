"""
Classes for setting up simulations.
"""

# Import external modules.
import numpy as np
from numpy.typing import NDArray


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
    x_axis: NDArray[np.float64]
        The discretised x-axis grid.
    x_dx: float
        The discretised x-axis grid spacing.
    """

    def __init__(self, x_min: float, x_max: float, num_points: int) -> None:
        """

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
        self.x_axis: NDArray[np.float64] = np.linspace(
            x_min, x_max, num_points, dtype=np.float64
        )
        self.x_dx: float = self.x_axis[1] - self.x_axis[0]
