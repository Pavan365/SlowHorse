"""
Simple test cases for the "src.mathematical" module.
"""

# Add source directory to path.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import external modules.
import numpy as np

# Import local modules.
import mathematical as math
import simulation as sim


def test_rescale_matrix() -> None:
    """
    Test that the "rescale_matrix" function performs as expected.
    """

    # Construct a simple matrix.
    diagonal: sim.RVector = np.linspace(-200, 200, 100, dtype=np.float64)
    matrix: sim.RMatrix = np.diag(diagonal)

    # Define the target interval.
    a: float = -1.0
    b: float = 1.0

    # Rescale the eigenvalue domain of the matrix.
    matrix_rs, scale, shift = math.rescale_matrix(matrix, a, b)

    # Check that the rescaled eigenvalue domain is as expected.
    eigenvalues_rs: sim.RVector = np.linalg.eigvalsh(matrix_rs).astype(np.float64)
    print(eigenvalues_rs)

    assert np.isclose(np.min(eigenvalues_rs), a)
    assert np.isclose(np.max(eigenvalues_rs), b)

    # Check that the scale and shift factors perform the correct inverse transformation.
    matrix_original: sim.GMatrix = (
        matrix_rs - (shift * np.identity(matrix.shape[0], dtype=matrix.dtype))
    ) / scale
    assert np.allclose(matrix, matrix_original)


if __name__ == "__main__":
    # Run test cases.
    test_rescale_matrix()
