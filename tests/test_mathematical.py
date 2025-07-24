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

    print("Function Tested: src.mathematical.rescale_matrix")
    print("------------------------------------------------")

    # Construct a simple matrix.
    diagonal: sim.RVector = np.linspace(-200, 200, 20, dtype=np.float64)
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


def test_rescale_vector() -> None:
    """
    Test that the "rescale_vector" function performs as expected.
    """

    print("Function Tested: src.mathematical.rescale_vector")
    print("------------------------------------------------")

    # Construct a simple vector.
    vector: sim.RVector = np.linspace(-200, 200, 20, dtype=np.float64)

    # Define the target interval.
    a: float = -1.0
    b: float = 1.0

    # Rescale the domain of the vector.
    vector_rs, scale, shift = math.rescale_vector(vector, a, b)

    # Check that the rescaled vector is as expected.
    print(vector_rs)

    assert np.isclose(np.min(vector_rs), a)
    assert np.isclose(np.max(vector_rs), b)

    # Check that the scale and shift factors perform the correct inverse transformation.
    vector_original: sim.RVector = (vector_rs - shift) / scale
    assert np.allclose(vector, vector_original)


if __name__ == "__main__":
    # Run test cases.
    test_rescale_matrix()
    test_rescale_vector()
