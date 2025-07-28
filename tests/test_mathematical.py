"""
Simple test cases for the "src.mathematical" module.
"""

# Add source directory to path.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import external modules.
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

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

    print("Passed")
    print("------")


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

    print("Passed")
    print("------")


def test_ch_gauss_nodes() -> None:
    """
    Test that the "ch_gauss_nodes" function performs as expected.
    """

    print("Function Tested: src.mathematical.ch_gauss_nodes")
    print("------------------------------------------------")

    # Check that the Chebyshev-Gauss nodes are generated as expected.
    num_nodes: int = 20
    nodes: sim.RVector = math.ch_gauss_nodes(num_nodes)

    # Print the Chebyshev-Gauss nodes.
    print(nodes)

    # Plot the Chebyshev-Gauss nodes.
    plt.plot(nodes)
    plt.title("Chebyshev-Gauss Nodes")
    plt.show()

    print("Passed")
    print("------")


def test_ch_lobatto_nodes() -> None:
    """
    Test that the "ch_lobatto_nodes" function performs as expected.
    """

    print("Function Tested: src.mathematical.ch_lobatto_nodes")
    print("--------------------------------------------------")

    # Check that the Chebyshev-Lobatto nodes are generated as expected.
    num_nodes: int = 20
    nodes: sim.RVector = math.ch_lobatto_nodes(num_nodes)

    # Print the Chebyshev-Lobatto nodes.
    print(nodes)

    # Plot the Chebyshev-Lobatto nodes.
    plt.plot(nodes)
    plt.title("Chebyshev-Lobatto Nodes")
    plt.show()

    print("Passed")
    print("------")


def test_ch_coefficients() -> None:
    """
    Test that the "ch_coefficients" function performs as expected.
    """

    print("Function Tested: src.mathematical.ch_coefficients")
    print("-------------------------------------------------")

    # Define a known function.
    def function(x: sim.RVector) -> sim.RVector:
        return x**3

    # Define the Chebyshev expansion.
    def ch_expansion(nodes: sim.RVector, coefficients: sim.RVector) -> sim.RVector:
        order: int = coefficients.shape[0]

        polynomial_minus_2: sim.RVector = np.ones(order, dtype=np.float64)
        polynomial_minus_1: sim.RVector = nodes.copy()

        expansion: sim.RVector = (coefficients[0] * polynomial_minus_2) + (
            coefficients[1] * polynomial_minus_1
        )

        for i in range(2, order):
            polynomial_n: sim.RVector = (
                2 * nodes * polynomial_minus_1
            ) - polynomial_minus_2
            expansion += coefficients[i] * polynomial_n

            polynomial_minus_2: sim.RVector = polynomial_minus_1
            polynomial_minus_1: sim.RVector = polynomial_n

        return expansion

    # Construct the exact solution.
    x_axis: sim.RVector = np.linspace(-1, 1, 100, dtype=np.float64)
    function_exact: sim.RVector = function(x_axis)

    # Construct the approximate solution using Chebyshev-Gauss nodes.
    order: int = 20
    nodes: sim.RVector = math.ch_gauss_nodes(order)

    function_values: sim.RVector = function(nodes)
    function_coefficients: sim.RVector = math.ch_coefficients(
        function_values[::-1], dct_type=2
    ).astype(np.float64)
    function_expansion: sim.RVector = ch_expansion(nodes, function_coefficients)

    # Check that the coefficients are as expected.
    print(function_coefficients)

    assert np.isclose(function_coefficients[1], 0.75)
    assert np.isclose(function_coefficients[3], 0.25)

    mask: NDArray[np.bool_] = np.ones(order, dtype=np.bool_)
    mask[[1, 3]] = False
    assert np.allclose(function_coefficients[mask], 0.0)

    # Plot the exact and approximate solutions.
    plt.plot(x_axis, function_exact, label="Exact")
    plt.plot(nodes, function_expansion, linestyle="--", label="Expansion")
    plt.title("Chebyshev-Gauss Coefficients")
    plt.legend()
    plt.show()

    # Construct the approximate solution using Chebyshev-Lobatto nodes.
    order: int = 20
    nodes: sim.RVector = math.ch_lobatto_nodes(order)

    function_values: sim.RVector = function(nodes)
    function_coefficients: sim.RVector = math.ch_coefficients(
        function_values[::-1], dct_type=1
    ).astype(np.float64)
    function_expansion: sim.RVector = ch_expansion(nodes, function_coefficients)

    # Check that the coefficients are as expected.
    print(function_coefficients)

    assert np.isclose(function_coefficients[1], 0.75)
    assert np.isclose(function_coefficients[3], 0.25)

    mask: NDArray[np.bool_] = np.ones(order, dtype=np.bool_)
    mask[[1, 3]] = False
    assert np.allclose(function_coefficients[mask], 0.0)

    # Plot the exact and approximate solutions.
    plt.plot(x_axis, function_exact, label="Exact")
    plt.plot(nodes, function_expansion, linestyle="--", label="Expansion")
    plt.title("Chebyshev-Lobatto Coefficients")
    plt.legend()
    plt.show()

    print("Passed")
    print("------")


def test_ch_expansion() -> None:
    """
    Test that the "ch_expansion" function performs as expected.
    """

    print("Function Tested: src.mathematical.ch_expansion")
    print("----------------------------------------------")

    # Define the position operator.
    domain: sim.HilbertSpace1D = sim.HilbertSpace1D(-5.0, 5.0, 100)
    position: sim.RMatrix = np.diag(domain.x_axis)

    # Define a wavefunction.
    wavefunction = np.exp(-domain.x_axis**2)

    # Define the operator function.
    def function(x: sim.RVector) -> sim.RVector:
        return x**2

    # Calculate the exact solution.
    exact: sim.RVector = position @ (position @ wavefunction)

    # Calculate the approximate solution.
    position_rs, scale, shift = math.rescale_matrix(position, -1, 1)

    order: int = 10
    nodes: sim.RVector = (math.ch_gauss_nodes(order) - shift) / scale

    function_values: sim.RVector = function(nodes)
    function_coefficients: sim.RVector = math.ch_coefficients(
        function_values[::-1], dct_type=2
    ).astype(np.float64)

    expansion: sim.RVector = math.ch_expansion(
        position_rs, wavefunction, function_coefficients
    ).astype(np.float64)

    # Check that the approximation is close to the exact solution.
    assert np.allclose(exact, expansion)

    # Plot the exact and approximate solutions.
    plt.plot(domain.x_axis, exact, label="Exact")
    plt.plot(domain.x_axis, expansion, linestyle="--", label="Expansion")
    plt.title("Chebyshev Expansion")
    plt.legend()
    plt.show()

    print("Passed")
    print("------")


def test_ch_ta_conversion() -> None:
    """
    Test that the "ch_ta_conversion" function performs as expected.
    """

    print("Function Tested: src.mathematical.ch_ta_conversion")
    print("--------------------------------------------------")

    # Define a known function.
    def function(x: sim.RVector | float) -> sim.RVector | float:
        return (3 * (x**3)) + (2 * (x**2)) + 1

    # Define the derivatives of the function.
    def function_dx_01(x: float) -> float:
        return (9 * (x**2)) + (4 * x)

    def function_dx_02(x: float) -> float:
        return (18 * x) + 4

    def function_dx_03(x: float) -> float:
        return 18.0

    def function_dx_04(x: float) -> float:
        return 0.0

    # Define a domain.
    x_min: float = 0.0
    x_max: float = 1.0

    # Define the number of derivatives.
    order: int = 5

    # Define the exact derivatives at the minimum point of the domain.
    exact: sim.RVector = np.array(
        [
            function(x_min),
            function_dx_01(x_min),
            function_dx_02(x_min),
            function_dx_03(x_min),
            function_dx_04(x_min),
        ],
        dtype=np.float64,
    )

    # Calculate the approximate Taylor-like derivatives.
    nodes: sim.RVector = math.ch_lobatto_nodes(order)
    nodes, scale, shift = math.rescale_vector(nodes, x_min, x_max)

    function_values: sim.RVector = np.asarray(function(nodes), dtype=np.float64)
    coefficients: sim.RVector = math.ch_coefficients(
        function_values[::-1], dct_type=1
    ).astype(np.float64)

    conversion_matrix: sim.RMatrix = math.ch_ta_conversion(order, x_min, x_max)
    derivatives: sim.RMatrix = conversion_matrix.T @ coefficients

    # Print the conversion matrix.
    print(conversion_matrix)

    # Check that the exact and approximated derivatives are similar.
    print(exact)
    with np.printoptions(formatter={"float": "{: 0.5e}".format}):
        print(derivatives)

    assert np.allclose(exact, derivatives)

    print("Passed")
    print("------")


def test_ne_coefficients() -> None:
    """
    Test that the "ne_coefficients" function performs as expected.
    """

    print("Function Tested: src.mathematical.ne_coefficients")
    print("-------------------------------------------------")

    # Define a known function.
    def function(x: sim.RVector) -> sim.RVector:
        return x**3

    # Define the Newtonian interpolation expansion.
    def ne_expansion(nodes: sim.RVector, coefficients: sim.RVector) -> sim.RVector:
        order: int = coefficients.shape[0]

        polynomial_n: sim.RVector = np.ones(order, dtype=np.float64)
        expansion: sim.RVector = coefficients[0] * polynomial_n

        for i in range(1, order):
            polynomial_n *= nodes - nodes[i - 1]
            expansion += coefficients[i] * polynomial_n

        return expansion

    # Construct the exact solution.
    x_axis: sim.RVector = np.linspace(-1, 1, 100, dtype=np.float64)
    function_exact: sim.RVector = function(x_axis)

    # Construct the approximate solution using Chebyshev-Lobatto nodes.
    order: int = 20
    nodes: sim.RVector = math.ch_lobatto_nodes(order)

    function_values: sim.RVector = function(nodes)
    function_coefficients: sim.RVector = math.ne_coefficients(
        nodes, function_values
    ).astype(np.float64)
    function_expansion: sim.RVector = ne_expansion(nodes, function_coefficients)

    # Print the coefficients.
    print(function_coefficients)

    # Plot the exact and approximate solutions.
    plt.plot(x_axis, function_exact, label="Exact")
    plt.plot(nodes, function_expansion, linestyle="--", label="Expansion")
    plt.title("Newtonian Coefficients")
    plt.legend()
    plt.show()

    print("Passed")
    print("------")


def test_ne_ta_conversion() -> None:
    """
    Test that the "ne_ta_conversion" function performs as expected.
    """

    print("Function Tested: src.mathematical.ne_ta_conversion")
    print("--------------------------------------------------")

    # Define a known function.
    def function(x: sim.RVector | float) -> sim.RVector | float:
        return (3 * (x**3)) + (2 * (x**2)) + 1

    # Define the derivatives of the function.
    def function_dx_01(x: float) -> float:
        return (9 * (x**2)) + (4 * x)

    def function_dx_02(x: float) -> float:
        return (18 * x) + 4

    def function_dx_03(x: float) -> float:
        return 18.0

    def function_dx_04(x: float) -> float:
        return 0.0

    # Define a domain.
    x_min: float = 0.0
    x_max: float = 1.0

    # Define the number of derivatives.
    order: int = 5

    # Define the exact derivatives at the minimum point of the domain.
    exact: sim.RVector = np.array(
        [
            function(x_min),
            function_dx_01(x_min),
            function_dx_02(x_min),
            function_dx_03(x_min),
            function_dx_04(x_min),
        ],
        dtype=np.float64,
    )

    # Calculate the approximate Taylor-like derivatives.
    nodes: sim.RVector = math.ch_lobatto_nodes(order)
    nodes, scale, shift = math.rescale_vector(nodes, x_min, x_max)

    function_values: sim.RVector = np.asarray(function(nodes), dtype=np.float64)
    coefficients: sim.RVector = math.ne_coefficients(nodes, function_values).astype(
        np.float64
    )

    conversion_matrix: sim.RMatrix = math.ne_ta_conversion(nodes)
    derivatives: sim.RMatrix = conversion_matrix.T @ coefficients

    # Print the conversion matrix.
    print(conversion_matrix)

    # Check that the exact and approximated derivatives are similar.
    print(exact)
    with np.printoptions(formatter={"float": "{: 0.5e}".format}):
        print(derivatives)

    assert np.allclose(exact, derivatives)

    print("Passed")
    print("------")


if __name__ == "__main__":
    # Run test cases.
    test_rescale_matrix()
    test_rescale_vector()
    test_ch_gauss_nodes()
    test_ch_lobatto_nodes()
    test_ch_coefficients()
    test_ch_expansion()
    test_ch_ta_conversion()
    test_ne_coefficients()
    test_ne_ta_conversion()
