"""
Functions for plotting and animating wavefunctions.
"""

# Add source directory to path.
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
)

# Import external modules.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

# Import local modules.
import simulation as sim


def plot_wavefunctions(
    wavefunctions: sim.GVectors,
    domain: sim.HilbertSpace1D,
    labels: list[str],
    filename: str,
) -> None:
    """
    Plots the probability densities of a set of wavefunctions.

    Parameters
    ----------
    wavefunctions: simulation.GVectors
        The wavefunctions to plot.
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    labels: list[str]
        The labels for the wavefunctions.
    filename: str
        The filename to save the plot to.
    """

    # Calculate the probability densities.
    prob_densities: sim.RVectors = np.abs(wavefunctions) ** 2

    # Plot the probability densities.
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(0, np.max(prob_densities) * 1.05)

    for prob_density, label in zip(prob_densities, labels):
        ax.plot(domain.x_axis, prob_density, label=label)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\psi(x)|^{2}$")
    ax.legend(loc="upper right")

    fig.savefig(filename, dpi=300)
    plt.close(fig)


def animate_wavefunctions(
    wavefunctions: sim.GVectors,
    domain: sim.HilbertSpace1D,
    time_domain: sim.TimeGrid,
    filename: str,
) -> None:
    """
    Animates a set of wavefunctions over time. This function creates an
    animation that contains the probability densities of the wavefunctions
    along with the real and complex components.

    Parameters
    ----------
    wavefunctions: simulation.GVectors
        The wavefunctions to animate.
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    time_domain: simulation.TimeGrid
        The time domain (grid) over which the wavefunctions are defined.
    filename: str
        The filename to save the animation to.
    """

    # Calculate the probability densities.
    prob_densities: sim.RVectors = np.abs(wavefunctions) ** 2

    # Store the real and complex components.
    wavefunctions_real: sim.RVectors = np.real(wavefunctions)
    wavefunctions_imag: sim.RVectors = np.imag(wavefunctions)

    # Animate the wavefunctions.
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(-1.0, 1.0)

    (line_prob_density,) = ax.plot(
        domain.x_axis,
        prob_densities[0],
        label=r"$|\psi(x)|^{2}$",
        color="rebeccapurple",
    )
    (line_real,) = ax.plot(
        domain.x_axis,
        wavefunctions_real[0],
        "--",
        label=r"$\mathrm{Re}[\psi(x)]$",
        color="royalblue",
        alpha=0.75,
    )
    (line_imag,) = ax.plot(
        domain.x_axis,
        wavefunctions_imag[0],
        "--",
        label=r"$\mathrm{Im}[\psi(x)]$",
        color="crimson",
        alpha=0.75,
    )

    ax.set_title(r"$\text{Wavefunction} \; (T = 0.00)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\psi(x)|^2, \; \psi(x)$")
    ax.legend(loc="upper right")

    def animate(frame):
        line_prob_density.set_ydata(prob_densities[frame])
        line_real.set_ydata(wavefunctions_real[frame])
        line_imag.set_ydata(wavefunctions_imag[frame])

        ax.set_title(
            r"$\text{Wavefunction} \;$" + f"$(T = {time_domain.t_axis[frame]:.2f})$"
        )

        return line_prob_density, line_real, line_imag

    frames = range(0, wavefunctions.shape[0], int(wavefunctions.shape[0] * 0.01))
    fps, bitrate, dpi = 30, 2000, 200

    ani = FuncAnimation(fig, animate, frames, blit=True, interval=(1000 / fps))
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)

    ani.save(filename, writer, dpi=dpi)
    plt.close(fig)
