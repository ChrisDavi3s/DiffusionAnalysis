import numpy as np
import matplotlib.pyplot as plt
from ..trajectory import PositionTrajectory
from typing import Optional, Union, Tuple, List

class RDFAnalysis:
    def __init__(self, position_trajectory: PositionTrajectory):
        """
        Initialize the RDFAnalysis class.

        Args:
            position_trajectory (PositionTrajectory): The position trajectory object.
        """
        self.position_trajectory = position_trajectory

    def calculate_rdf(self,
                    atom_type_1: Union[str, List[str]],
                    atom_type_2: Optional[Union[str, List[str]]] = None,
                    r_range: Tuple[float, float] = (0.0, 10.0),
                    num_bins: int = 100,
                    frame_indices: Optional[Union[int, List[int], np.ndarray]] = None,
                    average: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, List[np.ndarray]]]:
        positions_1, lattice_vectors = self.position_trajectory.get_relevant_positions(atom_indices=atom_type_1, frame_indices=frame_indices)

        if atom_type_2 is None:
            positions_2 = positions_1
            self_reference = True
        else:
            positions_2, _ = self.position_trajectory.get_relevant_positions(atom_indices=atom_type_2, frame_indices=frame_indices)
            self_reference = False

        num_atoms_1 = positions_1.shape[0]
        num_atoms_2 = positions_2.shape[0]
        num_frames = positions_1.shape[1]

        # Create the histogram bins
        bin_edges = np.linspace(r_range[0], r_range[1], num_bins + 1)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        shell_volumes = 4 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3) / 3

        # Calculate the RDF
        if average:
            rdf = np.zeros(num_bins)
            for frame in range(num_frames):
                if lattice_vectors.ndim == 3:
                    box = lattice_vectors[:, :, frame]
                else:
                    box = lattice_vectors

                inv_box = np.linalg.inv(box)

                # Calculate the minimum image distances
                delta = positions_1[:, frame, np.newaxis] - positions_2[:, frame]
                delta_frac = np.dot(delta, inv_box)
                delta_frac -= np.round(delta_frac)
                delta = np.dot(delta_frac, box)
                distances = np.linalg.norm(delta, axis=-1)

                if self_reference:
                    np.fill_diagonal(distances, np.inf)  # Exclude self-distance

                hist, _ = np.histogram(distances, bins=bin_edges)
                rdf += hist

            rdf /= num_frames * shell_volumes
            if self_reference:
                rdf /= num_atoms_1 - 1
            else:
                rdf /= num_atoms_1 * num_atoms_2

            return bin_centers, rdf
        else:
            rdf_frames = []
            for frame in range(num_frames):
                if lattice_vectors.ndim == 3:
                    box = lattice_vectors[:, :, frame]
                else:
                    box = lattice_vectors

                inv_box = np.linalg.inv(box)

                # Calculate the minimum image distances
                delta = positions_1[:, frame, np.newaxis] - positions_2[:, frame]
                delta_frac = np.dot(delta, inv_box)
                delta_frac -= np.round(delta_frac)
                delta = np.dot(delta_frac, box)
                distances = np.linalg.norm(delta, axis=-1)

                if self_reference:
                    np.fill_diagonal(distances, np.inf)  # Exclude self-distance

                hist, _ = np.histogram(distances, bins=bin_edges)
                rdf_frame = hist / (num_atoms_1 * num_atoms_2 * shell_volumes)
                if self_reference:
                    rdf_frame /= num_atoms_1 - 1
                rdf_frames.append(rdf_frame)

            return bin_centers, rdf_frames

    def plot_rdf(self, rdf_data: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, List[np.ndarray]]], labels: Optional[Union[str, List[str]]] = None, **kwargs) -> plt.Figure:
        """
        Plot the radial distribution function (RDF) data.

        Args:
            rdf_data (Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, List[np.ndarray]]]): The RDF data to plot, consisting of bin centers and RDF values.
            labels (Optional[Union[str, List[str]]]): The label or labels for the RDF curve(s).
            **kwargs: Additional keyword arguments for customizing the plot.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        bin_centers, rdf = rdf_data

        if isinstance(rdf, np.ndarray):
            # Plot a single RDF curve
            ax.plot(bin_centers, rdf, label=labels)
        else:
            # Plot multiple RDF curves
            if labels is None:
                labels = [f"Frame {i+1}" for i in range(len(rdf))]
            for i, rdf_frame in enumerate(rdf):
                ax.plot(bin_centers, rdf_frame, label=labels[i])

        ax.set_title(kwargs.get('title', 'Radial Distribution Function'))
        ax.set_xlabel(kwargs.get('xlabel', 'Distance (Å)'))
        ax.set_ylabel(kwargs.get('ylabel', 'g(r)'))

        if labels:
            ax.legend(loc=kwargs.get('legend_loc', 'best'))

        if kwargs.get('grid', True):
            ax.grid()

        return fig