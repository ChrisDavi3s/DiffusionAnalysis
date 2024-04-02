import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from ..trajectory import DisplacementTrajectory
from tqdm import tqdm

class VanHoveAnalyser:
    """
    Class for calculating the self and distinct parts of the van Hove correlation function.

    Parameters:
        displacement_trajectory (DisplacementTrajectory): The displacement trajectory object.
    """

    def __init__(self, displacement_trajectory: DisplacementTrajectory):
        self.displacement_trajectory = displacement_trajectory

    def calculate_van_hove(self, tau_values: List[int], r_range: Tuple[float, float], n_bins: int,
                           type_a_specs: Optional[List[Union[int, str]]] = None,
                           type_b_specs: Optional[List[Union[int, str]]] = None,
                           mode: str = 'self', progress_bar: bool = True, memory_limit_mb: float = 1000) -> List[Tuple[np.ndarray, np.ndarray, int, Union[None, Tuple[float, int]]]]:
        """
        Calculate the van Hove correlation function for the specified tau values and atom types.

        Parameters:
            tau_values (List[int]): The list of tau values to calculate the van Hove correlation function for.
            r_range (Tuple[float, float]): The range of distances to consider for the histogram.
            n_bins (int): The number of bins to use for the histogram.
            type_a_specs (Optional[List[Union[int, str]]]): The specifications for type A atoms.
            type_b_specs (Optional[List[Union[int, str]]]): The specifications for type B atoms.
            mode (str): The mode of the van Hove correlation function to calculate ('self' or 'distinct').
            progress_bar (bool): Whether to display a progress bar during the calculation.
            memory_limit_mb (float): The memory limit in megabytes for the distinct-part calculation.

        Returns:
            List[Tuple[np.ndarray, np.ndarray, int, Union[None, Tuple[float, int]]]]: A list of tuples containing:
                - bin_centers (np.ndarray): The centers of the histogram bins.
                - hist (np.ndarray): The histogram values.
                - tau (int): The tau value.
                - normalization_factor (Union[None, Tuple[float, int]]):
                    - For 'self' mode: None
                    - For 'distinct' mode: A tuple containing the volume of the system and the number of pairs used in the calculation.

        Raises:
            ValueError: If an invalid mode is specified. Supported modes are 'self' and 'distinct'.
        """
        results = []
        for tau in tqdm(tau_values, desc=f'Calculating Van Hove {mode}', disable=not progress_bar):
            if mode == 'self':
                bin_centers, hist = self._calculate_van_hove_self(tau, r_range, n_bins, type_a_specs)
                results.append((bin_centers, hist, tau, None))
            elif mode == 'distinct':
                bin_centers, hist, volume, max_pairs = self._calculate_van_hove_distinct(tau, r_range, n_bins, type_a_specs, type_b_specs, memory_limit_mb=memory_limit_mb)
                results.append((bin_centers, hist, tau, (volume, max_pairs)))
            else:
                raise ValueError(f"Invalid mode: {mode}. Supported modes are 'self' and 'distinct'.")
        return results

    def _calculate_van_hove_self(self, tau: int, r_range: Tuple[float, float], n_bins: int,
                                 type_a_specs: Optional[List[Union[int, str]]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the self part of the van Hove correlation function.

        Parameters:
            tau (int): The time lag for the van Hove correlation function.
            r_range (Tuple[float, float]): The range of distances to consider for the histogram.
            n_bins (int): The number of bins to use for the histogram.
            type_a_specs (Optional[List[Union[int, str]]]): The specifications for type A atoms.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The bin centers and histogram values for the self part
                of the van Hove correlation function.
        """
        displacements = self.displacement_trajectory.get_relevant_displacements(tracer_specs=type_a_specs)
        
        displacement_magnitudes = np.linalg.norm(displacements[:, :-tau] - displacements[:, tau:], axis=2)
        displacement_magnitudes = displacement_magnitudes.reshape(-1)

        finite_mask = np.isfinite(displacement_magnitudes)
        finite_displacement_magnitudes = displacement_magnitudes[finite_mask]

        if len(finite_displacement_magnitudes) == 0:
            print("Warning: No valid displacement magnitudes found for the self-part of the van Hove correlation function.")
            return np.array([]), np.array([])

        # density = True means that we normalize the histogram so that the integral is 1
        hist, bin_edges = np.histogram(finite_displacement_magnitudes, bins=n_bins, range=r_range, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

    def _calculate_van_hove_distinct(self, tau: int, r_range: Tuple[float, float], n_bins: int,
                                     type_a_specs: Optional[List[Union[int, str]]] = None,
                                     type_b_specs: Optional[List[Union[int, str]]] = None,
                                     memory_limit_mb: float = 1000) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Calculate the distinct part of the van Hove correlation function.

        Parameters:
            tau (int): The time lag for the van Hove correlation function.
            r_range (Tuple[float, float]): The range of distances to consider for the histogram.
            n_bins (int): The number of bins to use for the histogram.
            type_a_specs (Optional[List[Union[int, str]]]): The specifications for type A atoms.
            type_b_specs (Optional[List[Union[int, str]]]): The specifications for type B atoms.
            memory_limit_mb (float): The memory limit in megabytes for the calculation.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, int]: The bin centers, histogram values, volume of the system,
                and the number of pairs used in the calculation.
        """
        displacements_a = self.displacement_trajectory.get_relevant_displacements(tracer_specs=type_a_specs)
        displacements_b = self.displacement_trajectory.get_relevant_displacements(tracer_specs=type_b_specs)

        n_atoms_a = len(displacements_a)
        n_atoms_b = len(displacements_b)

        # Calculate the maximum number of pairs based on the available memory
        slope = 0.08943
        y_intercept = 800.0 
        max_pairs = int((memory_limit_mb - y_intercept) / slope)

        if type_a_specs == type_b_specs:
            max_pairs = min(max_pairs, n_atoms_a * (n_atoms_a - 1) // 2)
        else:
            max_pairs = min(max_pairs, n_atoms_a * n_atoms_b)

        pair_indices = np.random.choice(n_atoms_a * n_atoms_b, size=max_pairs, replace=False)
        atom_a_indices, atom_b_indices = np.unravel_index(pair_indices, (n_atoms_a, n_atoms_b))

        displacement_magnitudes = np.linalg.norm(displacements_a[atom_a_indices, :-tau] - displacements_b[atom_b_indices, tau:], axis=-1)

        if not np.all(np.isfinite(displacement_magnitudes)):
            raise ValueError("Non-finite values found in the distinct part of the van Hove correlation function.")

        if isinstance(self.displacement_trajectory.unique_lattice_vectors, list):
            lattice_vectors = self.displacement_trajectory.unique_lattice_vectors[0]
        else:
            lattice_vectors = self.displacement_trajectory.unique_lattice_vectors

        hist, bin_edges = np.histogram(displacement_magnitudes, bins=n_bins, range=r_range, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        volume = np.linalg.det(lattice_vectors)
        return bin_centers, hist, volume, max_pairs
    
    def plot_van_hove(self, data: List[Tuple[np.ndarray, np.ndarray, int, Union[None, Tuple[float, int]]]], mode: str, include_normalization: bool = True, **kwargs) -> plt.Figure:
        '''
        Plot the van Hove correlation function data.

        Parameters:
            data (List[Tuple[np.ndarray, np.ndarray, int, Union[None, Tuple[float, int]]]]): A list of tuples containing the bin centers,
                histogram values, time step, and normalization factor (if applicable) for each van Hove correlation function.
            mode (str): The mode for the van Hove correlation function ('self' or 'distinct').
            include_normalization (bool): Whether to include normalization in the plot (default: True).
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            plt.Figure: The figure containing the van Hove correlation function plot.
        '''
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        
        for bin_centers, hist, tau, normalization_factor in data:
            time_value = tau * self.displacement_trajectory.atoms_trajectory_loader.timestep
            time_unit = self.displacement_trajectory.atoms_trajectory_loader.time_unit.value
            label = kwargs.get('label', f'{mode.capitalize()} part, t = {time_value:.2f} {time_unit}')
            
            if include_normalization:
                if mode == 'self':
                    bin_width = bin_centers[1] - bin_centers[0]
                    hist = hist * 4 * np.pi * bin_centers**2 / bin_width
                elif mode == 'distinct':
                    if normalization_factor is None:
                        raise ValueError("Normalization factor is required for distinct part.")
                    volume, max_pairs = normalization_factor
                    print('This normalization is not correct. Its a rough guess but we need to know the density!.')
                    N = len(self.displacement_trajectory.atoms_trajectory_loader)
                    hist = hist * (N - 1) / (volume * max_pairs)
            
            ax.plot(bin_centers, hist, label=label)

        if include_normalization:
            if mode == 'self':
                y_label = kwargs.get('ylabel', '4πr^2G_s(r, t)')
            elif mode == 'distinct':
                y_label = kwargs.get('ylabel', 'G_d(r, t)/ρ')
        else:
            if mode == 'self':
                y_label = kwargs.get('ylabel', 'G_s(r, t) (unnormalized)')
            elif mode == 'distinct':
                y_label = kwargs.get('ylabel', 'G_d(r, t) (unnormalized)')
        
        ax.set_xlabel(kwargs.get('xlabel', 'r (Å)'))
        ax.set_ylabel(y_label)
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
        ax.set_title(kwargs.get('title', f'Van Hove {mode.capitalize()} Correlation Function'))
        ax.grid(kwargs.get('grid', True))
        fig.tight_layout()

        return fig