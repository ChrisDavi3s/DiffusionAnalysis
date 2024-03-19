import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from ..trajectory import DisplacementTrajectory
from tqdm import tqdm

class VanHoveAnalysis:
    '''
    Class for calculating the self and distinct parts of the van Hove correlation function.

    Parameters:
        displacement_trajectory (DisplacementTrajectory): The displacement trajectory object.
    '''

    def __init__(self, displacement_trajectory: DisplacementTrajectory):
        self.displacement_trajectory = displacement_trajectory

    def calculate_van_hove(self, tau: int, r_range: Tuple[float, float], n_bins: int,
                           atom_indices: Optional[np.ndarray] = None, framework_indices: Optional[np.ndarray] = None,
                           mode: str = 'self', correct_drift: bool = False, progress_bar: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if mode == 'self':
            return self._calculate_van_hove_self(tau, r_range, n_bins, atom_indices, framework_indices, correct_drift, progress_bar)
        elif mode == 'distinct':
            return self._calculate_van_hove_distinct(tau, r_range, n_bins, atom_indices, framework_indices, correct_drift, progress_bar)
        else:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are 'self' and 'distinct'.")

    def _calculate_van_hove_self(self, tau: int, r_range: Tuple[float, float], n_bins: int,
                                atom_indices: Optional[np.ndarray] = None, framework_indices: Optional[np.ndarray] = None,
                                correct_drift: bool = False, progress_bar: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
        '''
        Calculate the self part of the van Hove correlation function.

        Parameters:
            tau (int): The time step for the van Hove correlation function.
            r_range (Tuple[float, float]): The range of displacement magnitudes to consider.
            n_bins (int): The number of bins for the histogram.
            atom_indices (Optional[np.ndarray], optional): The indices of the atoms to consider. If None, all atoms are considered.
            framework_indices (Optional[np.ndarray], optional): The indices of the framework atoms. If provided, only atoms not in the framework are considered.
            correct_drift (bool, optional): Whether to correct for drift in the trajectory. Default is False.
            progress_bar (bool, optional): Whether to display a progress bar during calculation. Default is True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The bin centers and histogram values for the self part of the van Hove correlation function.
        '''

        displacements = self.displacement_trajectory.get_relevant_displacements(atom_indices, framework_indices, correct_drift)

        # Use NumPy's broadcasting to calculate displacement magnitudes efficiently
        displacement_magnitudes = np.linalg.norm(displacements[:, :-tau, np.newaxis, :] - displacements[:, tau:, np.newaxis, :], axis=3)
        displacement_magnitudes = displacement_magnitudes.reshape(-1)

        # Filter out NaN and infinite values
        finite_mask = np.isfinite(displacement_magnitudes)
        finite_displacement_magnitudes = displacement_magnitudes[finite_mask]

        if len(finite_displacement_magnitudes) == 0:
            print("Warning: No valid displacement magnitudes found for the self-part of the van Hove correlation function. Please check that the timestep is not larger than the number of timesteps in the trajectory.")
            return np.array([]), np.array([])

        # Use tqdm to display a progress bar
        if progress_bar:
            finite_displacement_magnitudes = np.array(list(tqdm(finite_displacement_magnitudes, desc="Calculating Van Hove")))

        hist, bin_edges = np.histogram(finite_displacement_magnitudes, bins=n_bins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        return bin_centers, hist

    def _calculate_van_hove_distinct(self, tau: int, r_range: Tuple[float, float], n_bins: int,
                                    atom_indices: Optional[np.ndarray] = None, framework_indices: Optional[np.ndarray] = None,
                                    correct_drift: bool = False, progress_bar: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
        '''
        Calculate the distinct part of the van Hove correlation function.

        Parameters:
            tau (int): The time step for the van Hove correlation function.
            r_range (Tuple[float, float]): The range of displacement magnitudes to consider.
            n_bins (int): The number of bins for the histogram.
            atom_indices (Optional[np.ndarray], optional): The indices of the atoms to consider. If None, all atoms are considered.
            framework_indices (Optional[np.ndarray], optional): The indices of the framework atoms. If provided, only atoms not in the framework are considered.
            correct_drift (bool, optional): Whether to correct for drift in the trajectory. Default is False.
            progress_bar (bool, optional): Whether to display a progress bar during calculation. Default is True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The bin centers and histogram values for the distinct part of the van Hove correlation function.
        '''

        displacements = self.displacement_trajectory.get_relevant_displacements(atom_indices, framework_indices, correct_drift)
        n_atoms = displacements.shape[0]

        # Use NumPy's broadcasting to calculate displacement magnitudes efficiently
        displacement_magnitudes = np.linalg.norm(displacements[:, :-tau, np.newaxis, :] - displacements[:, tau:, np.newaxis, :], axis=3)
        displacement_magnitudes = displacement_magnitudes.reshape(-1)

        # Filter out NaN and infinite values
        finite_mask = np.isfinite(displacement_magnitudes)
        finite_displacement_magnitudes = displacement_magnitudes[finite_mask]

        if len(finite_displacement_magnitudes) == 0:
            return np.array([]), np.array([])

         # Use tqdm to display a progress bar
        if progress_bar:
            finite_displacement_magnitudes = np.array(list(tqdm(finite_displacement_magnitudes, desc="Calculating Van Hove")))

        hist, bin_edges = np.histogram(displacement_magnitudes, bins=n_bins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        normalization_factor = n_atoms * (n_atoms - 1) * (len(finite_displacement_magnitudes) / len(displacement_magnitudes))
        normalized_hist = hist / normalization_factor

        return bin_centers, normalized_hist
    
    def plot_van_hove(self, data: List[Tuple[np.ndarray, np.ndarray, int]], mode: str, **kwargs) -> plt.Figure:
        '''
        Plot the van Hove correlation function data.

        Parameters:
            data (List[Tuple[np.ndarray, np.ndarray, int]]): A list of tuples containing the bin centers, histogram values, and time step for each van Hove correlation function.
            mode (str): The mode for the van Hove correlation function ('self' or 'distinct').
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            plt.Figure: The figure containing the van Hove correlation function plot.
        '''

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
    
        for bin_centers, hist, tau in data:
            time_value = tau * self.displacement_trajectory.timestep
            time_unit = self.displacement_trajectory.time_unit.value
            label = kwargs.get('label', f'{mode.capitalize()} part, t = {time_value:.2f} {time_unit}')
            ax.plot(bin_centers, hist, label=label)

        ax.set_xlabel(kwargs.get('xlabel', 'r (Å)'))
        ax.set_ylabel(kwargs.get('ylabel', 'G(r, t)'))
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
        ax.set_title(kwargs.get('title', f'Van Hove {mode.capitalize()} Correlation Function'))
        ax.grid(kwargs.get('grid', True))
        fig.tight_layout()

        return fig