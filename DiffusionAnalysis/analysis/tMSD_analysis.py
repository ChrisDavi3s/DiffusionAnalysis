import numpy as np
import matplotlib.pyplot as plt
from ..trajectory import DisplacementTrajectory
from typing import Optional, Tuple
from tqdm import tqdm

class tMSDAnalysis:
    '''
    Class for calculating and plotting the time-averaged mean squared displacement (tMSD).

    Parameters:
    displacement_trajectory (DisplacementTrajectory): The displacement trajectory object.
    '''

    def __init__(self, displacement_trajectory: DisplacementTrajectory):
        self.displacement_trajectory = displacement_trajectory

    #TODO memory consumption is high, need to optimise?? I think it's fine but a check would be good ... maybe against the max memory available in displacement_trajectory?
    def calculate_tMSD(self, min_tau: int, max_tau: int, num_points: int, atom_indices: Optional[np.ndarray] = None,
                    framework_indices: Optional[np.ndarray] = None, correct_drift: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate the time-averaged mean squared displacement (tMSD).

        Parameters:
        min_tau (int): The minimum time lag value to consider.
        max_tau (int): The maximum time lag value to consider.
        num_points (int): The number of time lag values to consider.
        atom_indices (np.ndarray, optional): The indices of the atoms to track. Defaults to None.
        framework_indices (np.ndarray, optional): The indices of the framework atoms. Defaults to None.
        correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The time lag values and the corresponding tMSD values.
        '''

        displacements = self.displacement_trajectory.get_relevant_displacements(atom_indices, framework_indices, correct_drift)

        tau_values = np.logspace(np.log10(min_tau), np.log10(max_tau), num_points, dtype=int)

        tMSD_values = []

        cumulative_displacements = np.cumsum(displacements, axis=1)

        for tau in tqdm(tau_values, desc="Calculating tMSD"):
            if tau >= displacements.shape[1]:
                break

            num_steps = displacements.shape[1]
            num_bins = num_steps - tau

            # Use broadcasting to calculate the displacement differences
            start_indices = np.arange(num_bins)[:, np.newaxis]
            end_indices = start_indices + tau
            displacement_diffs = cumulative_displacements[:, end_indices] - cumulative_displacements[:, start_indices]

            # Calculate the squared displacement for each atom and each bin
            squared_displacements = np.sum(displacement_diffs**2, axis=2)

            # Take the mean over atoms and bins
            tMSD = np.mean(squared_displacements)
            tMSD_values.append(tMSD)

        time_lag_values = tau_values[:len(tMSD_values)] * self.displacement_trajectory.timestep

        return time_lag_values, np.array(tMSD_values)

    def plot_tMSD(self, time_lag_values: np.ndarray, tMSD_values: np.ndarray, label: str, **kwargs) -> plt.Figure:
        '''
        Plot the time-averaged mean squared displacement (tMSD) data.

        Parameters:
        time_lag_values (np.ndarray): The time lag values (in ps or ns).
        tMSD_values (np.ndarray): The corresponding tMSD values.
        label (str): The label for the tMSD dataset.
        **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
        plt.Figure: The figure containing the tMSD plot.
        '''
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        ax.loglog(time_lag_values, tMSD_values, label=label)
        time_unit = self.displacement_trajectory.time_unit.value
        ax.set_xlabel(kwargs.get('xlabel', f'Time lag ({time_unit})'))
        ax.set_ylabel(kwargs.get('ylabel', r'tMSD ($\AA^2$)'))
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
        ax.set_title(kwargs.get('title', 'Time-averaged Mean Squared Displacement'))
        ax.grid(kwargs.get('grid', True))
        fig.tight_layout()

        return fig

    def plot_tMSD_exponent(self, time_lag_values: np.ndarray, tMSD_values: np.ndarray, label: str, window_size: int = 5, **kwargs) -> plt.Figure:
        '''
        Plot the exponent of the time-averaged mean squared displacement (tMSD) data.

        Parameters:
        time_lag_values (np.ndarray): The time lag values (in ps or ns).
        tMSD_values (np.ndarray): The corresponding tMSD values.
        label (str): The label for the tMSD dataset.
        window_size (int): The size of the window for smoothing the exponent values (default: 5).
        **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
        plt.Figure: The figure containing the exponent plot.
        '''
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        log_time_lag_values = np.log10(time_lag_values)
        log_tMSD_values = np.log10(tMSD_values)
        exponents = np.diff(log_tMSD_values) / np.diff(log_time_lag_values)

        # Handle RuntimeWarning for invalid values
        mask = np.isfinite(exponents)
        exponents = exponents[mask]
        time_lag_values = time_lag_values[:-1][mask]

        # Apply moving average to smooth the exponent values
        kernel = np.ones(window_size) / window_size
        smoothed_exponents = np.convolve(exponents, kernel, mode='valid')

        # Handle edge cases
        pad_size = window_size // 2
        exponents = np.pad(exponents, (pad_size, pad_size), mode='edge')
        smoothed_exponents = np.concatenate((exponents[:pad_size], smoothed_exponents, exponents[-pad_size:]))

        # Ensure time_lag_values and smoothed_exponents have the same length
        time_lag_values = time_lag_values[:len(smoothed_exponents)]

        ax.semilogx(time_lag_values, smoothed_exponents, label=label)
        time_unit = self.displacement_trajectory.time_unit.value
        ax.set_xlabel(kwargs.get('xlabel', f'Time lag ({time_unit})'))
        ax.set_ylabel(kwargs.get('ylabel', r'Exponent of $\langle r^2(t)\rangle$'))
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
        ax.set_title(kwargs.get('title', 'Exponent of Time-averaged Mean Squared Displacement'))
        ax.grid(kwargs.get('grid', True))
        # horizontal line at 1
        ax.axhline(y=1, color='r', linestyle='--')
        ax.axhline(y=2, color='purple', linestyle='--')
        fig.tight_layout()


        return fig  