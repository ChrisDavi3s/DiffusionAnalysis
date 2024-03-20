import numpy as np
import matplotlib.pyplot as plt
from ..trajectory import DisplacementTrajectory
from typing import Optional, Union, Tuple, List

class MSDAnalysis:
    def __init__(self, displacement_trajectory: DisplacementTrajectory):
        """
        Initialize the SimpleMSDAnalysis class.

        Args:
            displacement_trajectory (DisplacementTrajectory): The displacement trajectory object.
        """
        assert displacement_trajectory.use_cartesian is True, "Displacement trajectory must be in Cartesian coordinates."
        # TODO - add ability to convert to Cartesian coordinates (which isnt hard)
        self.displacement_trajectory = displacement_trajectory

    def calculate_msd(self, atom_indices: Optional[np.ndarray] = None, framework_indices: Optional[np.ndarray] = None, correct_drift: bool = False, return_3d_msd: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate the mean squared displacement (MSD) for the selected atoms.

        Args:
            atom_indices (np.ndarray, optional): The indices of the atoms to include. Defaults to None.
            framework_indices (np.ndarray, optional): The indices of the framework atoms. Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            return_3d_msd (bool, optional): Whether to return the 3D MSD array. Defaults to False.

        Returns:
            If return_3d_msd is True:
                np.ndarray: The 3D MSD array.
            If return_3d_msd is False:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The MSD arrays for each direction (x, y, z).
        """
        displacements = self.displacement_trajectory.get_relevant_displacements(atom_indices, framework_indices, correct_drift)

        cumulative_displacements = np.cumsum(displacements, axis=1)

        msd_x = np.mean(np.square(cumulative_displacements[:, :, 0]), axis=0)
        msd_y = np.mean(np.square(cumulative_displacements[:, :, 1]), axis=0)
        msd_z = np.mean(np.square(cumulative_displacements[:, :, 2]), axis=0)

        if return_3d_msd:
            return msd_x + msd_y + msd_z
        else:
            return msd_x, msd_y, msd_z

    def calculate_msd_along_lattice_vector(self, lattice_vector: np.ndarray, atom_indices: Optional[np.ndarray] = None, framework_indices: Optional[np.ndarray] = None, correct_drift: bool = False) -> np.ndarray:
        """
        Calculate the mean squared displacement (MSD) along a specific lattice vector for the selected atoms.

        Args:
            lattice_vector (np.ndarray): The lattice vector along which to calculate the MSD.
            atom_indices (np.ndarray, optional): The indices of the atoms to include. Defaults to None.
            framework_indices (np.ndarray, optional): The indices of the framework atoms. Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.

        Returns:
            np.ndarray: The MSD array along the lattice vector.
        """
        displacements = self.displacement_trajectory.get_relevant_displacements(atom_indices, framework_indices, correct_drift)

        # Normalize the lattice vector
        lattice_vector = lattice_vector / np.linalg.norm(lattice_vector)

        # Project the displacements onto the lattice vector
        projected_displacements = np.dot(displacements, lattice_vector)

        # Calculate the cumulative displacements along the lattice vector
        cumulative_displacements = np.cumsum(projected_displacements, axis=1)

        # Calculate the mean squared displacement along the lattice vector
        msd = np.mean(np.square(cumulative_displacements), axis=0)

        return msd
            
    def plot_msd(self, msd_data: Union[np.ndarray, Tuple[np.ndarray, ...]], labels: Optional[Union[str, List[str]]] = None, skip_points: int = 0, log_scale: bool = False, **kwargs) -> plt.Figure:
        """
        Plot the mean squared displacement (MSD) data.

        Args:
            msd_data (Union[np.ndarray, Tuple[np.ndarray, ...]]): The MSD data to plot. Can be a single array or a tuple of arrays.
            labels (Optional[Union[str, List[str]]], optional): The labels for the MSD curves. Defaults to None.
            skip_points (int, optional): The number of initial points to skip when plotting the MSD data. Defaults to 0.
            log_scale (bool, optional): Whether to use logarithmic scaling on both axes. Defaults to False.
            **kwargs: Additional keyword arguments for customizing the plot.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
        timestep = self.displacement_trajectory.timestep
        time_unit = self.displacement_trajectory.time_unit.value

        if isinstance(msd_data, tuple):
            for i, msd in enumerate(msd_data):
                if log_scale:
                    ax.loglog(np.arange(skip_points, len(msd)) * timestep, msd[skip_points:], label=labels[i] if labels else None)
                else:
                    ax.plot(np.arange(skip_points, len(msd)) * timestep, msd[skip_points:], label=labels[i] if labels else None)
        else:
            if log_scale:
                ax.loglog(np.arange(skip_points, len(msd_data)) * timestep, msd_data[skip_points:], label=labels)
            else:
                ax.plot(np.arange(skip_points, len(msd_data)) * timestep, msd_data[skip_points:], label=labels)

        ax.set_title(kwargs.get('title', 'Mean Squared Displacement'))
        ax.set_xlabel(kwargs.get('xlabel', f'Time ({time_unit})'))
        ax.set_ylabel(kwargs.get('ylabel', 'MSD (Å^2)'))

        if labels:
            ax.legend(loc=kwargs.get('legend_loc', 'best'))

        if kwargs.get('grid', True):
            ax.grid()

        return fig