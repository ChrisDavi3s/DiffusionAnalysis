import numpy as np
import matplotlib.pyplot as plt
from ..trajectory import DisplacementTrajectory
from typing import Optional, Union, Tuple, List

class SimpleMSDAnalysis:
    def __init__(self, displacement_trajectory: DisplacementTrajectory):
        """
        Initialize the SimpleMSDAnalysis class.

        Args:
            displacement_trajectory (DisplacementTrajectory): The displacement trajectory object.
        """
        assert displacement_trajectory.use_cartesian is True, "Displacement trajectory must be in Cartesian coordinates."
        # TODO - add ability to convert to Cartesian coordinates (which isnt hard)
        self.displacement_trajectory = displacement_trajectory

    def calculate_msd(self, atom_indices: Optional[np.ndarray] = None, framework_indices: Optional[np.ndarray] = None, correct_drift: bool = False, return_3d_msd: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        squared_displacements = np.square(displacements)

        if not return_3d_msd:
            msd_x = np.mean(squared_displacements[:, :, 0], axis=0)
            msd_y = np.mean(squared_displacements[:, :, 1], axis=0)
            msd_z = np.mean(squared_displacements[:, :, 2], axis=0)
            return msd_x, msd_y, msd_z
        else:
            msd_total = np.mean(np.sum(squared_displacements, axis=2), axis=0)
            return msd_total

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
        projected_displacements = np.dot(displacements, lattice_vector)
        squared_displacements = np.square(projected_displacements)
        msd = np.mean(squared_displacements, axis=0)
        return msd
    
    def plot_msd(self, msd_data: Union[np.ndarray, Tuple[np.ndarray, ...]], labels: Optional[Union[str, List[str]]] = None, skip_points: int = 0, **kwargs) -> plt.Figure:
            """
            Plot the mean squared displacement (MSD) data.

            Args:
                msd_data (Union[np.ndarray, Tuple[np.ndarray, ...]]): The MSD data to plot. Can be a single array or a tuple of arrays.
                labels (Optional[Union[str, List[str]]], optional): The labels for the MSD curves. Defaults to None.
                skip_points (int, optional): The number of initial points to skip when plotting the MSD data. Defaults to 0.
                **kwargs: Additional keyword arguments for customizing the plot.

            Returns:
                plt.Figure: The matplotlib figure object.
            """
            fig = plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            timestep = self.displacement_trajectory.timestep
            time_unit = self.displacement_trajectory.time_unit.value

            if isinstance(msd_data, tuple):
                for i, msd in enumerate(msd_data):
                    plt.plot(np.arange(skip_points, len(msd)) * timestep, msd[skip_points:], label=labels[i] if labels else None)
            else:
                plt.plot(np.arange(skip_points, len(msd_data)) * timestep, msd_data[skip_points:], label=labels if labels else None)

            plt.title(kwargs.get('title', 'Mean Squared Displacement'))
            plt.xlabel(kwargs.get('xlabel', f'Time ({time_unit})'))
            plt.ylabel(kwargs.get('ylabel', 'MSD (Å^2)'))
            plt.legend(loc=kwargs.get('legend_loc', 'best'))

            if kwargs.get('grid', True):
                plt.grid()

            plt.tight_layout()
            return fig