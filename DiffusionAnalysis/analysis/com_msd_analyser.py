import numpy as np
import matplotlib.pyplot as plt
from ..trajectory import DisplacementTrajectory
from typing import Optional, Union, Tuple, List

class COMMSDAnalyser:
    def __init__(self, displacement_trajectory: DisplacementTrajectory):
        """
        Initialize the COMMSDAnalysis class.

        Args:
            displacement_trajectory (DisplacementTrajectory): The displacement trajectory object.
        """
        assert displacement_trajectory.use_cartesian is True, "Displacement trajectory must be in Cartesian coordinates."
        self.displacement_trajectory = displacement_trajectory

    def calculate_com_msd(self,
                        tracer_specs: List[Union[int, str]],
                        framework_specs: Optional[List[Union[int, str]]] = None,
                        correct_drift: bool = False,
                        return_3d_msd: bool = False,
                        lattice_vector: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate the mean squared displacement (MSD) of the center of mass for the selected species per timestep.

        Args:
            tracer_specs (List[Union[int, str]]): The indices or symbols of the species to track.
            framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms. Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            return_3d_msd (bool, optional): Whether to return the 3D MSD array. Defaults to False.
            lattice_vector (np.ndarray, optional): The lattice vector along which to calculate the MSD. Defaults to None.

        Returns:
            If return_3d_msd is True:
                np.ndarray: The 3D MSD array per timestep.
            If return_3d_msd is False and lattice_vector is None:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The MSD arrays per timestep for each direction (x, y, z).
            If return_3d_msd is False and lattice_vector is not None:
                np.ndarray: The MSD array per timestep along the lattice vector.
        """
        species_displacements, framework_com_displacement = self.displacement_trajectory.get_relevant_displacements(
            tracer_specs, framework_specs, correct_drift, return_framework_com_displacement=True
        )
        species_com_displacement = np.mean(species_displacements, axis=0)

        cumulative_com_displacement = np.cumsum(species_com_displacement, axis=0)

        if lattice_vector is not None:
            lattice_vector = lattice_vector / np.linalg.norm(lattice_vector)
            projected_displacements = np.dot(cumulative_com_displacement, lattice_vector)
            msd_per_timestep = np.square(projected_displacements)
            return msd_per_timestep

        msd_x_per_timestep = np.square(cumulative_com_displacement[:, 0])
        msd_y_per_timestep = np.square(cumulative_com_displacement[:, 1])
        msd_z_per_timestep = np.square(cumulative_com_displacement[:, 2])

        if return_3d_msd:
            return msd_x_per_timestep + msd_y_per_timestep + msd_z_per_timestep
        else:
            return msd_x_per_timestep, msd_y_per_timestep, msd_z_per_timestep
        
    def calculate_com_displacement_per_atom(self, 
                                    tracer_specs: Optional[List[Union[int, str]]] = None,
                                    framework_specs: Optional[List[Union[int, str]]] = None,
                                    correct_drift: bool = False,
                                    time_lag: int = 1,
                                    use_3d: bool = False,
                                    lattice_vector: Optional[np.ndarray] = None,
                                    num_bins: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        """
        Calculate the COM displacement per atom for a given time lag and optionally return the displacement distribution.

            Args:
                tracer_specs (List[Union[int, str]], optional): The indices or symbols of the tracer atoms to track.
                                                                Defaults to None.
                framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms.
                                                                Defaults to None.
                correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
                time_lag (int, optional): The time lag (tau) for calculating the displacement. Defaults to 1.
                use_3d (bool, optional): Whether to calculate the 3D displacement. Defaults to False.
                lattice_vector (np.ndarray, optional): The lattice vector along which to calculate the displacement. Defaults to None which means the displacement is calculated in the x direction if 3D is False.
                num_bins (int, optional): The number of bins for the histogram. If provided, the displacement distribution will be returned. Defaults to None.

            Returns:
                Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The displacement per atom for the given time lag. If num_bins is provided, returns a tuple containing the bin centers and the probability distribution.
        """
        species_displacements, framework_com_displacement = self.displacement_trajectory.get_relevant_displacements(
            tracer_specs, framework_specs, correct_drift, return_framework_com_displacement=True
        )
        num_lags = species_displacements.shape[1] // time_lag
        remaining_lags = species_displacements.shape[1] % time_lag

        if remaining_lags > 0:
            species_displacements = species_displacements[:, :-remaining_lags]

        species_com_displacement = np.mean(species_displacements, axis=0)
        displacements = species_com_displacement.reshape(-1, num_lags, time_lag, 3)
        total_displacements = np.sum(displacements, axis=2)


        if use_3d:
            displacement_per_atom = np.sqrt(np.sum(np.square(total_displacements), axis=2))  # sqrt(x^2 + y^2 + z^2)
        elif lattice_vector is not None:
            #normalise lattice vector to unit vector
            lattice_vector = lattice_vector / np.linalg.norm(lattice_vector)
            displacement_per_atom = np.dot(total_displacements, lattice_vector)
        else:
            displacement_per_atom = total_displacements[:, :, 0]  # x

        if num_bins is not None:
            hist, bin_edges = np.histogram(displacement_per_atom, bins=num_bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return bin_centers, hist
        else:
            return displacement_per_atom
    
        
    def plot_com_msd(self, com_msd_data: Union[np.ndarray, Tuple[np.ndarray, ...]], labels: Optional[Union[str, List[str]]] = None, skip_points: int = 0, log_scale: bool = False, **kwargs) -> plt.Figure:
        """
        Plot the mean squared displacement (MSD) data.

        Args:
            com_msd_data (Union[np.ndarray, Tuple[np.ndarray, ...]]): The MSD data to plot. Can be a single array or a tuple of arrays.
            labels (Optional[Union[str, List[str]]], optional): The labels for the MSD curves. Defaults to None.
            skip_points (int, optional): The number of initial points to skip when plotting the MSD data. Defaults to 0.
            log_scale (bool, optional): Whether to use logarithmic scaling on both axes. Defaults to False.
            **kwargs: Additional keyword arguments for customizing the plot.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
        timestep = self.displacement_trajectory.atoms_trajectory_loader.timestep
        time_unit = self.displacement_trajectory.atoms_trajectory_loader.time_unit.value

        if isinstance(com_msd_data, tuple):
            for i, msd in enumerate(com_msd_data):
                if log_scale:
                    ax.loglog(np.arange(skip_points, len(msd)) * timestep, msd[skip_points:], label=labels[i] if labels else None)
                else:
                    ax.plot(np.arange(skip_points, len(msd)) * timestep, msd[skip_points:], label=labels[i] if labels else None)
        else:
            if log_scale:
                ax.loglog(np.arange(skip_points, len(com_msd_data)) * timestep, com_msd_data[skip_points:], label=labels)
            else:
                ax.plot(np.arange(skip_points, len(com_msd_data)) * timestep, com_msd_data[skip_points:], label=labels)

        ax.set_title(kwargs.get('title', 'COM Mean Squared Displacement'))
        ax.set_xlabel(kwargs.get('xlabel', f'Time ({time_unit})'))
        ax.set_ylabel(kwargs.get('ylabel', 'MSD (Å^2)'))

        if labels:
            ax.legend(loc=kwargs.get('legend_loc', 'best'))

        if kwargs.get('grid', True):
            ax.grid()

        return fig
    
    def plot_com_displacement_distribution(self, tracer_specs: Optional[List[Union[int, str]]] = None,
                                    framework_specs: Optional[List[Union[int, str]]] = None,
                                    correct_drift: bool = False,
                                    time_lags: List[int] = None,
                                    num_bins: int = 100,
                                    use_3d: bool = False,
                                    lattice_vector: Optional[np.ndarray] = None,
                                    **kwargs) -> plt.Figure:
        """
        Plot the probability distribution of displacement per atom for given time lags.

        Args:
            tracer_specs (List[Union[int, str]], optional): The indices or symbols of the tracer atoms to track.
                                                            Defaults to None.
            framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms.
                                                            Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            time_lags (List[int], optional): The list of time lags (tau) for calculating the displacement. Defaults to None.
            num_bins (int, optional): The number of bins for the histogram. Defaults to 100.
            use_3d (bool, optional): Whether to calculate the probability distribution for the 3D displacement. Defaults to False.
            lattice_vector (np.ndarray, optional): The lattice vector along which to calculate the probability distribution. Defaults to None.
            **kwargs: Additional keyword arguments for customizing the plot.

        Returns:
            plt.Figure: The matplotlib figure object.
        """

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        timestep = self.displacement_trajectory.atoms_trajectory_loader.timestep
        time_unit = self.displacement_trajectory.atoms_trajectory_loader.time_unit.value

        for time_lag in time_lags:
            bin_centers, hist = self.calculate_com_displacement_per_atom(tracer_specs, framework_specs, correct_drift,
                                                                    time_lag, use_3d, lattice_vector, num_bins)
            label = f"Time lag: {time_lag*timestep} ({time_unit})"
            ax.plot(bin_centers, hist, label=label)

        xlabel = kwargs.get('xlabel', 'Displacement (Å)')
        title = kwargs.get('title', 'Probability Distribution of COM Displacement')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(kwargs.get('ylabel', 'Probability'))
        ax.set_title(title)
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
        ax.grid(kwargs.get('grid', True))

        return fig
    