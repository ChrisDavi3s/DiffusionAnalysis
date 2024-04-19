import numpy as np
import matplotlib.pyplot as plt
from ..trajectory import DisplacementTrajectory
from typing import Optional, Union, Tuple, List


class TracerMSDAnalyser:
    def __init__(self, displacement_trajectory: DisplacementTrajectory):
        """
        Initialize the TracerMSDAnalyser class.

        Args:
            displacement_trajectory (DisplacementTrajectory): The displacement trajectory object.
        """
        assert displacement_trajectory.use_cartesian is True, "Displacement trajectory must be in Cartesian coordinates."
        self.displacement_trajectory = displacement_trajectory

    def calculate_msd(self, 
                      tracer_specs: Optional[List[Union[int, str]]] = None, 
                      framework_specs: Optional[List[Union[int, str]]] = None,
                      correct_drift: bool = False, 
                      return_3d_msd: bool = False, 
                      lattice_vector: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate the mean squared displacement (MSD) for the selected atoms.

        Args:
            tracer_specs (List[Union[int, str]], optional): The indices or symbols of the tracer atoms to track.
                                                            Defaults to None.
            framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms.
                                                            Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            return_3d_msd (bool, optional): Whether to return the 3D MSD array. Defaults to False.
            lattice_vector (np.ndarray, optional): The lattice vector along which to calculate the MSD. Defaults to None.
            step_size (int, optional): The step size for calculating the MSD. Defaults to None.

        Returns:
            If return_3d_msd is True:
                np.ndarray: The 3D MSD array.
            If return_3d_msd is False and lattice_vector is None:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The MSD arrays for each direction (x, y, z).
            If return_3d_msd is False and lattice_vector is not None:
                np.ndarray: The MSD array along the lattice vector.
        """
        assert not (return_3d_msd and lattice_vector is not None), "Cannot return 3D MSD and lattice vector MSD simultaneously."
        displacements = self.displacement_trajectory.get_relevant_displacements(tracer_specs, framework_specs, correct_drift)
        print(displacements.shape)

        if lattice_vector is not None:
            lattice_vector = lattice_vector / np.linalg.norm(lattice_vector)
            projected_displacements = np.dot(displacements, lattice_vector)
            cumulative_displacements = np.cumsum(projected_displacements, axis=1)
            msd = np.mean(np.square(cumulative_displacements), axis=0)
            return msd

        cumulative_displacements = np.cumsum(displacements, axis=1)

        msd_x = np.mean(np.square(cumulative_displacements[:, :, 0]), axis=0)
        msd_y = np.mean(np.square(cumulative_displacements[:, :, 1]), axis=0)
        msd_z = np.mean(np.square(cumulative_displacements[:, :, 2]), axis=0)

        if return_3d_msd:
            return msd_x + msd_y + msd_z
        else:
            return msd_x, msd_y, msd_z
    
    def calculate_msd_per_atom(self, tracer_specs: Optional[List[Union[int, str]]] = None,
                            framework_specs: Optional[List[Union[int, str]]] = None,
                            correct_drift: bool = False,
                            step_size: Optional[int] = None,
                            use_3d: bool = False,
                            lattice_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the mean squared displacement (MSD) per atom per timestep size.

        Args:
            tracer_specs (List[Union[int, str]], optional): The indices or symbols of the tracer atoms to track.
                                                            Defaults to None.
            framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms.
                                                            Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            step_size (int, optional): The step size for calculating the MSD. Defaults to None.
            use_3d (bool, optional): Whether to calculate the 3D MSD. Defaults to False.
            lattice_vector (np.ndarray, optional): The lattice vector along which to calculate the MSD. Defaults to None.

        Returns:
            np.ndarray: The MSD per atom per timestep size.
        """
        displacements = self.displacement_trajectory.get_relevant_displacements(tracer_specs, framework_specs, correct_drift)

        if step_size is not None:
            displacements = displacements[:, ::step_size]

        cumulative_displacements = np.sum(displacements, axis=1)  # (x, y, z)

        if use_3d:
            squared_displacements = np.sum(np.square(cumulative_displacements), axis=1)  # x^2 + y^2 + z^2
        elif lattice_vector is not None:
            lattice_vector = lattice_vector / np.linalg.norm(lattice_vector)
            projected_displacements = np.dot(cumulative_displacements, lattice_vector)
            squared_displacements = np.square(projected_displacements)
        else:
            squared_displacements = np.square(cumulative_displacements[:, 0])  # x^2

        msd_per_atom = squared_displacements

        return msd_per_atom
    
    def calculate_displacement_per_atom(self, tracer_specs: Optional[List[Union[int, str]]] = None,
                                    framework_specs: Optional[List[Union[int, str]]] = None,
                                    correct_drift: bool = False,
                                    time_lag: int = 1,
                                    use_3d: bool = False,
                                    lattice_vector: Optional[np.ndarray] = None,
                                    num_bins: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
            Calculate the displacement per atom for a given time lag and optionally return the displacement distribution.

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
        assert use_3d or lattice_vector is not None, "Must specify either 3D displacement or lattice vector."
        displacements = self.displacement_trajectory.get_relevant_displacements(tracer_specs, framework_specs, correct_drift)

        num_lags = displacements.shape[1] // time_lag
        displacements = displacements[:, :num_lags*time_lag].reshape(-1, num_lags, time_lag, 3)
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
        timestep = self.displacement_trajectory.atoms_trajectory_loader.timestep
        time_unit = self.displacement_trajectory.atoms_trajectory_loader.time_unit.value

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
    
    def plot_displacement_distribution(self, tracer_specs: Optional[List[Union[int, str]]] = None,
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
            bin_centers, hist = self.calculate_displacement_per_atom(tracer_specs, framework_specs, correct_drift,
                                                                    time_lag, use_3d, lattice_vector, num_bins)
            label = f"Time lag: {time_lag*timestep} ({time_unit})"
            ax.plot(bin_centers, hist, label=label)

        xlabel = kwargs.get('xlabel', 'Displacement (Å)')
        title = kwargs.get('title', 'Probability Distribution of Displacement')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(kwargs.get('ylabel', 'Probability'))
        ax.set_title(title)
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
        ax.grid(kwargs.get('grid', True))

        return fig
