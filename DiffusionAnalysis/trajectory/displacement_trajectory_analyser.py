import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Set, Iterable
from ase import Atoms
from tqdm import tqdm
from DiffusionAnalysis.loaders.base_structure_loader import StructureLoader
from DiffusionAnalysis.utils import TimeData, AtomSubsetMap
from DiffusionAnalysis.utils.trajectory_utils import calculate_center_of_mass
from DiffusionAnalysis.trajectory.displacement_trajectory import DisplacementTrajectory, TrajectoryDirectionEnum, BaseDisplacementTrajectory, MSDTrajectory, SquaredDisplacementTrajectory, tMSDTrajectory

class DisplacementTrajectoryAnalyser:
    def __init__(self, 
                 atoms_trajectory_loader: StructureLoader,
                 max_memory: float = 1024):
        
        self.atoms_trajectory_loader = atoms_trajectory_loader
        self.max_memory = max_memory
        
        self._num_atoms = None
        self._num_frames = None
        self.track_lattice_vectors = False
        self.atom_subset_map: Optional[AtomSubsetMap] = None
        self.unique_lattice_vectors: Optional[np.ndarray] = None
    
    def calculate_displacements(self,
                                 atoms_to_include: Optional[List[Union[int, str]]] = None,
                                 lattice_vectors: Optional[np.ndarray] = None,
                                 use_cartesian: bool = True,
                                 track_lattice_vectors: bool = False,
                                 show_progress: bool = True) -> None:
        '''
        Generate the displacement trajectory of the atoms in the MD simulation.

        Parameters:
        atom_indices_or_strings: List[int | str], optional
            A list of atom indices or atom symbols representing the atoms to include in the displacement trajectory.
            If not provided, all atoms are included.
        lattice_vectors: np.ndarray, optional
            The lattice vectors to use for the displacement trajectory.
            If not provided, the lattice vectors will be obtained from the atoms trajectory loader.
        use_cartesian: bool, optional
            If True, the positions of the atoms will be stored in Cartesian coordinates.
            If False, the positions will be stored in fractional coordinates.
            Default is True.
        track_lattice_vectors: bool, optional
            If True, the lattice vectors will be tracked and stored for each frame.
            If False, only the unique lattice vectors will be stored.
            Default is False.
        show_progress: bool, optional
            If True, a progress bar will be displayed during the generation of the displacement trajectory.
            Default is False.
        '''
        if lattice_vectors is None:
            assert self.atoms_trajectory_loader.has_lattice_vectors, 'Trajectory loaded is in a format where we cant recover lattice vectors. Please provide lattice vectors.'
        if use_cartesian:
            assert self.atoms_trajectory_loader.has_lattice_vectors or lattice_vectors is not None, 'If use_cartesian is True, we need to be able to get lattice vectors.'
        if lattice_vectors is not None:
            assert not track_lattice_vectors, 'If lattice_vectors is provided, we cannot track lattice vectors.'
        
        # Reset the atoms trajectory loader to the beginning
        self.atoms_trajectory_loader.reset()
        self._estimate_memory_usage()

        self.track_lattice_vectors = track_lattice_vectors
        self.use_cartesian = use_cartesian
        self.unique_lattice_vectors = lattice_vectors
        
        self._num_frames = self.atoms_trajectory_loader.get_total_steps()  

        atoms_trajectory: Iterable[Atoms] = self.atoms_trajectory_loader
        
        step = 0
        # empty placeholder ndarray for previous positions
        previous_positions = None
        
        if show_progress:
            atoms_trajectory = tqdm(atoms_trajectory, total=self._num_frames, desc='Generating displacement trajectory')
            
        for atoms in atoms_trajectory:
            self._update_lattice_vectors(atoms, step)
            if step == 0:
                self.atom_subset_map = AtomSubsetMap(atoms, atoms_to_include)
                self._num_atoms = len(self.atom_subset_map)
                relevant_atom_indices = self.atom_subset_map.get_original_indices()
                self.displacement_trajectory = np.zeros((len(relevant_atom_indices), self._num_frames, 3))
                
                previous_positions = atoms.get_positions()[relevant_atom_indices]
            else:
                if previous_positions is None:
                    raise ValueError('Loading failed. Previous positions are not available')
                
                current_positions = atoms.get_positions()[relevant_atom_indices]
                self._add_frame(current_positions, previous_positions, step)
                previous_positions = current_positions
            step += 1

        print(f'Completed with memory usage: {self._actual_memory_usage():.2f} MB')

    def _add_frame(self, 
                current_frame: np.ndarray, 
                previous_frame: np.ndarray, 
                step: int) -> None:
        '''
        Calculate and store the frame-to-frame displacements in the displacement_trajectory.

        The input frames are assumed to be in Cartesian coordinates. The displacements are
        calculated in Cartesian coordinates and wrapped based on the periodic boundary conditions
        determined by the average of the lattice vectors between the current and previous frames.

        If use_cartesian is True, the displacements are stored in Cartesian coordinates.
        If use_cartesian is False, the displacements are converted to fractional coordinates
        before storing.

        Parameters:
        current_frame: np.ndarray
            The positions of the atoms in the current frame (Cartesian coordinates).
        previous_frame: np.ndarray
            The positions of the atoms in the previous frame (Cartesian coordinates).
        step: int
            The current step or frame index.
        '''
        if self.track_lattice_vectors:
            if self.unique_lattice_vectors is None:
                raise ValueError('Unique lattice vectors are not available so we cant track them!.')
            avg_lattice_vectors = (self.unique_lattice_vectors[step] + self.unique_lattice_vectors[step - 1]) / 2
        else:
            avg_lattice_vectors = self.unique_lattice_vectors

        # Calculate displacements in Cartesian coordinates
        displacements_cartesian = current_frame - previous_frame

        # Convert Cartesian displacements to fractional coordinates
        displacements_fractional = np.linalg.solve(avg_lattice_vectors.T, displacements_cartesian.T).T

        # Wrap fractional displacements to the range [0, 1)
        displacements_fractional = displacements_fractional - np.rint(displacements_fractional)

        if self.use_cartesian:
            # Convert fractional displacements back to Cartesian coordinates
            if avg_lattice_vectors is None:
                raise ValueError('Lattice vectors are not available so we cant convert to Cartesian coordinates.')
            self.displacement_trajectory[:, step, :] = np.dot(displacements_fractional, avg_lattice_vectors)
        else:
            self.displacement_trajectory[:, step, :] = displacements_fractional

    def _update_lattice_vectors(self, atoms: Atoms, step: int) -> None:
        '''
        Update the lattice vectors and lattice indices for the current frame.
        '''
        if self._num_frames is None:
            raise ValueError('Number of frames not calculated yet.')
        if self.track_lattice_vectors:
            if step == 0:
                self.unique_lattice_vectors = np.zeros((self._num_frames, 3, 3))
            if self.unique_lattice_vectors is None:
                raise ValueError('Error with Unique lattice vectors.')
            self.unique_lattice_vectors[step, :, :] = atoms.get_cell()
        elif self.unique_lattice_vectors is None:
            if step == 0 :
                # jank but python deals with it 
                # get cell is a nd like aray of (3,3)
                self.unique_lattice_vectors = np.array([atoms.get_cell()])

    def _calculate_total_structures(self, structures_to_load: slice) -> int:
        '''
        Calculate the total number of structures to be loaded based on the slice.
        '''
        if structures_to_load == slice(None):
            return self.atoms_trajectory_loader.get_total_steps()
        start, stop, step = structures_to_load.indices(self.atoms_trajectory_loader.get_total_steps())
        total_structures = len(range(start, stop, step))
        return total_structures
    
    def _estimate_memory_usage(self) -> None:
        '''
        Check if the memory usage of the displacement trajectory and lattice vectors exceeds the specified limit.
        '''
        if self.max_memory is None:
            return
        
        safety_factor: float = 1.1 # Safety factor to account for additional memory usage

        displacement_memory = self.atoms_trajectory_loader.get_number_of_atoms() * self.atoms_trajectory_loader.get_total_steps() * 3 * 8  # Assuming 64-bit float for displacements
        
        if self.track_lattice_vectors:
            if self._num_frames is None:
                self._num_frames = self.atoms_trajectory_loader.get_total_steps()
            lattice_memory = self._num_frames * 9 * 8  # Assuming 64-bit float for lattice vectors
        else:
            lattice_memory = 9 * 8  # Assuming 64-bit float for unique lattice vectors
        
        total_memory = ((displacement_memory + lattice_memory) / 1024**2) * safety_factor  # Convert to MB
        
        if total_memory > self.max_memory:
            raise MemoryError(f'Estimated memory usage ({total_memory:.2f} MB) exceeds the specified limit ({self.max_memory:.2f} MB).')
        else:
            print(f'Estimated memory usage: {total_memory:.2f} MB')

    def _actual_memory_usage(self) -> float:
        '''
        Calculate the actual memory usage of the displacement trajectory and lattice vectors.
        '''
        lattice_vector_memory = 0
        if self.track_lattice_vectors and self.unique_lattice_vectors:
            lattice_vector_memory = self.unique_lattice_vectors.nbytes
        if self.displacement_trajectory is None:
            raise ValueError('Displacement trajectory not calculated yet.')
        return(self.displacement_trajectory.nbytes + lattice_vector_memory) / 1024**2
    
    @property
    def number_of_atoms(self) -> int:
        if self.atom_subset_map is None:
            raise ValueError('Displacement trajectory not calculated yet.')
        self._num_atoms = len(self.atom_subset_map)
        return self._num_atoms

    @property
    def number_of_frames(self) -> int:
        if self.atom_subset_map is None:
            raise ValueError('Displacement trajectory not calculated yet.')
        return self.atoms_trajectory_loader.get_total_steps()
        
    def _get_relevant_displacements(self,
                                tracer_specs: Optional[Union[List[Union[str,int]],slice]] = None,
                                framework_specs: Optional[Union[List[Union[str,int]],slice]] = None,
                                correct_drift: bool = False,
                                return_framework_com_displacement: bool = False,
                                return_indices: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Get the displacements for the selected atoms and optionally correct for framework drift.

        Args:
            tracer_specs (List[Union[int, str]], optional): The indices or symbols of the tracer atoms to track.
                                                            Defaults to None. None means include all atoms.
            framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms.
                                                            Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            return_framework_com_displacement (bool, optional): Whether to return the center of mass displacement
                                                                of the framework atoms. Defaults to False.
            return_indices (bool, optional): Whether to return the atom and framework indices along with the
                                            displacements. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: The displacements for the selected atoms, and optionally
                                                        the atom indices, framework indices, and framework center
                                                        of mass displacement.

        Raises:
            ValueError: If the displacement trajectory has not been generated.
            ValueError: If tracer specs and framework specs overlap.
            ValueError: If framework specs are not provided when drift correction is enabled.
            NotImplementedError: If the displacement calculation is requested in fractional coordinates.
        """
        if not self.use_cartesian:
            raise NotImplementedError('Displacement calculation is not YET supported in fractional coordinates.')

        if return_framework_com_displacement is True and framework_specs is None:
            raise ValueError("Framework specs must be provided to return the framework center of mass displacement.")

        if self.displacement_trajectory is None:
            raise ValueError("Displacement trajectory has not been generated.")
        
        if tracer_specs is None:
            assert framework_specs is None and correct_drift is False, 'If tracer_specs is None, which means include all atoms, framework_specs should also be None.'
        
        if self.atom_subset_map is None:
            raise ValueError("atom_subset_map is not initialized.")

        
        tracer_spec_subset = self.atom_subset_map[tracer_specs]
        tracer_indices = self.atom_subset_map.get_corresponding_indices_from_other_map(tracer_spec_subset)

        if len(tracer_indices) == 0:
            raise ValueError("No atoms found for the provided tracer specs.")

        if framework_specs is not None:
            framework_spec_subset = self.atom_subset_map[tracer_specs]
            framework_indices = self.atom_subset_map.get_corresponding_indices_from_other_map(tracer_spec_subset)
            #asssert no overlap
            assert len(set(framework_indices) & set(tracer_indices)) == 0, 'Tracer and framework indices overlap.'
        
        tracer_displacements = self.displacement_trajectory[tracer_indices]

        framework_center_of_mass = None

        if correct_drift or return_framework_com_displacement:
            framework_displacements = self.displacement_trajectory[framework_indices]
            framework_atomic_numbers = framework_spec_subset.get_subset_atomic_numbers()
            framework_center_of_mass = calculate_center_of_mass(framework_displacements, framework_atomic_numbers)

            if correct_drift:
                tracer_displacements -= framework_center_of_mass[np.newaxis]
                
        return_values = (tracer_displacements,)
        if return_indices:
            return_values += (tracer_indices, framework_indices)
        if return_framework_com_displacement:
            return_values += (framework_center_of_mass,)

        return return_values[0] if len(return_values) == 1 else return_values
    
    def get_displacement_trajectory(self,
                                tracer_specs: Optional[Union[List[Union[str, int]], slice]] = None,
                                framework_specs: Optional[Union[List[Union[str, int]], slice]] = None,
                                correct_drift: bool = False,
                                is_com: bool = False,
                                trajectory_type: TrajectoryDirectionEnum = TrajectoryDirectionEnum.X_Y_Z,
                                lattice_vector: Optional[np.ndarray] = None
                                    ) -> DisplacementTrajectory:
        '''
        Get the displacement trajectory for the selected atoms.

        Args:
            tracer_specs (List[Union[int, str]], optional): The indices or symbols of the tracer atoms to track.
                                                            Defaults to None. None means include all atoms.
            framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms.
                                                            Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            is_com (bool, optional): Whether to calculate the center of mass displacement. Defaults to False.
            trajectory_type (TrajectoryDirectionEnum, optional): The type of trajectory to calculate.
                                                                Defaults to TrajectoryDirectionEnum.X_Y_Z.
            lattice_vector (np.ndarray, optional): The lattice vector to project the displacement onto.
                                                Defaults to None.

        Returns:
            DisplacementTrajectory: The displacement trajectory for the selected atoms.

        Raises:
            ValueError: If lattice_vector is provided but trajectory_type is not TrajectoryDirectionEnum.LATTICE_VECTOR.
            ValueError: If trajectory_type is TrajectoryDirectionEnum.LATTICE_VECTOR but lattice_vector is None.
        '''
        if lattice_vector is not None and trajectory_type != TrajectoryDirectionEnum.LATTICE_VECTOR:
            raise ValueError("If lattice_vector is provided, trajectory_type must be TrajectoryDirectionEnum.LATTICE_VECTOR.")
        
        if trajectory_type == TrajectoryDirectionEnum.LATTICE_VECTOR and lattice_vector is None:
            raise ValueError("If trajectory_type is TrajectoryDirectionEnum.LATTICE_VECTOR, lattice_vector must be provided.")

        tracer_displacements = self._get_relevant_displacements(
            tracer_specs, framework_specs, correct_drift, return_framework_com_displacement=False
        )

        tracer_displacements = np.cumsum(tracer_displacements, axis=1)

        if self.atom_subset_map is None:
            raise ValueError("atom_subset_map is not initialized.")

        tracer_map = self.atom_subset_map[tracer_specs]

        if is_com:
            tracer_atomic_numbers = tracer_map.get_subset_atomic_numbers()
            tracer_displacements = calculate_center_of_mass(tracer_displacements, tracer_atomic_numbers)

        if trajectory_type == TrajectoryDirectionEnum.THREE_DIMENSIONAL:
            tracer_displacements = np.linalg.norm(tracer_displacements, axis=2, keepdims=True)
        elif trajectory_type == TrajectoryDirectionEnum.LATTICE_VECTOR:
            lattice_vector = lattice_vector / np.linalg.norm(lattice_vector)
            tracer_displacements = np.dot(tracer_displacements, lattice_vector)

        if isinstance(tracer_specs, slice):
            raise NotImplementedError('Slice is not supported for tracer_specs.')

        if isinstance(framework_specs, slice):
            raise NotImplementedError('Slice is not supported for framework_specs.')

        assert isinstance(tracer_displacements, np.ndarray), 'Tracer displacements must be an ndarray.'

        return DisplacementTrajectory(tracer_displacements,
                                    tracer_specs,
                                    framework_specs,
                                    tracer_map,
                                    self.atoms_trajectory_loader.get_trajectory_time_info,
                                    trajectory_type,
                                    is_com=is_com,
                                    is_drift_corrected=correct_drift)
        

    def get_msd_trajectory(self,
                       tracer_specs: Optional[Union[List[Union[str, int]], slice]] = None,
                       framework_specs: Optional[Union[List[Union[str, int]], slice]] = None,
                       correct_drift: bool = False,
                       is_com: bool = False,
                       trajectory_type: TrajectoryDirectionEnum = TrajectoryDirectionEnum.X_Y_Z,
                       lattice_vector: Optional[np.ndarray] = None
                       ) -> MSDTrajectory:
        '''
        Get the mean squared displacement (MSD) trajectory for the selected atoms.

        Args:
            tracer_specs (List[Union[int, str]], optional): The indices or symbols of the tracer atoms to track.
                                                            Defaults to None. None means include all atoms.
            framework_specs (List[Union[int, str]], optional): The indices or symbols of the framework atoms.
                                                            Defaults to None.
            correct_drift (bool, optional): Whether to correct for framework drift. Defaults to False.
            is_com (bool, optional): Whether to calculate the center of mass MSD. Defaults to False.
            trajectory_type (TrajectoryDirectionEnum, optional): The type of trajectory to calculate.
                                                                Defaults to TrajectoryDirectionEnum.X_Y_Z.
            lattice_vector (np.ndarray, optional): The lattice vector to project the displacement onto.
                                                Defaults to None.

        Returns:
            MSDTrajectory: The MSD trajectory for the selected atoms.

        Raises:
            ValueError: If lattice_vector is provided but trajectory_type is not TrajectoryDirectionEnum.LATTICE_VECTOR.
            ValueError: If trajectory_type is TrajectoryDirectionEnum.LATTICE_VECTOR but lattice_vector is None.
        '''
        if lattice_vector is not None and trajectory_type != TrajectoryDirectionEnum.LATTICE_VECTOR:
            raise ValueError("If lattice_vector is provided, trajectory_type must be TrajectoryDirectionEnum.LATTICE_VECTOR.")
        
        if trajectory_type == TrajectoryDirectionEnum.LATTICE_VECTOR and lattice_vector is None:
            raise ValueError("If trajectory_type is TrajectoryDirectionEnum.LATTICE_VECTOR, lattice_vector must be provided.")

        tracer_displacements = self._get_relevant_displacements(
            tracer_specs, framework_specs, correct_drift, return_framework_com_displacement=False
        )  # tracer_displacements shape: (n_atoms, n_frames, 3)

        if self.atom_subset_map is None:
            raise ValueError("atom_subset_map is not initialized.")

        tracer_map = self.atom_subset_map[tracer_specs]

        if is_com:
            tracer_atomic_numbers = tracer_map.get_subset_atomic_numbers()
            tracer_displacements = calculate_center_of_mass(tracer_displacements, tracer_atomic_numbers)  # tracer_displacements shape: (n_frames, 3)

        if trajectory_type == TrajectoryDirectionEnum.THREE_DIMENSIONAL:
            if is_com:
                # x^2 + y^2 + z^2
                msd_data = np.sum(tracer_displacements**2, axis=1)  # msd_data shape: (n_frames,)
            else:
                # 
                msd_data = np.mean(np.sum(tracer_displacements**2, axis=2), axis=0)  # msd_data shape: (n_frames,)
        
        elif trajectory_type == TrajectoryDirectionEnum.LATTICE_VECTOR:
            lattice_vector = lattice_vector / np.linalg.norm(lattice_vector)
            if is_com:
                projected_displacements = np.dot(tracer_displacements, lattice_vector)  # projected_displacements shape: (n_frames,)
                msd_data = projected_displacements**2  # msd_data shape: (n_frames,)
            else:
                projected_displacements = np.dot(tracer_displacements, lattice_vector.T)  # projected_displacements shape: (n_atoms, n_frames)
                msd_data = np.mean(projected_displacements**2, axis=0)  # msd_data shape: (n_frames,)
        else:  # TrajectoryDirectionEnum.X_Y_Z
            if is_com:
                msd_data = tracer_displacements**2  # msd_data shape: (n_frames, 3)
            else:
                msd_data = np.mean(tracer_displacements**2, axis=0)  # msd_data shape: (n_atoms, n_frames, 3)
        
        
        if isinstance(tracer_specs, slice):
            raise NotImplementedError('Slice is not supported for tracer_specs.')

        if isinstance(framework_specs, slice):
            raise NotImplementedError('Slice is not supported for framework_specs.')

        assert isinstance(msd_data, np.ndarray), 'MSD data must be an ndarray.'

        return MSDTrajectory(msd_data,
                            tracer_specs,
                            framework_specs,
                            tracer_map,
                            self.atoms_trajectory_loader.get_trajectory_time_info,
                            trajectory_type,
                            is_com=is_com,
                            is_drift_corrected=correct_drift)