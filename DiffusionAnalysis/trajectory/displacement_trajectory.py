import numpy as np
from typing import List, Tuple, Iterable, Optional
from ase import Atoms
from ..loaders import StructureLoader
from tqdm import tqdm
import itertools

class DisplacementTrajectory:
    '''
    Class to store and calculate the displacement trajectory of atoms in a MD simulation.

    Attributes:
    atoms_trajectory: StructureLoader
        A StructureLoader object that provides an iterable of Atoms objects representing the
        trajectory of atoms in the MD simulation.
    displacement_trajectory: np.ndarray
        Array of shape (n_atoms, n_frames, 3) that contains the displacement
        trajectory of the atoms in the MD simulation.
    unique_lattice_vectors: np.ndarray or None
        Array of shape (n_frames, 3, 3) that contains the unique lattice
        vectors of the MD simulation if track_lattice_vectors is True.
        Array of shape (3, 3)  if lattice_vectors are provided.
    atomic_number: np.ndarray
        Array of shape (n_atoms,) that contains the atomic numbers of the atoms in the MD simulation.
    host_atom_indices: np.ndarray
        Array of shape (n_host_atoms,) that contains the indices of the host atoms in the MD simulation.
    framework_atom_indices: np.ndar ray
        Array of shape (n_framework_atoms,) that contains the indices of the framework atoms in the MD simulation.
    use_cartesian: bool
        If True, the positions of the atoms will be stored in Cartesian coordinates.
        If False, the positions will be stored in fractional coordinates.
    track_lattice_vectors: bool
        If True, the lattice vectors will be tracked and stored for each frame.
        If False, only the unique lattice vectors will be stored.
    '''

    def __init__(self, 
                 atoms_trajectory_loader: StructureLoader,
                 lattice_vectors: Optional[np.ndarray] = None,
                 use_cartesian: bool = True,
                 track_lattice_vectors: bool = False,
                 max_memory : float = 1024 ) -> None:
        
        self.atoms_trajectory_loader: StructureLoader = atoms_trajectory_loader
        self.displacement_trajectory: np.ndarray = None
        self.track_lattice_vectors = track_lattice_vectors
        self.use_cartesian = use_cartesian
        self.atomic_numbers : np.ndarray = None
        self.max_memory = max_memory
        
        self.host_atom_indices = None
        self.framework_atom_indeces = None
        
        self._num_atoms = self.atoms_trajectory_loader.get_number_of_atoms()
        self._num_frames = self.atoms_trajectory_loader.get_total_steps()     
        self._check_memory_usage()
        
        if lattice_vectors is None:
            assert self.atoms_trajectory_loader.has_lattice_vectors, 'Trajectory loaded is in a format where we cant recover lattice vectors. Please provide lattice vectors.'
        if use_cartesian:
            assert self.atoms_trajectory_loader.has_lattice_vectors or lattice_vectors is not None, 'If use_cartesian is True, we need to be able to get lattice vectors.'
        if lattice_vectors is not None:
            assert not self.track_lattice_vectors, 'If lattice_vectors is provided, we cannot track lattice vectors.'
        
        self.unique_lattice_vectors = lattice_vectors
            
    def _check_memory_usage(self) -> None:
        '''
        Check if the memory usage of the displacement trajectory and lattice vectors exceeds the specified limit.
        '''
        num_atoms = self._num_atoms
        num_frames = self._num_frames
        safety_factor: float = 1.00 # Safety factor to account for additional memory usage
        
        displacement_memory = num_atoms * num_frames * 3 * 8  # Assuming 64-bit float for displacements
        
        if self.track_lattice_vectors:
            lattice_memory = num_frames * 9 * 8  # Assuming 64-bit float for lattice vectors
        else:
            lattice_memory = 9 * 8  # Assuming 64-bit float for unique lattice vectors
        
        total_memory = ((displacement_memory + lattice_memory) / 1024**2) * safety_factor  # Convert to MB
        
        if self.max_memory is not None and total_memory > self.max_memory:
            raise MemoryError(f'Estimated memory usage ({total_memory:.2f} MB) exceeds the specified limit ({self.max_memory:.2f} MB).')
        else:
            print(f'Estimated memory usage: {total_memory:.2f} MB')

    def _actual_memory_usage(self) -> float:
        '''
        Calculate the actual memory usage of the displacement trajectory and lattice vectors.
        '''
        lattice_vector_memory = 0
        if self.track_lattice_vectors:
            lattice_vector_memory = self.unique_lattice_vectors.nbytes
        return(self.displacement_trajectory.nbytes + lattice_vector_memory) / 1024**2

    def generate_displacement_trajectory(self, structures_to_load: slice = slice(None), 
                                         host_atoms = Optional[List[ int | str]],
                                         framework_atoms = Optional[List[ int | str]], 
                                         show_progress: bool = False) -> None:
        '''
        Generate the displacement trajectory of the atoms in the MD simulation.

        Parameters:
        structures_to_load: slice, optional
            A slice object specifying the range of structures to load from the atoms_trajectory.
            Default is slice(None), which loads all structures.
        host_atoms: List[int | str], optional
            A list of atom indices or atom symbols representing the host atoms.
            If not provided, all atoms are assumed to be host atoms.
        framework_atoms: List[int | str], optional
            A list of atom indices or atom symbols representing the framework atoms.
            If not provided, no framework atoms will be specified.
        show_progress: bool, optional
            If True, a progress bar will be displayed during the generation of the displacement trajectory.
            Default is False.
        '''

        self.atoms_trajectory_loader.reset() 
 
        atoms_trajectory: Iterable[Atoms] = itertools.islice(self.atoms_trajectory_loader, structures_to_load.start, structures_to_load.stop, structures_to_load.step)
        self._num_frames = self._calculate_total_structures(structures_to_load)

        step = 0
        previous_positions = None
        
        if show_progress:
            atoms_trajectory = tqdm(atoms_trajectory, total=self._num_frames, desc='Generating displacement trajectory')
            
        for atoms in atoms_trajectory:
            self._update_lattice_vectors(atoms, step)

            if step == 0:
                # Initialize the displacement trajectory and store the first positions
                self._num_atoms = len(atoms)
                self.atomic_numbers: np.ndarray = atoms.numbers
                self._populate_host_and_framework_atom_indices(atoms, host_atoms, framework_atoms)
                self.displacement_trajectory = np.zeros((self._num_atoms, self._num_frames, 3))
                previous_positions = atoms.get_positions()
            else:
                current_positions = atoms.get_positions()
                self._add_frame(current_positions,previous_positions, step)
                previous_positions = current_positions
            step += 1

        print(f'Completed with memory usage: {self._actual_memory_usage():.2f} MB')

    def _populate_host_and_framework_atom_indices(self, atoms: Atoms, host_atoms: Optional[List[int|str]], framework_atoms: Optional[List[int|str]]) -> None:
        if host_atoms is None:
            self.host_atom_indices = np.arange(len(atoms))
            self.framework_atom_indices = np.array([])
            return
        
        if isinstance(host_atoms[0], str):
            host_atom_indices = []
            framework_atom_indices = []
            
            for atom in atoms:
                if atom.symbol in host_atoms:
                    host_atom_indices.append(atom.index)
                elif framework_atoms is not None and isinstance(framework_atoms, list) and atom.symbol in framework_atoms:
                    framework_atom_indices.append(atom.index)
            
            self.host_atom_indices = np.array(host_atom_indices)
            self.framework_atom_indices = np.array(framework_atom_indices)
        
        else:
            self.host_atom_indices = np.array(host_atoms)
            if framework_atoms is not None and isinstance(framework_atoms, list):
                self.framework_atom_indices = np.array(framework_atoms)
            else:
                self.framework_atom_indices = np.array([])

    def _add_frame(self, 
                current_frame: np.ndarray, 
                previous_frame: np.ndarray, 
                step: int) -> None:
        '''
        Calculate and store the frame-to-frame displacements in the displacement_trajectory.

        The input frames are assumed to be in Cartesian coordinates. The displacements are
        calculated in Cartesian coordinates and wrapped based on the periodic boundary conditions
        determined by the lattice vectors. If track_lattice_vectors is True, the average lattice
        vectors between the current and previous frames are used for the PBC calculations.

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
            current_lattice_vectors = self.unique_lattice_vectors[step]
            previous_lattice_vectors = self.unique_lattice_vectors[step - 1]
            avg_lattice_vectors = (current_lattice_vectors + previous_lattice_vectors) / 2
        else:
            avg_lattice_vectors = self.unique_lattice_vectors

        # Calculate displacements in Cartesian coordinates
        displacements_cartesian = current_frame - previous_frame

        # Calculate displacements in Cartesian coordinates
        displacements_cartesian = current_frame - previous_frame

        # Wrap displacements based on PBC
        displacements_fractional = np.linalg.solve(avg_lattice_vectors.T, displacements_cartesian.T).T
        displacements_fractional = displacements_fractional - np.round(displacements_fractional)

        if self.use_cartesian:
            displacements_cartesian = np.dot(displacements_fractional, avg_lattice_vectors)
            self.displacement_trajectory[:, step, :] = displacements_cartesian
        else:
            self.displacement_trajectory[:, step, :] = displacements_fractional

    def _update_lattice_vectors(self, atoms: Atoms, step: int) -> None:
        '''
        Update the lattice vectors and lattice indices for the current frame.
        '''
        if self.track_lattice_vectors:
            if step == 0:
                self.unique_lattice_vectors = np.zeros((self._num_frames, 3, 3))
            self.unique_lattice_vectors[step, :, :] = atoms.get_cell()
        else:
            if step == 0 :
                self.unique_lattice_vectors = atoms.get_cell()

    def _calculate_total_structures(self, structures_to_load: slice) -> int:
        '''
        Calculate the total number of structures to be loaded based on the slice.
        '''
        start, stop, step = structures_to_load.indices(self.atoms_trajectory_loader.get_total_steps())
        total_structures = len(range(start, stop, step))
        return total_structures