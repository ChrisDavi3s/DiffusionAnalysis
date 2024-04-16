import numpy as np
from typing import Optional, Union, List, Tuple
from ase import Atoms
from DiffusionAnalysis.loaders import StructureLoader
from DiffusionAnalysis.utils.time_unit import TimeUnit
import itertools
from tqdm import tqdm
import itertools
from ase.data import atomic_numbers

class PositionTrajectory:
    def __init__(self,
                 atoms_trajectory_loader: StructureLoader,
                 lattice_vectors: Optional[np.ndarray] = None,
                 use_fractional: bool = False,
                 track_lattice_vectors: bool = False,
                 max_memory: float = 1024,
                 timestep: float = 1,
                 time_unit: Union[str, TimeUnit] = 'ps') -> None:
        
        self.atoms_trajectory_loader = atoms_trajectory_loader
        self.position_trajectory: np.ndarray = None
        self.track_lattice_vectors = track_lattice_vectors
        self.use_fractional = use_fractional
        self.atomic_numbers: np.ndarray = None
        self.max_memory = max_memory
        self.timestep = timestep
        self.time_unit = TimeUnit(time_unit) if isinstance(time_unit, str) else time_unit
        
        self._num_atoms = self.atoms_trajectory_loader.get_number_of_atoms()
        self._num_frames = self.atoms_trajectory_loader.get_total_steps()
        
        if lattice_vectors is None:
            assert self.atoms_trajectory_loader.has_lattice_vectors, 'Trajectory loaded is in a format where we cant recover lattice vectors. Please provide lattice vectors.'
        if use_fractional:
            assert self.atoms_trajectory_loader.has_lattice_vectors or lattice_vectors is not None, 'If use_fractional is True, we need to be able to get lattice vectors.'
        if lattice_vectors is not None:
            assert not self.track_lattice_vectors, 'If lattice_vectors is provided, we cannot track lattice vectors.'
        
        self.unique_lattice_vectors = lattice_vectors
        
        self._check_memory_usage()
    
    def _check_memory_usage(self) -> None:
        if self.max_memory is None:
            return
        
        num_atoms = self._num_atoms
        num_frames = self._num_frames
        safety_factor: float = 1.1 # Safety factor to account for additional memory usage
        
        position_memory = num_atoms * num_frames * 3 * 8  # Assuming 64-bit float for positions
        
        if self.track_lattice_vectors:
            lattice_memory = num_frames * 9 * 8  # Assuming 64-bit float for lattice vectors
        else:
            lattice_memory = 9 * 8  # Assuming 64-bit float for unique lattice vectors
        
        total_memory = ((position_memory + lattice_memory) / 1024**2) * safety_factor  # Convert to MB
        
        if total_memory > self.max_memory:
            raise MemoryError(f'Estimated memory usage ({total_memory:.2f} MB) exceeds the specified limit ({self.max_memory:.2f} MB).')
        else:
            print(f'Estimated memory usage: {total_memory:.2f} MB')
    
    def _actual_memory_usage(self) -> float:
        lattice_vector_memory = 0
        if self.track_lattice_vectors:
            lattice_vector_memory = self.unique_lattice_vectors.nbytes
        return (self.position_trajectory.nbytes + lattice_vector_memory) / 1024**2
    
    def generate_position_trajectory(self,
                                    structures_to_load: Union[slice, int] = slice(None),
                                    show_progress: bool = False) -> None:
        self.atoms_trajectory_loader.reset()
        
        if structures_to_load == slice(None):
            structures_to_load = slice(0, self._num_frames, 1)
        
        if isinstance(structures_to_load, int):
            atoms_trajectory = [next(itertools.islice(self.atoms_trajectory_loader, structures_to_load, structures_to_load+1))]
            self._num_frames = 1
        else:
            atoms_trajectory = itertools.islice(self.atoms_trajectory_loader, structures_to_load.start, structures_to_load.stop, structures_to_load.step)
            self._num_frames = self._calculate_total_structures(structures_to_load)
        
        if show_progress:
            atoms_trajectory = tqdm(atoms_trajectory, total=self._num_frames, desc='Generating position trajectory')
        
        for step, atoms in enumerate(atoms_trajectory):
            self._update_lattice_vectors(atoms, step)
            
            if step == 0:
                self._num_atoms = len(atoms)
                self.atomic_numbers = atoms.numbers
                self.position_trajectory = np.zeros((self._num_atoms, self._num_frames, 3))
            
            positions = atoms.get_positions()
            
            if self.use_fractional:
                if self.track_lattice_vectors:
                    lattice_vectors = self.unique_lattice_vectors[step]
                else:
                    lattice_vectors = self.unique_lattice_vectors
                
                fractional_positions = np.linalg.solve(lattice_vectors.T, positions.T).T
                self.position_trajectory[:, step, :] = fractional_positions
            else:
                self.position_trajectory[:, step, :] = positions
        
        print(f'Completed with memory usage: {self._actual_memory_usage():.2f} MB')

        
    def _update_lattice_vectors(self, atoms: Atoms, step: int) -> None:
        if self.track_lattice_vectors:
            if step == 0:
                self.unique_lattice_vectors = np.zeros((self._num_frames, 3, 3))
            self.unique_lattice_vectors[step] = atoms.get_cell()
        else:
            if step == 0:
                self.unique_lattice_vectors = atoms.get_cell()
    
    def _calculate_total_structures(self, structures_to_load: slice) -> int:
        if structures_to_load == slice(None):
            return self.atoms_trajectory_loader.get_total_steps()
        start, stop, step = structures_to_load.indices(self.atoms_trajectory_loader.get_total_steps())
        total_structures = len(range(start, stop, step))
        return total_structures
    
    def get_relevant_positions(self,
                            atom_indices: Optional[Union[np.ndarray, List[str]]] = None,
                            fractional: Optional[bool] = None,
                            frame_indices: Optional[Union[int, List[int], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        assert self.position_trajectory is not None, "Position trajectory has not been generated."

        if atom_indices is None:
            positions = self.position_trajectory
        else:
            if isinstance(atom_indices, str):
                atom_indices = [atom_indices]

            atom_indices = [np.where(self.atomic_numbers == atomic_numbers[atom])[0] for atom in atom_indices]
            atom_indices = np.concatenate(atom_indices)
            positions = self.position_trajectory[atom_indices]


        if frame_indices is not None:
            if isinstance(frame_indices, int):
                frame_indices = [frame_indices]
            positions = positions[:, frame_indices, :]

        if fractional is None:
            fractional = self.use_fractional

        if self.track_lattice_vectors:
            lattice_vectors = self.unique_lattice_vectors[frame_indices]
        else:
            lattice_vectors = self.unique_lattice_vectors

        if fractional != self.use_fractional:
            if fractional:
                positions = np.linalg.solve(lattice_vectors.transpose(1, 2, 0), positions.transpose(1, 2, 0)).transpose(1, 2, 0)
            else:
                positions = np.matmul(positions, lattice_vectors)

        return positions, lattice_vectors