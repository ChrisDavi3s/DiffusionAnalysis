import os
from ..utils import TimeUnit, TimeData
from .base_structure_loader import StructureLoader
from ase.io import read
from ase.atoms import Atoms
from typing import cast, Optional, Union, Iterator, Tuple, List

class DatDirectoryStructureLoader(StructureLoader):
    '''
    Implementation of StructureLoader for a directory of LAMMPS dump files.

    Parameters
    ----------
    dump_dir : str
        Path to directory containing LAMMPS dump files.
    structures_slice : Optional[Union[slice, int]]
        Slice object or index to select a subset of structures.
    md_temperature : temperature: Union[float, List[float]] = 300
        default=300 k. A float or list of floats representing the temperature(s) of the MD simulation. If a list
        is provided, the length should match the number of steps in the trajectory.
    md_timestep : Optional[float], default=1
        The time step of the MD simulation, representing the time difference between consecutive frames.
    md_time_unit : Union[str, TimeUnit], default='ps'
        The unit of time for the `md_timestep`. Accepted string values are 'fs' (femtoseconds),
        'ps' (picoseconds), 'ns' (nanoseconds), 'us' (microseconds), 'ms' (milliseconds), or 's' (seconds).
        Alternatively, a `TimeUnit` enum value can be provided.
    md_start_offset : Optional[float], default=None
        The starting time offset of the trajectory in the specified time unit. If `None`, the starting time
        will be determined based on the `structures_slice`.
    atom_map : Optional[dict], default=None
        A dictionary to change the atom types in the trajectory. The dictionary should have the format
        `{old_atom_type: new_atom_type}`. If `None`, the atom types will not be changed.

    Returns
    -------
    Iterable[ase.atoms.Atoms]
        An iterable of ASE Atoms objects representing the structures in the trajectory.
    '''

    def __init__(self, 
                 dump_dir: str, 
                 structures_slice: Optional[Union[slice, int]] = None,
                 md_temperature: Optional[Union[float, List[float]]] = None,
                 md_timestep: float = 1, 
                 md_time_unit: Union[str, TimeUnit] = 'ps',
                 md_start_offset: Optional[float] = None,
                 atom_map: Optional[dict] = None):
        
        # Get the list of all .dat files
        all_files = [f for f in os.listdir(dump_dir) if f.endswith('.dat')]

        # Define a custom sort function that extracts the number from the file name
        def sort_key(file_name):
            number_part = file_name.split('.')[-2]
            return int(number_part)

        # Sort the files based on the custom sort function
        sorted_files = sorted(all_files, key=sort_key)

        super().__init__(dump_dir, structures_slice, md_temperature, md_timestep, md_time_unit, md_start_offset, atom_map)

        # Apply the slice or index to the sorted files
        if self.structures_slice is not None:
            if isinstance(self.structures_slice, slice):
                self.files = sorted_files[self.structures_slice]
            else:
                self.files = [sorted_files[self.structures_slice]]
        else:
            self.files = sorted_files

        self.file_index = 0
        self.dump_dir = dump_dir
        self._total_steps = None
        self._total_atoms = None

        print(f'Loading {"slice " + str(self.structures_slice) if isinstance(structures_slice, slice) else "index " + str(structures_slice)} from {len(self.files)} files')

    def __iter__(self) -> Iterator[Atoms]:
        return self

    def __next__(self) -> Atoms:
        if self.file_index < len(self.files):
            try:
                file_path = os.path.join(self.dump_dir, self.files[self.file_index])
                # Any here as we should only get one structure
                atoms: Atoms = cast(Atoms, read(file_path, format='lammps-dump-text'))
                self.file_index += 1
                if self.atom_map is not None:
                    for old_symbol, new_symbol in self.atom_map.items():
                        atoms.symbols[atoms.symbols == old_symbol] = new_symbol
                return atoms
            except:
                raise ValueError(f'Error reading file {self.files[self.file_index]}')
        else:
            raise StopIteration()

    def __len__(self) -> int:
        '''
        Returns the total number of steps (files) in the trajectory.
        '''
        if self._total_steps is None:
            self._total_steps = len(self.files)
        return self._total_steps

    def get_total_steps(self) -> int:
        '''
        Count the number of .dat files in the directory.
        '''
        return len(self.files)

    @property
    def has_lattice_vectors(self) -> bool:
        '''
        Returns True if the trajectory file contains lattice vectors.
        '''
        return True

    def reset(self) -> None:
        self.file_index = 0
        self._total_steps = None
        self._total_atoms = None

    def get_number_of_atoms(self) -> int:
        '''
        Returns the number of atoms in the first step of the trajectory file.
        '''
        if self._total_atoms is None:
            file_path = os.path.join(self.dump_dir, self.files[0])
            atoms: Atoms = cast(Atoms, read(file_path, format='lammps-dump-text'))
            self._total_atoms = len(atoms)
        return self._total_atoms

    @property
    def get_trajectory_time_info(self) -> TimeData:
        """
        Returns a tuple containing the start time, end time, timestep, and time unit of the trajectory.

        Returns
        -------
        Tuple[float, float, float, TimeUnit]
            A tuple containing the start time, end time, timestep, and time unit of the trajectory.
        """
        return super().get_trajectory_time_info
    
    def get_temperature(self, index: int) -> float:
        return super().get_temperature(index)