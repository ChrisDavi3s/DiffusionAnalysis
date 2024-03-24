import os
from .base_structure_loader import StructureLoader
from ase.io import read
from ase.atoms import Atoms
from typing import cast, Optional, Union

class DatDirectoryStructureLoader(StructureLoader):
    '''
    Implementation of StructureLoader for a directory of LAMMPS dump files.
    
    Args:
        dump_dir : str
            Path to directory containing LAMMPS dump files.
        structures_slice : slice or int
            Slice object or index to select a subset of structures.
    
    Returns:
        Iterable ASE Atoms object to iterate over the structures in the trajectory.
    '''
    def __init__(self, dump_dir: str, structures_slice: Optional[Union[slice, int]] = None):
        # Get the list of all .dat files
        all_files = [f for f in os.listdir(dump_dir) if f.endswith('.dat')]
        
        # Define a custom sort function that extracts the number from the file name
        def sort_key(file_name):
            number_part = file_name.split('.')[-2]
            return int(number_part)
        
        # Sort the files based on the custom sort function
        sorted_files = sorted(all_files, key=sort_key)
        
        # Apply the slice or index to the sorted files
        if structures_slice is not None:
            if isinstance(structures_slice, slice):
                self.files = sorted_files[structures_slice]
            else:
                self.files = [sorted_files[structures_slice]]
        else:
            self.files = sorted_files
        
        self.file_index = 0
        self.dump_dir = dump_dir
        self.structures_slice = structures_slice
        self._total_steps = None
        
        print(f'Loading {"slice " + str(self.structures_slice) if isinstance(structures_slice, slice) else "index " + str(structures_slice)} from {len(self.files)} files')
    
    def __iter__(self) -> StructureLoader:
        return self
    
    def __next__(self) -> Atoms:
        if self.file_index < len(self.files):
            try:
                file_path = os.path.join(self.dump_dir, self.files[self.file_index])
                # Any here as we _should_ only get one structure
                atoms: Atoms = cast(Atoms, read(file_path, format='lammps-dump-text'))
                self.file_index += 1
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
        Count the number of .xyz files in the directory.
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
    
    def get_number_of_atoms(self) -> int:
        '''
        Returns the number of atoms in the first step of the trajectory file.
        '''
        file_path = os.path.join(self.dump_dir, self.files[0])
        atoms: Atoms = cast(Atoms, read(file_path, format='lammps-dump-text'))
        return len(atoms)