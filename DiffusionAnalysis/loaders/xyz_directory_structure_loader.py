from .base_structure_loader import StructureLoader
from ase.io import read
from ase.atoms import Atoms
import os
from typing import Optional, Iterator

class XYZDirectoryStructureLoader(StructureLoader):
    '''
    Implementation of the StructureLoader interface to wrap a lazy loader iterable of structures from a directory of .xyz files.

    Args:
        dump_dir: str
            Path to the directory containing .xyz files
        structures_slice: slice, optional
            Slice object to specify which files to load. None means load all files in the directory

    Returns:
        Iterator over the structures in the .xyz files.
    '''

    def __init__(self, 
        dump_dir: str, structures_slice: Optional[slice] = None):
        # Get the list of all .xyz files
        all_files = [f for f in os.listdir(dump_dir) if f.endswith('.xyz')]

        # Define a custom sort function that extracts the number from the file name
        def sort_key(file_name):
            number_part = file_name.split('.')[0]  # Assuming file names are in the format 'number.xyz'
            return int(number_part)

        # Sort the files based on the custom sort function
        sorted_files = sorted(all_files, key=sort_key)

        # Apply the slice to the sorted files
        if structures_slice is not None:
            self.files = sorted_files[structures_slice]
        else:
            self.files = sorted_files

        self.file_index = 0
        self.dump_dir = dump_dir
        self.structures_slice = structures_slice
        self._total_steps = None
        print(f'Loading slice {self.structures_slice} from {len(self.files)} files')

    def __iter__(self) -> Iterator[Atoms]:
        self.file_index = 0
        return self

    def __next__(self) -> Atoms:
        if self.file_index >= len(self.files):
            raise StopIteration()

        file_path = os.path.join(self.dump_dir, self.files[self.file_index])
        self.file_index += 1

        try:
            return read(file_path, index=':')
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")

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
        return False
    
    def reset(self) -> None:
        self.file_index = 0

    def get_number_of_atoms(self) -> int:
            '''
            Read the first file to get the number of atoms. SLOW!
            '''
            if len(self.files) == 0:
                raise ValueError("No files found in the directory.")
            
            file_path = os.path.join(self.dump_dir, self.files[0])
            atoms = read(file_path)
            return len(atoms)
