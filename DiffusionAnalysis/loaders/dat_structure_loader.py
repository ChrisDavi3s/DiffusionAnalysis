from .base_structure_loader import StructureLoader
from ase.io import read, iread
from ase.atoms import Atoms
from typing import Optional, Union

class DatStructureLoader(StructureLoader):
    '''
    Implementation of the StructureLoader interface to wrap a lazy loader iterable of structures from a xyz file.
    
    Args:
        filepath: str
            Path to the trajectory file
        structures_slice: slice or int
            Slice object or integer index to specify which steps to load. None for the end means load all steps in the file
    
    Returns:
        Iterable ASE Atoms object to iterate over the structures in the trajectory.
    '''
    def __init__(self, filepath: str, structures_slice: Optional[Union[slice, int]] = None):
        self.filepath = filepath
        self.structures_slice = structures_slice
        self._total_steps = None
        self._iterator = iter(iread(self.filepath, index=self.structures_slice))
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Atoms:
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopIteration()
        
    #TODO A dat file already has this information in the header (maybe?), so we can just read it from there
    def _count_steps(self) -> int:
        '''
        Count the number of steps in a .dat file. Adds around 0.3s for 10k steps.

        We actually cheat here and read the first line of the file - this says 'TIMESTEP' and then the number of the timestep.
        So we just count the number of lines that say TIMESTEP.
        '''
        with open(self.filepath, 'r') as file:
            first_line = file.readline().strip() # Read the first line
            total = sum(1 for line in file if line.strip() == str(first_line))
        
        # Apply the slice or index to the total and then calculate the number of steps
        if isinstance(self.structures_slice, slice):
            start, stop, step = self.structures_slice.indices(total)
            total = len(range(start, stop, step))
        elif isinstance(self.structures_slice, int):
            total = 1
        
        return total

    #TODO A dat file already has this information in the header (I think?), so we can just read it from there
    def get_total_steps(self) -> int:
        '''
        Count the number of steps in a .xyz file. Adds around 0.3s for 10k steps.
        '''
        if self._total_steps is None:
            self._total_steps = self._count_steps()
        return self._total_steps
    
    def __len__(self) -> int:
        return super().__len__()
    
    @property
    def has_lattice_vectors(self) -> bool:
        '''
        Returns True if the trajectory file contains lattice vectors.
        '''
        return True
    
    def reset(self) -> None:
        '''
        Reset the iterator to the first step.
        '''
        self._iterator = iter(iread(self.filepath, index=self.structures_slice))
    
    def get_number_of_atoms(self) -> int:
        '''
        Read the first step to get the number of atoms. SLOW!
        '''
        return len(read(self.filepath, index=0))