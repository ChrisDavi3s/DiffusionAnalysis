import os
from .base_structure_loader import StructureLoader
from ase.io import read
from ase.atoms import Atoms
from typing import List, Optional

class ASEListStructureLoader(StructureLoader):
    '''
    A StructureLoader that loads structures from a list of Atoms objects.
    '''

    def __init__(self, 
                 atoms_list: List[Atoms], 
                 structures_slice: Optional[slice] = None):
        '''
        Initialize the AtomsListStructureLoader.

        Parameters
        ----------
        atoms_list : List[Atoms]
            The list of Atoms objects to load.
        structures_slice : slice, optional
            Slice object to specify which structures to load.
            None means load all structures.
            NOT RECOMMENDED TO USE THIS ARGUMENT FOR THIS LOADER 
            (Just pass the slice to the list directly before passing it to this loader)
        '''
        if structures_slice is None:
            self.structures_slice = slice(None)
        else:
            self.structures_slice = structures_slice
        self.atoms_list = atoms_list[self.structures_slice]
        self.iter_obj = iter(self.atoms_list)

    def __iter__(self):
        return self

    def __next__(self) -> Atoms:
        return next(self.iter_obj)

    def get_total_steps(self) -> int:
        return len(self.atoms_list)

    def __len__(self) -> int:
        if self._total_steps is None:
            self._total_steps = len(self.atoms_list)
        return self._total_steps
    
    @property
    def has_lattice_vectors(self) -> bool:
        return self.atoms_list[0].cell is not None
    
    def reset(self) -> None:
         self.iter_obj = iter(self.atoms_list)

    def get_number_of_atoms(self) -> int:
        return len(self.atoms_list[0])