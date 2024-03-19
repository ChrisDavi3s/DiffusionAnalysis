from abc import ABC, ABCMeta, abstractmethod
from ase.atoms import Atoms
from typing import Optional, Iterator

class StructureLoader(ABC):
    '''
    Abstract class (interface) to load and store an iterable of structures that lazy load.

    Parameters
    ----------
    filepath: str
        Path to the trajectory file
    steps_to_load: slice
        Slice object to specify which steps to load. None for the end means load all steps in the file

    Returns
    -------
    AseIterable object to iterate over the structures in the trajectory.
    '''

    @abstractmethod
    def __init__(self, filepath: str, structures_slice: Optional[slice]):
        self.filepath = filepath
        self.structures_slice = structures_slice
        self._total_steps = None

    @abstractmethod
    def __iter__(self) -> Iterator[Atoms]:
        pass

    @abstractmethod
    def __next__(self) -> Atoms:
        '''
        Returns the next structure in the trajectory.
        '''
        pass

    @abstractmethod
    def __len__(self) -> int:
        '''
        Returns the total number of steps in the trajectory.
        '''
        if self._total_steps is None:
            self._total_steps = self.get_total_steps()
        return self._total_steps

    @abstractmethod
    def get_total_steps(self) -> int:
        '''
        Count the number of steps in the trajectory file.
        '''
        pass

    @property
    def has_lattice_vectors(self) -> bool:
        '''
        Returns True if the trajectory file contains lattice vectors.
        '''
        pass

    @abstractmethod
    def reset(self) -> None:
        '''
        Reset the iterator to the first step.
        '''
        pass

    @staticmethod
    def get_number_of_atoms(self) -> int:
        '''
        Returns the number of atoms in the first step of the trajectory file.
        '''
        pass