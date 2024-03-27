from .base_structure_loader import StructureLoader
from ase.io import read, iread
from ase.atoms import Atoms
from typing import Optional, Iterator, Tuple, Union
from ..utils.time_unit import TimeUnit

class XYZStructureLoader(StructureLoader):
    '''
    Implementation of the StructureLoader interface to wrap a lazy loader iterable of structures from a xyz file.

    Parameters
    ----------
    filepath : str
        Path to the trajectory file.
    structures_slice : Optional[slice]
        Slice object to specify which steps to load. None means load all steps in the file.
    md_timestep : Optional[float], default=1
        The time step of the MD simulation, representing the time difference between consecutive frames.
    md_time_unit : Union[str, TimeUnit], default='ps'
        The unit of time for the `md_timestep`. Accepted string values are 'fs' (femtoseconds),
        'ps' (picoseconds), 'ns' (nanoseconds), 'us' (microseconds), 'ms' (milliseconds), or 's' (seconds).
        Alternatively, a `TimeUnit` enum value can be provided.
    md_start_offset : Optional[float], default=None
        The starting time offset of the trajectory in the specified time unit. If `None`, the starting time
        will be determined based on the `structures_slice`.

    Returns
    -------
    Iterable[ase.atoms.Atoms]
        An iterable of ASE Atoms objects representing the structures in the trajectory.
    '''

    def __init__(self, filepath: str, structures_slice: Optional[slice] = None,
                 md_timestep: Optional[float] = 1, md_time_unit: Union[str, TimeUnit] = 'ps',
                 md_start_offset: Optional[float] = None):
        super().__init__(filepath, structures_slice, md_timestep, md_time_unit, md_start_offset)
        self._total_steps = None
        self._total_atoms = None
        self._iterator = iter(iread(self.filepath, index=self.structures_slice))

    def __iter__(self) -> Iterator[Atoms]:
        return self

    def __next__(self) -> Atoms:
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopIteration()

    def _count_steps(self) -> int:
        '''
        Count the number of steps in a .xyz file. Adds around 0.3s for 10k steps.
        '''
        with open(self.filepath, 'r') as file:
            first_line = file.readline().strip()
            number_of_atoms = int(first_line) if first_line.isdigit() else None
            if number_of_atoms is None:
                raise ValueError("The file does not seem to be in the correct .xyz format.")
            total = sum(1 for line in file if line.strip() == str(number_of_atoms))

        # Apply the slice to the total and then calculate the number of steps
        if self.structures_slice is not None:
            start, stop, step = self.structures_slice.indices(total)
            total = len(range(start, stop, step))

        return total

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
        return False

    def reset(self) -> None:
        '''
        Reset the iterator to the first step.
        '''
        self._iterator = iter(iread(self.filepath, index=self.structures_slice))

    def get_number_of_atoms(self) -> int:
        '''
        Read the first step to get the number of atoms. SLOW!
        '''
        if self._total_atoms is None:
            self._total_atoms = len(read(self.filepath, index=0))
        return self._total_atoms

    @property
    def get_trajectory_time_info(self) -> Tuple[float, float, float, TimeUnit]:
        """
        Returns a tuple containing the start time, end time, timestep, and time unit of the trajectory.

        Returns
        -------
        Tuple[float, float, float, TimeUnit]
            A tuple containing the start time, end time, timestep, and time unit of the trajectory.
        """
        return super().get_trajectory_time_info