from .base_structure_loader import StructureLoader
from ase.atoms import Atoms
from typing import List, Optional, Union, Iterator, Tuple
from ..utils.time_utils import TimeUnit,TimeData

class ASEListStructureLoader(StructureLoader):
    '''
    A StructureLoader that loads structures from a list of Atoms objects.

    Parameters
    ----------
    atoms_list : List[Atoms]
        The list of Atoms objects to load.
    structures_slice : Optional[Union[slice, int]]
        Slice object or index to specify which structures to load.
        None means load all structures.
        NOT RECOMMENDED TO USE THIS ARGUMENT FOR THIS LOADER
        (Just pass the slice to the list directly before passing it to this loader)
    md_temperature : temperature: Union[float, List[float]] = 300
        default=300 k. A float or list of floats representing the temperature(s) of the MD simulation. If a list
        is provided, the length should match the number of steps in the trajectory.
    md_timestep : float, default=1
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
    '''

    def __init__(self,
                 atoms_list: List[Atoms],
                 structures_slice: Optional[Union[slice, int]] = None,
                 md_temperature: Optional[Union[float, List[float]]] = None,
                 md_timestep: float = 1,
                 md_time_unit: Union[str, TimeUnit] = 'ps',
                 md_start_offset: Optional[float] = None,
                 atom_map: Optional[dict] = None):
        '''
        Initialize the ASEListStructureLoader.
        '''
        self.atoms_list = atoms_list
        super().__init__('', structures_slice, md_temperature ,md_timestep, md_time_unit, md_start_offset, atom_map)

        if self.structures_slice is None:
            self.atoms_list = atoms_list
        elif isinstance(self.structures_slice, slice): 
            self.atoms_list = atoms_list[self.structures_slice]
        else:
            self.atoms_list = [atoms_list[self.structures_slice]]

        self.iter_obj = iter(self.atoms_list)

    def __iter__(self) -> Iterator[Atoms]:
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
        self._total_steps = None
        self.iter_obj = iter(self.atoms_list)

    def get_number_of_atoms(self) -> int:
        # this is quick enough to not need to cache it
        return len(self.atoms_list[0])

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
