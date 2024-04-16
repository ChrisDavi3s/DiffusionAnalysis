from DiffusionAnalysis.utils.time_unit import TimeUnit
from abc import ABC, abstractmethod
from ase.atoms import Atoms
from typing import Optional, Iterator, Union, Tuple, List

class StructureLoader(ABC):
    '''
    Abstract class (interface) to load and store an iterable of structures with lazy loading.

    The `StructureLoader` class provides a generic interface for loading and iterating over structures
    from a trajectory file. It supports lazy loading, allowing efficient memory usage by loading structures
    on-the-fly as they are accessed.

    Concrete implementations of the `StructureLoader` class should inherit from this abstract base class
    and provide implementations for the abstract methods.

    Parameters
    ----------
    filepath : str
        The path to the trajectory file.
    structures_slice : Optional[Union[slice, int]]
        A slice object or an integer specifying the range or specific steps to load from the trajectory.
        If a slice object is provided, it should have the format `slice(start, stop, step)`, where `start`
        is the starting step (inclusive), `stop` is the ending step (exclusive), and `step` is the step size.
        If an integer is provided, it represents a specific step to load.
        If `None`, all steps from the trajectory will be loaded.
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

    Attributes
    ----------
    filepath : str
        The path to the trajectory file.
    structures_slice : Optional[Union[slice, int]]
        The slice object or integer specifying the range or specific steps to load from the trajectory.
    md_temperature : Union[float, List[float]]
        The temperature of the MD simulation.
    timestep : float
        The time step of the MD simulation, adjusted based on the `structures_slice` and `md_timestep`.
    time_unit : TimeUnit
        The unit of time for the `timestep`, adjusted based on the `structures_slice` and `md_time_unit`.
    start_time : float
        The starting time of the trajectory, considering the `md_start_offset` and `structures_slice`.
    end_time : float
        The ending time of the trajectory, calculated based on the `start_time`, `structures_slice`, and `timestep`.

    Methods
    -------
    __iter__() -> Iterator[Atoms]
        Returns an iterator over the structures in the trajectory.
    __next__() -> Atoms
        Returns the next structure in the trajectory.
    __len__() -> int
        Returns the total number of steps in the trajectory.
    get_total_steps() -> int
        Counts the total number of steps in the trajectory file.
    has_lattice_vectors() -> bool
        Returns `True` if the trajectory file contains lattice vectors.
    reset() -> None
        Resets the iterator to the first step.
    get_number_of_atoms() -> int
        Returns the number of atoms in the first step of the trajectory file.
    get_trajectory_time_info() -> Tuple[float, float, float, TimeUnit]
        Returns a tuple containing the start time, end time, timestep, and time unit of the trajectory.
    '''

    @abstractmethod
    def __init__(self, filepath: str, 
                structures_slice: Optional[Union[slice, int]], 
                md_temperature: Optional[Union[float, List[float]]] = None,
                md_timestep: float = 1, 
                md_time_unit: Union[str, TimeUnit] = 'ps', 
                md_start_offset: Optional[float] = None,
                atom_map: Optional[dict] = None):

        '''
        Initializes the StructureLoader instance.

        When initializing, the time units and timestep are reshaped based on the provided slice.
        '''
        self.filepath = filepath
        self.structures_slice = structures_slice
        self.temperature = md_temperature
        self._total_steps = 0
        self.timestep = md_timestep
        self.time_unit = TimeUnit(md_time_unit) if isinstance(md_time_unit, str) else md_time_unit
        self.start_time = 0
        self.end_time = 0
        self.atom_map = atom_map 

        if self.structures_slice is None:
            start = 0
            stop = self.get_total_steps()
        elif isinstance(self.structures_slice, slice):
            step = self.structures_slice.step if self.structures_slice.step is not None else 1
            self.timestep *= step
            self.timestep, self.time_unit = TimeUnit.adjust_timestep_and_unit(self.timestep, self.time_unit)

            start = self.structures_slice.start if self.structures_slice.start is not None else 0
            stop = self.structures_slice.stop if self.structures_slice.stop is not None else self.get_total_steps()
        else:
            raise ValueError("structures_slice must be either None or a slice object.")

        if md_start_offset is None:
            self.start_time = start * self.timestep
        else:
            self.start_time = md_start_offset

        self.end_time = self.start_time + (stop - start) * self.timestep

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
        Counts the total number of steps in the trajectory file.
        '''
        pass

    @property
    def has_lattice_vectors(self) -> bool:
        '''
        Returns True if the trajectory file contains lattice vectors.
        '''
        raise NotImplementedError("Subclasses must implement has_lattice_vectors")

    @abstractmethod
    def reset(self) -> None:
        '''
        Resets the iterator to the first step.
        '''
        pass

    @abstractmethod
    def get_number_of_atoms(self) -> int:
        '''
        Returns the number of atoms in the first step of the trajectory file.
        '''
        pass

    @property
    def get_trajectory_time_info(self) -> Tuple[float, float, float, TimeUnit]:
        """
        Returns a tuple containing the start time, end time, timestep, and time unit of the trajectory.

        Returns
        -------
        Tuple[float, float, float, TimeUnit]
            A tuple containing the start time, end time, timestep, and time unit of the trajectory.
        """
        return self.start_time, self.end_time, self.timestep, self.time_unit
    
    @abstractmethod
    def get_temperature(self, index: int) -> float:
        if isinstance(self.temperature, list):
            return self.temperature[index]
        else:
            return self.temperature