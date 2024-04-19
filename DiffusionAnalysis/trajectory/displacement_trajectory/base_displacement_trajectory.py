import numpy as np
from typing import List, Union, Tuple, Optional
from ...utils import AtomSubsetMap, TimeData, TimeUnit
from enum import Enum
from abc import ABC, abstractmethod

class TrajectoryDirectionEnum(Enum):
    """
    Enumeration for the type of MSD data.
    """
    # 3D MSD (N,1)
    THREE_DIMENSIONAL = '3d'
    # MSD in the x, y, and z direction (N,3)
    X_Y_Z = 'xyz' 
    # MSD in the normalised direction [a, b, c] (N,1) 
    LATTICE_VECTOR = 'lattice'

class BaseDisplacementTrajectory(ABC):
    """
    Base class for displacement-related trajectory data.
    """

    def __init__(self,
                 displacement_data: np.ndarray,
                 tracer_specs: Optional[List[Union[int, str]]],
                 framework_specs: Optional[List[Union[int, str]]],
                 atoms_map: AtomSubsetMap,
                 time_data: TimeData,
                 trajectory_type: TrajectoryDirectionEnum,
                 is_com: bool = False,
                 is_drift_corrected: bool = False):
        self._displacement_data = displacement_data
        self._tracer_specs = tracer_specs
        self._framework_specs = framework_specs
        self._atoms_map = atoms_map
        self._time_data = time_data
        self._trajectory_type = trajectory_type
        self._is_com = is_com
        self._is_drift_corrected = is_drift_corrected

    def get_displacement_data(self) -> np.ndarray:
        """
        Get the displacement data.

        Returns:
            np.ndarray: Array containing the displacement data.
        """
        return self._displacement_data

    def get_atom_species(self) -> List[str]:
        """
        Get the atom specifications.

        Returns:
            Optional[List[Union[int, str]]]: List of atom specifications, or None if not available.
        """
        return self._atoms_map.get_subset_chemical_symbols()

    def get_atoms_map(self) -> AtomSubsetMap:
        """
        Get the atoms map.

        Returns:
            AtomsMap: The atoms map associated with the trajectory.
        """
        return self._atoms_map

    def get_time_data(self) -> TimeData:
        """
        Get the time data.

        Returns:
            TimeData: The time data associated with the trajectory.
        """
        return self._time_data

    def get_trajectory_type(self) -> TrajectoryDirectionEnum:
        """
        Get the trajectory type.

        Returns:
            TrajectoryDirectionEnum: The type of the trajectory.
        """
        return self._trajectory_type

    def is_com(self) -> bool:
        """
        Check if the trajectory is for the center of mass.

        Returns:
            bool: True if the trajectory is for the center of mass, False otherwise.
        """
        return self._is_com

    def is_drift_corrected(self) -> bool:
        """
        Check if the trajectory is drift-corrected.

        Returns:
            bool: True if the trajectory is drift-corrected, False otherwise.
        """
        return self._is_drift_corrected