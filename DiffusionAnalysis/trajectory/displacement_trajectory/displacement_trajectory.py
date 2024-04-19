import numpy as np
from typing import List, Union, Tuple, Optional
from ...utils import AtomSubsetMap, TimeData, TimeUnit
from enum import Enum
from abc import ABC, abstractmethod
from .base_displacement_trajectory import BaseDisplacementTrajectory, TrajectoryDirectionEnum

class DisplacementTrajectory(BaseDisplacementTrajectory):
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
        super().__init__(displacement_data, tracer_specs, framework_specs, atoms_map, time_data, trajectory_type, is_com, is_drift_corrected)

