from enum import Enum
from typing import Tuple, Union

class TimeUnit(Enum):
    '''
    Time unit enumeration for the mean squared displacement (MSD) analysis.
    TODO - this class should probably be made optional and the user should be able to specify the time unit in the plot_msd method.
    '''
    FEMTOSECONDS = 'fs'
    PICOSECONDS = 'ps'
    NANOSECONDS = 'ns'
    MICROSECONDS = 'us'
    MILLISECONDS = 'ms'
    SECONDS = 's'

    def get_time_factor(self) -> float:
        """
        Get the time factor based on the time unit.

        Returns:
            float: The time factor to convert the timestep to the desired time unit.
        """
        if self == TimeUnit.FEMTOSECONDS:
            return 1e-15
        elif self == TimeUnit.PICOSECONDS:
            return 1e-12
        elif self == TimeUnit.NANOSECONDS:
            return 1e-9
        elif self == TimeUnit.MICROSECONDS:
            return 1e-6
        elif self == TimeUnit.MILLISECONDS:
            return 1e-3
        else:  # SECONDS
            return 1
        
    @staticmethod
    def adjust_timestep_and_unit(timestep: float, time_unit: Union[str, 'TimeUnit']) -> Tuple[float, 'TimeUnit']:
        """
        Adjust the timestep and time unit based on the magnitude of the timestep.

        Returns:
            Tuple[float, TimeUnit]: The adjusted timestep and time unit.
        """
        if isinstance(time_unit, str):
            time_unit = TimeUnit(time_unit)

        time_units = list(TimeUnit)
        current_index = time_units.index(time_unit)

        while timestep >= 1000 and current_index < len(time_units) - 1:
            timestep /= 1000
            current_index += 1
            time_unit = time_units[current_index]

        while timestep < 1 and current_index > 0:
            timestep *= 1000
            current_index -= 1
            time_unit = time_units[current_index]

        return timestep, time_unit