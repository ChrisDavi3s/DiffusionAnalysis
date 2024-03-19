from enum import Enum

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
