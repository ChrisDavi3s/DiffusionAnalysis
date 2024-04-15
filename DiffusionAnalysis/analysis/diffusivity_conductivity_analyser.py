import numpy as np
from typing import Tuple

class DiffusionCoefficientAnalyser:
    """
    Class for calculating the diffusion coefficient and conductivity from the mean squared displacement (MSD) data.
    """

    @staticmethod
    def calculate_diffusion_coefficient(msd_data: np.ndarray, 
                                        tau_values: np.ndarray,
                                        step_size: float, 
                                        num_atoms: int, 
                                        dimension: int = 3) -> np.ndarray:
        """
        Calculate the diffusion coefficient from the MSD data.

        Args:
            msd_data (np.ndarray): The MSD data (in Å^2).
            time_values (np.ndarray): The time values corresponding to the MSD data (in step).
            step_size (float): The time step size (in s).
            dimension (int): The dimensionality of the system (1, 2, or 3).
            num_atoms (int): The number of atoms used in the MSD calculation.

        Returns:
            np.ndarray: The diffusion coefficients (in cm^2/s).
        """
        time_step = step_size * tau_values
        return (msd_data * 1e-16) /(2 * dimension * num_atoms * time_step)  # Convert from Å^2/s to cm^2/s

    @staticmethod
    def calculate_com_diffusion_coefficient(msd_data: np.ndarray, 
                                            tau_values: np.ndarray, 
                                            step_size: float,
                                            num_atoms: int, 
                                            dimension: int = 3) -> np.ndarray:
        """
        Calculate the diffusion coefficient of the center of mass (CoM) from the MSD data.

        Args:
            msd_data (np.ndarray): The MSD data of the CoM (in Å^2).
            tau_values (np.ndarray): The time values corresponding to the MSD data (in step).
            strp_size (float): The time step size (in s).
            dimension (int): The dimensionality of the system (1, 2, or 3).
            num_atoms (int): The number of atoms used in the MSD calculation.

        Returns:
            np.ndarray: The diffusion coefficients of the CoM (in cm^2/s).
        """
        time_step = step_size * tau_values
        return (num_atoms * msd_data * 1e-16) / (2 * dimension * time_step)   # Convert from Å^2/s to cm^2/s

    @staticmethod
    def calculate_conductivity(diffusion_coefficients: np.ndarray, temperature: float, num_atoms: int, volume: float) -> np.ndarray:
        """
        Calculate the conductivity from the diffusion coefficients.

        Args:
            diffusion_coefficients (np.ndarray): The diffusion coefficients (in cm^2/s).
            temperature (float): The temperature (in K) for calculating the conductivity.
            num_atoms (int): The number of atoms used in the MSD calculation.
            volume (float): The volume of the system (in cm^3).

        Returns:
            np.ndarray: The conductivity (in S/cm).
        """
        q = 1.602e-19  # Elementary charge in Coulombs
        kb = 1.380649e-23  # Boltzmann constant in J/K
        return (q ** 2 * num_atoms) * diffusion_coefficients / (kb * temperature*volume)

    def analyze(self, msd_data: np.ndarray, time_values: np.ndarray, temperature: float, dimension: int, num_atoms: int, volume: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the diffusion coefficient and conductivity analysis.

        Args:
            msd_data (np.ndarray): The MSD data (in Å^2).
            time_values (np.ndarray): The time values corresponding to the MSD data (in s).
            temperature (float): The temperature (in K) for calculating the conductivity.
            dimension (int): The dimensionality of the system (1, 2, or 3).
            num_atoms (int): The number of atoms used in the MSD calculation.
            volume (float): The volume of the system (in cm^3).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The diffusion coefficients (in cm^2/s), CoM diffusion coefficients (in cm^2/s), and conductivity (in S/cm).
        """
        diffusion_coefficients = self.calculate_diffusion_coefficient(msd_data, time_values, dimension, num_atoms)
        com_diffusion_coefficients = self.calculate_com_diffusion_coefficient(msd_data, time_values, dimension)
        conductivity = self.calculate_conductivity(diffusion_coefficients, temperature, num_atoms, volume)

        return diffusion_coefficients, com_diffusion_coefficients, conductivity
    
    @staticmethod
    def convert_conductivity_to_different_temperature(conductivity: np.ndarray,
                                                      activation_energy: float, 
                                                      temperature: float, 
                                                      new_temperature: float) -> np.ndarray:
        """
        Convert the conductivity from one temperature to another. This uses an arrhenius-like relationship.

        ln(sigma) =  ln(sigma0) - E_a / k_b * (1/T0 - 1/T)

        where:

        Args:

            conductivity (np.ndarray): The conductivity (in S/cm).
            activation_energy (float): The activation energy (in J).
            temperature (float): The original temperature (in K).
            new_temperature (float): The new temperature (in K).

        Returns:
            np.ndarray: The converted conductivity (in S/cm).
        """
        e_a = activation_energy  #J
        kb = 1.380649e-23
        ln_sigma = np.log(conductivity) - (e_a / kb * (-1 / temperature + 1 / new_temperature))
        return np.exp(ln_sigma)