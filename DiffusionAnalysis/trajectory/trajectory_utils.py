import numpy as np
from typing import Union

def convert_to_fractional(positions: np.ndarray, lattice_vectors: Union[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Convert positions from Cartesian coordinates to fractional coordinates.
    
    Args:
        positions (np.ndarray): Array of shape (n_atoms, 3) or (n_frames, n_atoms, 3) containing Cartesian positions.
        lattice_vectors (Union[np.ndarray, np.ndarray]): Array of shape (3, 3) or (n_frames, 3, 3) containing the lattice vectors.
    
    Returns:
        np.ndarray: Array of shape (n_atoms, 3) or (n_frames, n_atoms, 3) containing fractional positions.
    """
    if lattice_vectors.ndim == 2:
        return np.linalg.solve(lattice_vectors.T, positions.T).T
    elif lattice_vectors.ndim == 3:
        return np.linalg.solve(np.transpose(lattice_vectors, (2, 0, 1)), positions.transpose(1, 2, 0)).transpose(2, 0, 1)
    else:
        raise ValueError("Invalid lattice_vectors shape. Expected (3, 3) or (n_frames, 3, 3).")

def convert_to_cartesian(positions: np.ndarray, lattice_vectors: Union[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Convert positions from fractional coordinates to Cartesian coordinates.
    
    Args:
        positions (np.ndarray): Array of shape (n_atoms, 3) or (n_frames, n_atoms, 3) containing fractional positions.
        lattice_vectors (Union[np.ndarray, np.ndarray]): Array of shape (3, 3) or (n_frames, 3, 3) containing the lattice vectors.
    
    Returns:
        np.ndarray: Array of shape (n_atoms, 3) or (n_frames, n_atoms, 3) containing Cartesian positions.
    """
    if lattice_vectors.ndim == 2:
        return np.dot(positions, lattice_vectors)
    elif lattice_vectors.ndim == 3:
        return np.einsum('ijk,ikl->ijl', positions, lattice_vectors)
    else:
        raise ValueError("Invalid lattice_vectors shape. Expected (3, 3) or (n_frames, 3, 3).")

def calculate_center_of_mass(positions: np.ndarray, atomic_numbers: np.ndarray) -> Union[np.ndarray, np.ndarray]:
    """
    Calculate the center of mass of a set of atoms.
    
    Args:
        positions (np.ndarray): Array of shape (n_atoms, 3) or (n_frames, n_atoms, 3) containing atomic positions.
        atomic_numbers (np.ndarray): Array of shape (n_atoms,) containing atomic numbers.
    
    Returns:
        Union[np.ndarray, np.ndarray]: Array of shape (3,) or (n_frames, 3) containing the center of mass coordinates.
    """
    masses = get_atomic_masses(atomic_numbers)
    total_mass = np.sum(masses)
    
    if positions.ndim == 2:
        return np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    elif positions.ndim == 3:
        return np.sum(positions * masses[np.newaxis, :, np.newaxis], axis=1) / total_mass
    else:
        raise ValueError("Invalid positions shape. Expected (n_atoms, 3) or (n_frames, n_atoms, 3).")

def get_atomic_masses(atomic_numbers: np.ndarray) -> np.ndarray:
    """
    Get the atomic masses for a set of atomic numbers.
    
    Args:
        atomic_numbers (np.ndarray): Array of shape (n_atoms,) containing atomic numbers.
    
    Returns:
        np.ndarray: Array of shape (n_atoms,) containing atomic masses.
    """
    from ase.data import atomic_masses
    return np.array([atomic_masses[num] for num in atomic_numbers])
