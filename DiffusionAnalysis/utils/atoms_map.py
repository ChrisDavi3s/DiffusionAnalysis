from typing import Dict, Set, Union, List, Optional
from ase import Atoms
from ase.data import chemical_symbols
import numpy as np

class AtomsMap:
    """
    A class to map atomic numbers to atom indices and provide filtering functionality.
    """
    def __init__(self, atoms: Atoms, atom_specs: Optional[List[Union[int, str]]] = None):
        """
        Initialize an AtomsMap object.

        Args:
            atoms (Optional[Atoms]): An ASE Atoms object to create the atom indices map from.
            atom_specs (Optional[List[Union[int, str]]]): A list of atom indices or symbols to filter the atom indices map.
        """
        self.atom_indices_map: Dict[int, np.ndarray] = {}
        if atoms is not None:
            self._create_atom_indices_map(atoms)
        if atom_specs is not None:
            self._filter_atom_indices_map(atom_specs)

    def _create_atom_indices_map(self, atoms: Atoms) -> None:
        """
        Create the initial atom indices map from an ASE Atoms object.

        Args:
            atoms (Atoms): An ASE Atoms object.
        """
        atomic_numbers = atoms.get_atomic_numbers()
        # Get the unique atomic numbers and the indices of the unique atomic numbers in the original array
        #For example, if atomic_numbers = [8, 1, 1, 8, 6], then unique_atomic_numbers would be [1, 6, 8] and
        # indices would be [2, 0, 0, 2, 1]
        unique_atomic_numbers, indices = np.unique(atomic_numbers, return_inverse=True)
        # Create a dictionary mapping atomic numbers to numpy arrays of atom indices
        self.atom_indices_map = {num: np.where(indices == i)[0] for i, num in enumerate(unique_atomic_numbers)}

    def _filter_atom_indices_map(self, atom_specs: List[Union[int, str]]) -> None:
        """
        Filter the atom indices map based on specific atom indices or symbols.

        Args:
            atom_specs (List[Union[int, str]]): A list of atom indices or symbols to filter the atom indices map.
        """
        filtered_indices = set()
        for spec in atom_specs:
            if isinstance(spec, int):
                filtered_indices.add(spec)
            elif isinstance(spec, str):
                atomic_number = chemical_symbols.index(spec)
                if atomic_number in self.atom_indices_map:
                    filtered_indices.update(self.atom_indices_map[atomic_number])
            else:
                raise ValueError(f"Invalid atom specification: {spec}")

        for atomic_number in list(self.atom_indices_map.keys()):
            self.atom_indices_map[atomic_number] = np.array(list(set(self.atom_indices_map[atomic_number]) & filtered_indices))
            if len(self.atom_indices_map[atomic_number]) == 0:
                del self.atom_indices_map[atomic_number]

    def get_indices(self, specs: Optional[List[Union[int, str]]] = None) -> np.ndarray:
        """
        Get the atom indices based on the provided specifications (indices or symbols).

        Args:
            specs (Optional[List[Union[int, str]]]): A list of atom indices or symbols. If None, all indices are returned.

        Returns:
            np.ndarray: The atom indices corresponding to the provided specifications.
        """
        if specs is None:
            return np.concatenate(list(self.atom_indices_map.values()))

        indices = []
        for spec in specs:
            if isinstance(spec, int):
                indices.append(spec)
            elif isinstance(spec, str):
                atomic_number = chemical_symbols.index(spec)
                if atomic_number in self.atom_indices_map:
                    indices.extend(self.atom_indices_map[atomic_number])
            else:
                raise ValueError(f"Invalid specification: {spec}")

        return np.array(indices)
    
    def __len__(self) -> int:
        """
        Get the total number of atom indices in the atom indices map.

        Returns:
            int: The total number of atom indices.
        """
        return sum(len(indices) for indices in self.atom_indices_map.values())

    def get_atom_indices_map(self) -> Dict[int, np.ndarray]:
        """
        Get the atom indices map.

        Returns:
            Dict[int, np.ndarray]: A dictionary mapping atomic numbers to numpy arrays of atom indices.
        """
        return self.atom_indices_map

    def get_all_atomic_numbers(self) -> List[int]:
        """
        Get the atomic numbers present in the atom indices map.

        Returns:
            List[int]: A list of atomic numbers.
        """
        return list(self.atom_indices_map.keys())

    def get_chemical_symbols(self) -> List[str]:
        """
        Get the chemical symbols corresponding to the atomic numbers in the atom indices map.

        Returns:
            List[str]: A list of chemical symbols.
        """
        return [chemical_symbols[num] for num in self.atom_indices_map.keys()]
    
    def get_atomic_numbers_from_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Get the atomic numbers for a given set of indices.

        Args:
            indices (np.ndarray): An array of atom indices.

        Returns:
            np.ndarray: An array of atomic numbers corresponding to the indices, preserving the order.
        """
        atomic_numbers = np.full_like(indices, 0 , dtype=int)
        for atomic_number, atom_indices in self.atom_indices_map.items():
            mask = np.isin(indices, atom_indices)
            atomic_numbers[mask] = atomic_number
        return atomic_numbers
    
    def get_atom_strings(self) -> List[str]:
        """
        Get the atom strings in this atom map.

        Returns:
            List[str]: A list of atom strings.
        """
        return [f"{chemical_symbols[num]}{i}" for num, indices in self.atom_indices_map.items() for i in indices]