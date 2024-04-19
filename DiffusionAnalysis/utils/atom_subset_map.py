from typing import List, Optional, Union
import numpy as np
from ase import Atoms

class AtomSubsetMap:
    def __init__(self, original_atoms: Atoms, selection: Optional[Union[List[Union[str, int]], slice, np.ndarray]] = None):
        """
        Initialize an AtomSubsetMap object.

        Args:
            original_atoms (Atoms): The original Atoms object.
            selection (Optional[Union[List[Union[str, int]], slice, np.ndarray]]): The selection to create a subset.
                It can be a list of indices or symbols, a slice object, or a NumPy array of indices.
                If None, the entire original_atoms is used.
        """
        self._original_atoms = original_atoms
        self._original_indices = np.arange(len(original_atoms))
        self._subset_indices = self._get_sorted_subset_indices(selection) if selection is not None else self._original_indices.copy()
        self._chemical_symbols = self._original_atoms.get_chemical_symbols()
        self._atomic_numbers = self._original_atoms.get_atomic_numbers()

    def select(self, selection: Union[List[Union[int, str]], slice]) -> 'AtomSubsetMap':
        """
        Create a new AtomSubsetMap object based on the given selection.

        Args:
            selection (Union[List[Union[int, str]], slice]): The selection to create a subset.
                It can be a list of indices or symbols, or a slice object.

        Returns:
            AtomSubsetMap: A new AtomSubsetMap object representing the selected subset.
        """
        subset_indices = self._get_sorted_subset_indices(selection)
        return AtomSubsetMap(self._original_atoms, self._original_indices[subset_indices])

    def get_subset_atomic_numbers(self) -> np.ndarray:
        """
        Get the atomic numbers of the atoms in the subset.

        Returns:
            np.ndarray: The atomic numbers of the atoms in the subset.
        """
        return self._atomic_numbers[self._subset_indices]

    def get_subset_chemical_symbols(self) -> List[str]:
        """
        Get the chemical symbols of the atoms in the subset.

        Returns:
            List[str]: The chemical symbols of the atoms in the subset.
        """
        return [self._chemical_symbols[i] for i in self._subset_indices]

    def get_subset_indices_by_atomic_number(self, atomic_number: int) -> np.ndarray:
        """
        Get the subset indices of the atoms with the given atomic number.

        Args:
            atomic_number (int): The atomic number to search for.

        Returns:
            np.ndarray: The subset indices of the atoms with the given atomic number.
        """
        mask = np.isin(self._atomic_numbers[self._subset_indices], atomic_number)
        return np.where(mask)[0]

    def get_subset_indices_by_symbol(self, symbol: str) -> np.ndarray:
        """
        Get the subset indices of the atoms with the given chemical symbol.

        Args:
            symbol (str): The chemical symbol to search for.

        Returns:
            np.ndarray: The subset indices of the atoms with the given chemical symbol.

        Raises:
            ValueError: If the chemical symbol is not found in the subset.
        """
        indices = np.where(np.isin(np.array(self._chemical_symbols)[self._subset_indices], symbol))[0]
        if len(indices) == 0:
            raise ValueError(f"Chemical symbol '{symbol}' not found in the subset.")
        return indices

    @property
    def atoms(self) -> Atoms:
        """
        Get the Atoms object representing the subset.

        Returns:
            Atoms: The Atoms object representing the subset.
        """
        return self._original_atoms[self._subset_indices]

    def __len__(self) -> int:
        """
        Get the number of atoms in the subset.

        Returns:
            int: The number of atoms in the subset.
        """
        return len(self._subset_indices)

    def __getitem__(self, selection: Optional[Union[int, slice, List[Union[int, str]]]]) -> 'AtomSubsetMap':
        """
        Create a new AtomSubsetMap object based on the given selection.

        Args:
            selection (Optional[Union[int, slice, List[Union[int, str]]]]): The selection to create a subset.
                It can be an integer index, a slice object, or a list of indices or symbols.
                If None, the current AtomSubsetMap object is returned.

        Returns:
            AtomSubsetMap: A new AtomSubsetMap object representing the selected subset.
        """
        if selection is None:
            return self
        if isinstance(selection, int):
            selection = [selection]
        return self.select(selection)

    def get_subset_indices(self) -> np.ndarray:
        """
        Get the subset indices.

        Returns:
            np.ndarray: The subset indices.
        """
        return self._subset_indices.copy()

    def get_original_indices(self) -> np.ndarray:
        """
        Get the original indices of the atoms in the subset.

        Returns:
            np.ndarray: The original indices of the atoms in the subset.
        """
        return self._original_indices[self._subset_indices]

    def get_subset_indices_from_original_indices(self, original_indices: Union[int, List[int], slice]) -> np.ndarray:
        """
        Get the subset indices corresponding to the given original indices.

        Args:
            original_indices (Union[int, List[int], slice]): The original indices to search for.
                It can be an integer index, a list of indices, or a slice object.

        Returns:
            np.ndarray: The subset indices corresponding to the given original indices.

        Raises:
            ValueError: If one or more original indices are not present in the subset.
        """
        if isinstance(original_indices, int):
            original_indices = [original_indices]
        elif isinstance(original_indices, slice):
            original_indices = self._original_indices[original_indices]
        subset_indices = np.where(np.isin(self._original_indices[self._subset_indices], original_indices))[0]
        if len(subset_indices) != len(original_indices):
            raise ValueError("One or more original indices are not present in the subset.")
        return subset_indices

    def get_original_indices_from_subset_indices(self, subset_indices: Union[int, List[int], slice]) -> np.ndarray:
        """
        Get the original indices corresponding to the given subset indices.

        Args:
            subset_indices (Union[int, List[int], slice]): The subset indices to search for.
                It can be an integer index, a list of indices, or a slice object.

        Returns:
            np.ndarray: The original indices corresponding to the given subset indices.
        """
        if isinstance(subset_indices, list):
            subset_indices = np.array(subset_indices)
        original_indices = self._original_indices[self._subset_indices[subset_indices]]
        return original_indices

    def get_corresponding_indices_from_other_map(self, other: 'AtomSubsetMap') -> np.ndarray:
        """
        Get the subset indices corresponding to the atoms in the other AtomSubsetMap object.

        Args:
            other (AtomSubsetMap): The other AtomSubsetMap object.

        Returns:
            np.ndarray: The subset indices corresponding to the atoms in the other AtomSubsetMap object.
        """
        other_original_indices = other.get_original_indices()
        subset_indices = self.get_subset_indices_from_original_indices(other_original_indices)
        return subset_indices

    def get_corresponding_indices_in_other_map(self, other: 'AtomSubsetMap', subset_indices: Union[int, List[int], slice]) -> np.ndarray:
        """
        Get the subset indices in the other AtomSubsetMap object corresponding to the given subset indices.        

        Args:
            other (AtomSubsetMap): The other AtomSubsetMap object.
            subset_indices (Union[int, List[int], slice]): The subset indices to search for.
                It can be an integer index, a list of indices, or a slice object.

        Returns:
            np.ndarray: The subset indices in the other AtomSubsetMap object corresponding to the given subset indices.
        """
        original_indices = self.get_original_indices_from_subset_indices(subset_indices)
        other_subset_indices = other.get_subset_indices_from_original_indices(original_indices)
        return other_subset_indices

    def _get_sorted_subset_indices(self, selection: Union[List[Union[int, str]], slice, np.ndarray]) -> np.ndarray:
        """
        Get the sorted subset indices based on the given selection.

        Args:
            selection (Union[List[Union[int, str]], slice, np.ndarray]): The selection to create a subset.
                It can be a list of indices or symbols, a slice object, or a NumPy array of indices.

        Returns:
            np.ndarray: The sorted subset indices based on the given selection.
        """
        if isinstance(selection, slice):
            return np.arange(len(self._subset_indices))[selection]
        elif isinstance(selection, np.ndarray) and selection.dtype == int:
            return np.sort(selection)
        else:
            indices = []
            for item in selection:
                if isinstance(item, int):
                    indices.append(item)
                elif isinstance(item, str):
                    symbol_indices = self.get_subset_indices_by_symbol(item)
                    indices.extend(symbol_indices)
                else:
                    raise ValueError(f"Invalid selection item: {item}")
            return np.sort(indices)