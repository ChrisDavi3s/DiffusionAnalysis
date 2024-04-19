# tests/test_atom_subset_map.py
import unittest
import numpy as np
from ase import Atoms
from DiffusionAnalysis.utils.atom_subset_map import AtomSubsetMap

class TestAtomSubsetMap(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.original_atoms = Atoms('H2O')
        self.original_atoms.extend(Atoms('CH4'))
        self.subset_map = AtomSubsetMap(self.original_atoms)

    def test_initialization(self):
        self.assertEqual(len(self.subset_map), len(self.original_atoms))
        np.testing.assert_array_equal(self.subset_map.get_subset_indices(), np.arange(len(self.original_atoms)))

    def test_get_atomic_numbers(self):
        np.testing.assert_array_equal(self.subset_map.get_subset_atomic_numbers(), [1, 1, 8, 6, 1, 1, 1, 1])

    def test_get_chemical_symbols(self):
        self.assertEqual(self.subset_map.get_subset_chemical_symbols(), ['H', 'H', 'O', 'C', 'H', 'H', 'H', 'H'])

    def test_get_indices_by_atomic_number(self):
        np.testing.assert_array_equal(self.subset_map.get_subset_indices_by_atomic_number(1), [0, 1, 4, 5, 6, 7])
        np.testing.assert_array_equal(self.subset_map.get_subset_indices_by_atomic_number(8), [2])
        np.testing.assert_array_equal(self.subset_map.get_subset_indices_by_atomic_number(6), [3])

    def test_get_indices_by_symbol(self):
        np.testing.assert_array_equal(self.subset_map.get_subset_indices_by_symbol('H'), [0, 1, 4, 5, 6, 7])
        np.testing.assert_array_equal(self.subset_map.get_subset_indices_by_symbol('O'), [2])
        np.testing.assert_array_equal(self.subset_map.get_subset_indices_by_symbol('C'), [3])
        with self.assertRaises(ValueError):
            self.subset_map.get_subset_indices_by_symbol('N')

    def test_get_indices_by_symbol_with_selection(self):
        new_map = self.subset_map.select(['H'])
        np.testing.assert_array_equal(new_map.get_subset_indices_by_symbol('H'), [0, 1, 2, 3, 4, 5])

    def test_create_subset(self):
        subset = self.subset_map.select([0, 2, 4])
        self.assertEqual(len(subset), 3)
        np.testing.assert_array_equal(subset.get_subset_indices(), [0, 2, 4])

    def test_indexing(self):
        subset = self.subset_map[1:]
        self.assertEqual(len(subset), 7)
        np.testing.assert_array_equal(subset.get_subset_indices(), [ 1, 2, 3, 4, 5, 6,7])

    def test_subset_of_subset(self):
        subset1 = self.subset_map.select([0, 2, 4])
        subset2 = subset1.select([1, 2])
        self.assertEqual(len(subset2), 2)
        np.testing.assert_array_equal(subset2.get_subset_indices(), [1, 2])

    def test_subset_of_subset_by_symbol(self):
        subset1 = self.subset_map.select([0, 2, 4])
        subset2 = subset1.select(['H'])
        self.assertEqual(len(subset2), 2)
        np.testing.assert_array_equal(subset2.get_subset_indices(), [0, 2])

    def test_get_original_indices_with_selection(self):
        subset = self.subset_map.select([0, 2, 4])
        np.testing.assert_array_equal(subset.get_original_indices(), [0, 2, 4])

    def test_atoms_property(self):
        subset = self.subset_map.select([0, 2, 4])
        subset_atoms = subset.atoms
        self.assertIsInstance(subset_atoms, Atoms)
        self.assertEqual(len(subset_atoms), 3)
        np.testing.assert_array_equal(subset_atoms.get_atomic_numbers(), [1, 8, 1])

    def test_indexing_by_symbol(self):
        subset = self.subset_map[['H', 'C']]
        self.assertEqual(len(subset), 7)
        np.testing.assert_array_equal(subset.get_subset_indices(), [0, 1, 3, 4, 5, 6, 7])

    def test_get_indices_from_original(self):
        subset = self.subset_map.select([0, 2, 4])
        np.testing.assert_array_equal(subset.get_subset_indices_from_original_indices([0, 2, 4]), [0, 1, 2])
        np.testing.assert_array_equal(subset.get_subset_indices_from_original_indices(slice(0, 5, 2)), [0, 1, 2])

    def test_get_original_indices_from_indices(self):
        subset = self.subset_map.select([0, 2, 4])
        #print(np.array([0,1,4])[slice(1,3)])
        np.testing.assert_array_equal(subset.get_original_indices_from_subset_indices([1, 2]), [2, 4])
        np.testing.assert_array_equal(subset.get_original_indices_from_subset_indices(slice(1, 3)), [2, 4])

    def test_get_indices_from_other(self):
        subset1 = self.subset_map.select([0, 2, 4, 6])
        subset2 = self.subset_map.select([2, 4, 6])
        np.testing.assert_array_equal(subset1.get_corresponding_indices_from_other_map(subset2), [1, 2, 3])

    def test_get_other_indices_from_indices(self):
        subset1 = self.subset_map.select([0, 2, 4])
        subset2 = self.subset_map.select([2, 4, 6])
        np.testing.assert_array_equal(subset1.get_corresponding_indices_in_other_map(subset2, [1, 2]), [0, 1])

if __name__ == '__main__':
    unittest.main()