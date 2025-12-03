from src.model import Model  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore

IGNORE_TEST = False
IS_PLOT = False

MODEL = """
J1: S1 -> S2 + S3; k1*S1
J2: S2 -> S4; k2*S2
J3: S3 + S4 -> S5; k3*S3*S4

$S1 = 10
S2 = 0
S3 = 0
S4 = 0
S5 = 0
k1 = 0.1
k2 = 0.2
k3 = 0.3
"""

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = Model(MODEL)

    def test_species_names(self):
        species_names = self.model.species_names

        self.assertIn('S1', species_names)
        self.assertIn('S2', species_names)
        self.assertIn('S3', species_names)
        self.assertIn('S4', species_names)
        self.assertIn('S5', species_names)

    def test_reaction_names(self):
        reaction_names = self.model.reaction_names

        self.assertIn('J1', reaction_names)
        self.assertIn('J2', reaction_names)
        self.assertIn('J3', reaction_names)

    def test_reaction_dct(self):
        reaction_dct = self.model.reaction_dct

        self.assertIn('J1', reaction_dct)
        self.assertIn('J2', reaction_dct)
        self.assertIn('J3', reaction_dct)

        reaction_0 = reaction_dct['J1']
        self.assertEqual(reaction_0.name, 'J1')
        self.assertEqual(reaction_0.reactants, ['S1'])
        self.assertEqual(reaction_0.products, ['S2', 'S3'])
        self.assertEqual(reaction_0.kinetic_species, ['S1'])



if __name__ == '__main__':
    unittest.main()