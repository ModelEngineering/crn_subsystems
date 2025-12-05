from src.model import Model  # type: ignore

import unittest
import tellurium as te  # type: ignore

IGNORE_TEST = False
IS_PLOT = False

MODEL = """
model test_model()
J1: S1 -> S2 + S3; k1*S1
J2: S2 -> S4; k2*S2
J3: S3 + S4 -> S5; k3*S3*S4

S1 = 10
S2 = 1
S3 = 1
S4 = 1
S5 = 1
k1 = 0.1
k2 = 0.2
k3 = 0.3
end
"""
SPECIES_NAMES = ["S1", "S2", "S3", "S5"]
REACTION_NAMES = ["J1", "J3"]

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = Model(MODEL, species_names=SPECIES_NAMES,
                reaction_names=REACTION_NAMES)

    def test_species_names(self):
        if IGNORE_TEST:
            return
        species_names = self.model.species_names
        for species_name in SPECIES_NAMES:
            self.assertIn(species_name, species_names)

    def test_reaction_names(self):
        if IGNORE_TEST:
            return
        reaction_names = self.model.reaction_names
        for reaction_name in REACTION_NAMES:
            self.assertIn(reaction_name, reaction_names)

    def test_reaction_dct(self):
        if IGNORE_TEST:
            return
        reaction_dct = self.model.reaction_dct
        #
        for reaction_name in REACTION_NAMES:
            self.assertIn(reaction_name, reaction_dct)
        #
        reaction_0 = reaction_dct['J1']
        self.assertEqual(reaction_0.name, 'J1')
        self.assertEqual(reaction_0.reactants, ['S1'])
        self.assertEqual(reaction_0.products, ['S2', 'S3'])
        self.assertEqual(reaction_0.kinetic_species, ['S1'])

    def testMakeConstrainedSBMLModel(self):
        if IGNORE_TEST:
            return
        antimony_str = self.model.makeAntimony()
        self.assertFalse("J4" in antimony_str)
        self.assertTrue("$S4" in antimony_str)
        rr = te.loada(antimony_str)
        _ = rr.simulate()



if __name__ == '__main__':
    unittest.main()