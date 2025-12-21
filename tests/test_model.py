from src.model import Model  # type: ignore
from src.lti_crn import LtiCrn  # type: ignore

import numpy as np
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
        
    def test_parameter_dct(self):
        if IGNORE_TEST:
            return
        parameter_dct = self.model.parameter_dct
        self.assertIn("k1", parameter_dct)
        self.assertIn("k3", parameter_dct)
        self.assertIn("k2", parameter_dct.keys())
        self.assertEqual(parameter_dct["k1"], 0.1)
        self.assertEqual(parameter_dct["k3"], 0.3)
    
    def testMakeKineticConstantDct(self):
        if IGNORE_TEST:
            return
        kinetic_constant_dct = self.model.kinetic_constant_dct
        self.assertIsInstance(kinetic_constant_dct, dict)
        self.assertEqual(len(kinetic_constant_dct), 3)
        self.assertIn("k1", kinetic_constant_dct)
        self.assertIn("k2", kinetic_constant_dct)
        self.assertIn("k3", kinetic_constant_dct)
        trues = [kinetic_constant_dct["k1"] == 0.1,
                kinetic_constant_dct["k2"] == 0.2,
                kinetic_constant_dct["k3"] == 0.3]
        self.assertTrue(all(trues))

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

    def testMakeConstrainedSBMLModel2(self):
        #if IGNORE_TEST:
        #    return
        for _ in range(5):
            num_species = 6
            num_reaction = 8
            antimony_str = LtiCrn(num_species=num_species, num_reaction=num_reaction, seed=None).antimony_str
            original_model = Model(antimony_str)
            excluded_species_names = [f"S{n+1}_" for n in np.random.randint(num_species, size=2)]
            species_names = list(set(original_model.species_names) - set(excluded_species_names))
            excluded_reaction_names = [f"_J{n+1}" for n in np.random.randint(num_reaction, size=2)]
            reaction_names = list(set(original_model.reaction_names) - set(excluded_reaction_names))
            model = Model(antimony_str, species_names=species_names, reaction_names=reaction_names)
            # Ensure valid model
            antimony_str = model.makeAntimony()
            rr = te.loada(antimony_str)
            _ = rr.simulate()
            # Check that excludes are absent
            for species_name in excluded_species_names:
                self.assertFalse(species_name in model.species_names)
            for reaction_name in excluded_reaction_names:
                self.assertFalse(reaction_name in model.reaction_names)


if __name__ == '__main__':
    unittest.main()