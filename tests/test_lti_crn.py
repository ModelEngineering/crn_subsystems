from src.lti_crn import LtiCrn  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore

IGNORE_TEST = False
IS_PLOT = False

class TestLtiCrn(unittest.TestCase):

    def setUp(self):
        pass

    def makeModel(self)->None:
        self.num_species = np.random.randint(5, 10)
        self.max_num_reaction = np.random.randint(5, 10)
        self.max_num_product = np.random.randint(2, 10)
        self.max_kinetic_constant = np.random.uniform(1, 10)
        self.max_stoichiometry = np.random.randint(5, 10)
        self.model = LtiCrn(
                    self.num_species,
                    self.max_num_reaction,
                    num_products_bounds=(1, self.max_num_product),
                    kinetic_constant_bounds=(0, self.max_kinetic_constant),
                    stoichiometry_bounds=(1, self.max_stoichiometry),
                    seed=42  # For reproducibility
                )

    def testBasicJacobian(self):
        if IGNORE_TEST:
            return
        for _ in range(5):
            try:
                self.makeModel()
            except:
                self.fail("generateCrn raised an exception unexpectedly!")
            try:
                rr = te.loada(self.model.antimony_str)  # type: ignore
                self.assertIsNotNone(rr)
            except Exception as e:
                self.fail(e)
            except:
                self.fail("generateCrn raised an exception unexpectedly!")
            try:
                rr = te.loada(self.model.antimony_str)  # type: ignore
                self.assertIsNotNone(rr)
            except Exception as e:
                self.fail(f"Generated Antimony is not valid: {e}")
            try:
                data = rr.simulate(0, 10, 100)
                self.assertTrue(data.shape[0] > 0)
            except Exception as e:
                self.fail(f"Could not simulate the generated model: {e}")

    def testStableJacobian(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            self.makeModel()
            rr = te.loada(self.model.antimony_str)  # type: ignore
            jacobian_arr = rr.getFullJacobian()
            eigenvalues = np.linalg.eigvals(jacobian_arr)
            for eig in eigenvalues:
                if eig.real > 0:
                    import pdb; pdb.set_trace()
                    pass
            rr = te.loada(self.model.antimony_str)  # type: ignore
            jacobian_arr = rr.getFullJacobian()
            eigenvalues = np.linalg.eigvals(jacobian_arr)
            for eig in eigenvalues:
                if eig.real > 0:
                    import pdb; pdb.set_trace()
                    pass
    
    def testMultipleInputs(self):
        if IGNORE_TEST:
            return
        NUM_SPECIES = 10
        for _ in range(10):
            input_species_indices = np.random.choice(range(1, NUM_SPECIES+1),
                    size=np.random.randint(1,4), replace=False).tolist()
            lti_crn = LtiCrn(num_species=NUM_SPECIES,
                    num_reaction=10,
                    num_products_bounds=(1, 5),
                    kinetic_constant_bounds= (0.1, 1),
                    stoichiometry_bounds=(1, 3),
                    input_species_indices=input_species_indices)
            model = lti_crn.antimony_str
            for input_idx in input_species_indices:
                input_species_name = f"S{input_idx}_"
                self.assertIn(f"${input_species_name}", model)
    
    def testStartSpeciesIndex(self):
        if IGNORE_TEST:
            return
        NUM_SPECIES = 10
        input_species_indices = np.random.choice(range(1, NUM_SPECIES+1),
                size=np.random.randint(1,4), replace=False).tolist()
        lti_crn= LtiCrn(num_species=NUM_SPECIES,
                num_reaction=10,
                num_products_bounds=(1, 5),
                kinetic_constant_bounds= (0.1, 1),
                stoichiometry_bounds=(1, 3),
                input_species_indices=input_species_indices,
                starting_species_index=50)
        model = lti_crn.antimony_str
        for idx in range(NUM_SPECIES):
            species_name = f"S{idx+50}_"
            self.assertIn(species_name, model)


if __name__ == '__main__':
    unittest.main()