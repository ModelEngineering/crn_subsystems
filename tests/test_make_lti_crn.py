from src.make_lti_crn import makeLtiCrn  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore

IGNORE_TEST = False
IS_PLOT = False

class TestFunction(unittest.TestCase):

    def setUp(self):
        pass

    def testBasicJacobian(self):
        if IGNORE_TEST:
            return
        for _ in range(5):
            num_species = np.random.randint(5, 10)
            max_num_reaction = np.random.randint(5, 10)
            max_num_product = np.random.randint(2, 10)
            max_kinetic_constant = np.random.uniform(1, 10)
            max_stoichiometry = np.random.randint(5, 10)
            try:
                model = makeLtiCrn(
                    num_species,
                    max_num_reaction,
                    num_products_bounds=(1, max_num_product),
                    kinetic_constant_bounds=(0, max_kinetic_constant),
                    stoichiometry_bounds=(1, max_stoichiometry),
                    seed=42  # For reproducibility
                )
            except:
                self.fail("generateCrn raised an exception unexpectedly!")
            try:
                rr = te.loada(model)  # type: ignore
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
            model = makeLtiCrn(num_species=10,
                    num_reaction=10,
                    num_products_bounds=(1, 5),
                    kinetic_constant_bounds= (0.1, 1),
                    stoichiometry_bounds=(1, 3))
            rr = te.loada(model)  # type: ignore
            jacobian_arr = rr.getFullJacobian()
            eigenvalues = np.linalg.eigvals(jacobian_arr)
            for eig in eigenvalues:
                if eig.real > 0:
                    import pdb; pdb.set_trace()
                    pass


if __name__ == '__main__':
    unittest.main()