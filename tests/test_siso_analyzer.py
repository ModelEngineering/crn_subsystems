from src.siso_analyzer import SISOAnalyzer  # type: ignore
from src.make_lti_crn import makeLtiCrn  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
from scipy import signal  # type: ignore
import numpy as np
import unittest
import tellurium as te  # type: ignore
import pandas as pd  # type: ignore
import sympy as sp  # type: ignore
from typing import Optional

IGNORE_TEST = False
IS_PLOT = False

MODEL = """
S1_ -> S2_; k2*S1_
S2_ -> S3_; k3*S2_
S3_ -> ; k4*S3_
$S1_ = 1
S2_ = 0
S3_ = 0
k1 = 1
k2 = 1
k3 = 3
k4 = 4
"""

MODEL2 = """
-> $S1_; k1
S1_ -> 2 S2_ + S3; k2*S1_
S2_ -> 3 S3_; k3*S2_
S3_ -> ; k4*S3_
S3_ -> S4_ + S1_; k5*S3_
S1_ = 1
S2_ = 0
S3_ = 0
k1 = 1
k2 = 1
k3 = 3
k4 = 4
k5 = 5
"""


class TestMakeSequentialAntimony(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.num_stage = 3
        self.builder = SISOAnalyzer(MODEL)

    def testBasicStructure(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        self.assertIsInstance(antimony_str, str)
        self.assertIn("model *sequential_network()", antimony_str)
        self.assertIn("end", antimony_str)

    def testValidAntimony(self):
        if IGNORE_TEST:
            return
        # Test that the generated Antimony can be loaded by Tellurium
        antimony_str = self.builder.makeSequentialAntimony(3)
        try:
            rr = te.loada(antimony_str)
            self.assertIsNotNone(rr)
        except Exception as e:
            self.fail(f"Generated Antimony is not valid: {e}")

    def testSingleStage(self):
        if IGNORE_TEST:
            return
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(1)
        
        # Should have 2 reactions: S0 -> S1_ and S0 -> ;
        self.assertIn("S0 -> S1", antimony_str)
        self.assertIn("S0 -> ;", antimony_str)
        
        # Should have 2 kinetic constants: k1 and k2
        self.assertIn("k1 = 1", antimony_str)
        self.assertIn("k2 = 2", antimony_str)
        
        # Should have 2 species: S0 and S1_
        self.assertIn("S0 = 0", antimony_str)
        self.assertIn("S1 = 0", antimony_str)

    def testThreeStages(self):
        if IGNORE_TEST:
            return
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(3)
        
        # Check reactions
        self.assertIn("S0 -> S1; k1*S0", antimony_str)
        self.assertIn("S0 -> ; k2*S0", antimony_str)
        self.assertIn("S1 -> S2; k3*S1", antimony_str)
        self.assertIn("S1 -> ; k4*S1", antimony_str)
        self.assertIn("S2 -> S3; k5*S2", antimony_str)
        self.assertIn("S2 -> ; k6*S2", antimony_str)
        
        # Check kinetic constants
        for i in range(1, 7):
            self.assertIn(f"k{i} = {i}", antimony_str)
        
        # Check species initialization
        for i in range(4):
            self.assertIn(f"S{i} = 0", antimony_str)

    def testKineticConstants(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        
        # Load model and verify kinetic constant values
        rr = te.loada(antimony_str)
        
        # For 3 stages, we should have 6 kinetic constants (k1 through k6)
        expected_num_constants = 2 * self.num_stage
        params = rr.getGlobalParameterIds()
        self.assertEqual(len(params), expected_num_constants)
        
        # Verify each constant has the correct value
        for i in range(1, expected_num_constants + 1):
            param_name = f"k{i}"
            self.assertIn(param_name, params)
            value = rr.getValue(param_name)
            self.assertEqual(value, i, f"Expected {param_name} = {i}, got {value}")

    def testSpeciesCount(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        rr = te.loada(antimony_str)
        
        # Should have num_stage + 1 species (S0, S1_, ..., S_num_stage)
        species = rr.getFloatingSpeciesIds()
        self.assertEqual(len(species), self.num_stage + 1)
        
        # Verify species names
        for i in range(self.num_stage + 1):
            expected_species = f"S{i}"
            self.assertIn(expected_species, species)

    def testReactionCount(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        rr = te.loada(antimony_str)
        
        # Should have 2 * num_stage reactions
        # (one forward, one degradation per stage)
        num_reactions = rr.getNumReactions()
        self.assertEqual(num_reactions, 2 * self.num_stage)

    def testVariousStages(self):
        if IGNORE_TEST:
            return
        # Test that various numbers of stages work correctly
        for num_stage in [1, 2, 5, 10]:
            builder = SISOAnalyzer(MODEL)
            antimony_str = builder.makeSequentialAntimony(num_stage)
            
            # Verify it's valid
            rr = te.loada(antimony_str)
            
            # Verify species count
            species = rr.getFloatingSpeciesIds()
            self.assertEqual(len(species), num_stage + 1)
            
            # Verify parameter count
            params = rr.getGlobalParameterIds()
            self.assertEqual(len(params), 2 * num_stage)
            
            # Verify reaction count
            self.assertEqual(rr.getNumReactions(), 2 * num_stage)

    def testSequentialKineticValues(self):
        if IGNORE_TEST:
            return
        # Ensure kinetic constants are truly sequential (1, 2, 3, ...)
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(5)
        rr = te.loada(antimony_str)
        
        for i in range(1, 11):  # 5 stages = 10 kinetic constants
            value = rr.getValue(f"k{i}")
            self.assertEqual(value, i)

    def testModelStructure(self):
        if IGNORE_TEST:
            return
        # Test that the model has the expected structure
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(2)

        # Split into lines for detailed checking
        lines = [line.strip() for line in antimony_str.split('\n') if line.strip()]
        
        # First line should be model declaration
        self.assertEqual(lines[0], "model *sequential_network()")
        
        # Last line should be end
        self.assertEqual(lines[-1], "end")
        
        # Should contain reaction lines
        reaction_lines = [line for line in lines if '->' in line]
        self.assertEqual(len(reaction_lines), 4)  # 2 stages * 2 reactions each

    def testTransferFunction1a(self):
        if IGNORE_TEST:
            return
        analyzer = SISOAnalyzer(MODEL2, output_name="S4_")
        if IS_PLOT:
            analyzer.plotTransferFunctionValidation()

    def testTransferFunction1(self):
        if IGNORE_TEST:
            return
        analyzer = SISOAnalyzer(MODEL)
        k2, k3, k4, s = sp.symbols('k2 k3 k4 s')
        expected_tf_expr = 1.0*k2*k3/(k3*k4 + k3*s + k4*s + s**2)
        tf_expr = analyzer.transfer_function_expr
        self.assertEqual(sp.simplify(tf_expr - expected_tf_expr), 0)
        if IS_PLOT:
            analyzer.plotTransferFunctionValidation()

    def testTransferFunction2(self):
        if IGNORE_TEST:
            return
        model = """
        -> $S1_; k1
        S2_ -> 2 S4_ + 2 S5_ + S5_ + 3 S3_; k2 * S2_
        S1_ -> 2 S5_ + 2 S3_; k3 * S1_
        S3_ -> S4_ + 2 S3_; k4 * S3_
        S1_ -> 3 S5_ + 3 S5_ + S3_ + S4_ + 2 S3_; k5 * S1_
        S3_ -> 3 S4_ + 2 S4_ + S3_ + 3 S4_; k6 * S3_
        S3_ -> S5_ + 2 S4_ + S5_ + 2 S4_; k7 * S3_
        S2_ -> 2 S3_; k8 * S2_
        S5_ -> 3 S4_; k9 * S5_
        S2_ -> 2 S5_ + 2 S5_ + 2 S5_ + 3 S5_; k10 * S2_
        # Rate constants
        k1 = 1
        k2 = 0.2631
        k3 = 0.8447
        k4 = 0.9716
        k5 = 0.1637
        k6 = 0.8069
        k7 = 0.5813
        k8 = 0.4093
        k9 = 0.6556
        k10 = 0.3726
        # Species initialization
        S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0
        # Degradation reactions
        S4_ -> ; kd_1 * S4_
        kd_1 = 13.4695
        S5_ -> ; kd_2 * S5_
        kd_2 = 5.1147
        S3_ -> ; kd_3 * S3_
        kd_3 = 2.1980
        """
        model1 = """
        species S1_, S2_, S3_, S4_, S5_
        -> $S1_; k1
        S1_ -> S3_; k2 * S1_
        S3_ -> 3 S3_; k4 * S3_
        # Rate constants
        k1 = 1
        k2 = 0.2631
        k3 = 0.8447
        k4 = 0.9716
        k5 = 0.1637
        k6 = 0.8069
        k7 = 0.5813
        k8 = 0.4093
        k9 = 0.6556
        k10 = 0.3726
        # Species initialization
        S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0
        # Degradation reactions
        S4_ -> ; kd_1 * S4_
        kd_1 = 13.4695
        S5_ -> ; kd_2 * S5_
        kd_2 = 5.1147
        S3_ -> ; kd_3 * S3_
        kd_3 = 2.1980
        """

        model = """
        -> $S1_; k1
        S4_ -> 2 S3_ + 3 S5_ + S3_; k2 * S4_
        S1_ -> S4_ + S5_; k3 * S1_
        S5_ -> 3 S4_ + 2 S4_ + 2 S5_ + 3 S3_ + 3 S4_; k4 * S5_
        S1_ -> 3 S3_ + 3 S3_ + S3_ + 2 S5_; k5 * S1_
        S2_ -> S5_ + 2 S3_ + S4_ + 3 S4_ + 3 S4_; k6 * S2_
        S1_ -> 2 S5_ + S5_ + 2 S4_ + 3 S3_ + S3_; k7 * S1_
        S5_ -> 3 S4_ + 2 S3_ + S4_; k8 * S5_
        S2_ -> S3_ + 2 S4_ + 3 S3_; k9 * S2_
        S4_ -> 2 S4_; k10 * S4_

        # Rate constants
        k1 = 0.2175
        k2 = 0.1769
        k3 = 0.4090
        k4 = 0.6329
        k5 = 0.9204
        k6 = 0.2265
        k7 = 0.8298
        k8 = 0.5267
        k9 = 0.2583
        k10 = 0.5946

        # Species initialization
        S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0


        # Degradation reactions
        S4_ -> ; kd_0 * S4_
        kd_0 = 10.6588
        S3_ -> ; kd_1 * S3_
        kd_1 = 5.4659
        S5_ -> ; kd_2 * S5_
        kd_2 = 0.9497
        """
        analyzer = SISOAnalyzer(model, output_name="S3_")
        transfer_function = analyzer.makeTransferFunction()
        self.assertIsNotNone(transfer_function)
        if IS_PLOT:
            analyzer.plotTransferFunctionValidation()

    def testBug2(self):
        if IGNORE_TEST:
            return
        model = """
        model random_crn()

        S5_ -> 2 S4_ + 3 S4_ + S4_ + 2 S4_ + 3 S4_; k2 * S5_
        S1_ -> 3 S4_ + 2 S5_ + 3 S5_ + S3_ + S3_; k3 * S1_
        S2_ -> 3 S4_ + S3_ + 2 S3_ + S4_ + 3 S5_; k4 * S2_
        S1_ -> S5_; k5 * S1_
        S2_ -> 3 S3_ + 2 S3_ + 2 S3_ + S5_ + S5_; k6 * S2_
        S1_ -> S4_ + 2 S5_ + 2 S4_ + S3_ + S5_; k7 * S1_
        S2_ -> S4_ + 2 S7_ + S3_ + 3 S6_; k8 * S2_
        S6_ -> ; k9 * S6_
        S7_ -> ; k9 * S7_
        S2_ -> S4_ + 3 S3_; k9 * S2_
        S1_ -> S5_ + 2 S4_ + S4_; k10 * S1_

        # Rate constants
        k1 = 0.9742
        k2 = 0.7012
        k3 = 0.1698
        k4 = 0.4629
        k5 = 0.5307
        k6 = 0.4652
        k7 = 0.8688
        k8 = 0.8631
        k9 = 0.7551
        k10 = 0.1057

        # Species initialization
        $S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0


        # Degradation reactions
        S5_ -> ; kd_0 * S5_
        kd_0 = 3.6785
        S4_ -> ; kd_1 * S4_
        kd_1 = 12.3013
        S3_ -> ; kd_2 * S3_
        kd_2 = 11.3991
        end
        """
        analyzer = SISOAnalyzer(model, output_name="S5_")
        # Make sure that the Jacobian is computed correctly
        jacobian_df = analyzer.jacobian_df
        self.assertIsNotNone(jacobian_df)
        self.assertEqual(jacobian_df.shape[0], jacobian_df.shape[1]) 
        #tf1 = analyzer.transfer_function_expr
        tf1 = analyzer.makeTransferFunction()
        self.assertGreater(tf1.num[-1]/tf1.den[-1], 0) # type: ignore

    def testBug3(self):
        if IGNORE_TEST:
            return
        model = """
        species S1_;

        S3_ -> S5_ + 2 S3_ + 3 S5_ + 3 S4_ + 3 S5_; k2 * S3_
        S5_ -> 3 S4_ + 3 S3_ + 3 S5_ + 3 S3_ + S3_; k3 * S5_
        S1_ -> S4_ + 2 S5_ + 2 S3_; k4 * S1_
        S5_ -> S3_ + 2 S5_; k5 * S5_
        S2_ -> 3 S4_ + 2 S3_ + 3 S4_ + 3 S4_ + 3 S5_; k6 * S2_
        S2_ -> 3 S3_; k7 * S2_
        S1_ -> 3 S4_ + S4_ + 2 S4_; k8 * S1_
        S2_ -> S5_ + 2 S4_ + S4_ + 2 S3_ + 3 S5_; k9 * S2_
        S5_ -> 3 S3_ + S3_ + 3 S5_; k10 * S5_

        # Rate constants
        k2 = 0.8954
        k3 = 0.5287
        k4 = 0.9260
        k5 = 0.5231
        k6 = 0.3714
        k7 = 0.4442
        k8 = 0.2517
        k9 = 0.7125
        k10 = 0.8282

        # Species initialization
        $S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0


        # Degradation reactions
        S3_ -> ; kd_0 * S3_
        kd_0 = 13.1259
        S5_ -> ; kd_1 * S5_
        kd_1 = 14.8158
        S4_ -> ; kd_2 * S4_
        kd_2 = 10.7276
        """
        analyzer = SISOAnalyzer(model, output_name="S5_")
        tf1 = analyzer.makeTransferFunction()
        self.assertGreater(tf1.num[-1]/tf1.den[-1], 0) # type: ignore
        if IS_PLOT:
            analyzer.plotTransferFunctionValidation()

    def compareStepResponse(self, model: str)-> bool:
        # Return False if couldn't do the comparison
        analyzer = SISOAnalyzer(model)
        try:
            analyzer.original_roadrunner.steadyState()
        except Exception:
            return False
        simulated_output = analyzer.original_roadrunner[analyzer.output_name]
        #
        step_size = 1.0
        step_response = analyzer.calculateStepResponse(step_size)
        self.assertIsInstance(step_response, float)
        self.assertAlmostEqual(simulated_output, step_response, places=5)
        return True

    def testCalculateStepResponseSmall(self):
        if IGNORE_TEST:
            return
        self.compareStepResponse(MODEL)

    def testCalculateStepResponseBig(self):
        if IGNORE_TEST:
            return
        for _ in range(5):
            model = makeLtiCrn(num_species=10,
                    num_reaction=100,
                    num_products_bounds=(1, 5),
                    kinetic_constant_bounds= (0.1, 1),
                    stoichiometry_bounds=(1, 3))
            if IS_PLOT and not self.compareStepResponse(model):
                print("Could not compare step response for generated model")

    def testCalculateEigenvalues(self):
        if IGNORE_TEST:
            return
        analyzer = SISOAnalyzer(MODEL)
        eigenvalueIds = analyzer.roadrunner.getEigenValueIds()
        rr_eigenvalues = np.array(
                [analyzer.roadrunner.getValue(eid) for eid in eigenvalueIds
                if "eigen(" in eid])
        rr_eigenvalues = -np.sort(-rr_eigenvalues)
        eigenvalues = analyzer.calculateEigenvalues()
        self.assertIsInstance(eigenvalues, np.ndarray)
        self.assertEqual(len(eigenvalues), 3)
        self.assertTrue(np.all(np.diff(eigenvalues) <= 0))  # Check descending order
        self.assertTrue(np.allclose(eigenvalues, rr_eigenvalues))


if __name__ == '__main__':
    unittest.main()
