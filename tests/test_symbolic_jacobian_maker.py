from src.symbolic_jacobian_maker import SymbolicJacobianMaker
from src.model import Model  # type: ignore

import libsbml # type: ignore
import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict, Optional

IGNORE_TEST = False

MODEL1 = """
J1: -> S1; k1
J2: S2 -> 2 S3 + S1; k2*S2
J3: S2 -> ; k3*S2
J4: S3 -> ; k4*S2
J5: S3 -> 3 S2 + S1; k2*S3
S1 = 10
S2 = 0
S3 = 0
k1 = 1
k2 = 2
k3 = 3
k4 = 4
"""
MODEL2 = """
        model random_crn()

        J1: -> $S1_; k1
        J2: S5_ -> 2 S4_ + 3 S4_ + S4_ + 2 S4_ + 3 S4_; k2 * S5_
        J3: S1_ -> 3 S4_ + 2 S5_ + 3 S5_ + S3_ + S3_; k3 * S1_
        J4: S2_ -> 3 S4_ + S3_ + 2 S3_ + S4_ + 3 S5_; k4 * S2_
        J5: S1_ -> S5_; k5 * S1_
        J6: S2_ -> 3 S3_ + 2 S3_ + 2 S3_ + S5_ + S5_; k6 * S2_
        J7: S1_ -> S4_ + 2 S5_ + 2 S4_ + S3_ + S5_; k7 * S1_
        J8: S2_ -> S4_ + 2 S5_ + S3_ + 3 S3_; k8 * S2_
        J9: S2_ -> S4_ + 3 S3_; k9 * S2_
        J10: S1_ -> S5_ + 2 S4_ + S4_; k10 * S1_

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
        S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0


        # Degradation reactions
        JD1: S5_ -> ; kd_0 * S5_
        kd_0 = 3.6785
        JD2: S4_ -> ; kd_1 * S4_
        kd_1 = 12.3013
        JD3: S3_ -> ; kd_2 * S3_
        kd_2 = 11.3991
        end
""" 


MODEL3 = """
J1: -> S1 + 2 S2; k1
J2: S2 + 2 S1 -> 2 S3 + S1; k2*S2*S1*S1
J3: S2 -> ; k3*S2
J4: S3 -> ; k4*S2
J5: S3 -> 3 S2 + S1; k2*S3
S1 = 10
S2 = 0
S3 = 0
k1 = 1
k2 = 2
k3 = 3
k4 = 4
"""

class TestSymbolicJacobianMaker(unittest.TestCase):

    def setUp(self):
        self.model = Model(MODEL1)
        self.maker = SymbolicJacobianMaker(self.model)
        if not IGNORE_TEST:
            self.maker.initialize()

    def getReactions(self, maker: Optional[SymbolicJacobianMaker]=None)->Dict[str, libsbml.Reaction]:
        # Gets the reactions from the model
        if maker is None:
            maker = self.maker
        dct = {}
        for i in range(maker.model.libsbml_model.getNumReactions()):
            reaction = maker.model.libsbml_model.getReaction(i)
            reaction_id = reaction.getId()
            dct[reaction_id] = maker.model.libsbml_model.getReaction(i)
        return dct

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsInstance(self.maker, SymbolicJacobianMaker)

    def testMakeModel(self):
        if IGNORE_TEST:
            return
        self.assertIsNotNone(self.maker.model)
        self.assertEqual(self.maker.model.libsbml_model.getNumSpecies(), 3)
        self.assertEqual(self.maker.model.libsbml_model.getNumReactions(), 5)

    def testMakeKineticConstantDct(self):
        if IGNORE_TEST:
            return
        kinetic_constant_dct = self.maker._makeKineticConstantDct()
        self.assertIsInstance(kinetic_constant_dct, dict)
        self.assertEqual(len(kinetic_constant_dct), 4)
        self.assertIn("k1", kinetic_constant_dct)
        self.assertIn("k2", kinetic_constant_dct)
        self.assertIn("k3", kinetic_constant_dct)
        trues = [kinetic_constant_dct["k1"] == 1.0,
                kinetic_constant_dct["k2"] == 2.0,
                kinetic_constant_dct["k3"] == 3.0]
        self.assertTrue(all(trues))

    def testMakeReactionSymbolicJacobianJ1(self):
        # J1: -> S1; k1
        if IGNORE_TEST:
            return
        reaction_dct = self.getReactions()
        reaction = reaction_dct["J1"]
        result_j1 = self.maker._makeReactionLti(reaction)
        expected_A_smat = sp.Matrix([[0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]])
        expected_b_smat = sp.Matrix([[sp.Symbol("k1")],
                                        [0],
                                        [0]])
        self.assertTrue(result_j1.A_smat.equals(expected_A_smat))
        self.assertTrue(result_j1.b_smat.equals(expected_b_smat))

    def testMakeReactionSymbolicJacobianJ2(self):
        # J2: S2 -> 2 S3 + S1; k2*S2
        if IGNORE_TEST:
            return
        reaction_dct = self.getReactions()
        reaction = reaction_dct["J2"]
        result_j2 = self.maker._makeReactionLti(reaction)
        expected_A_smat = sp.Matrix([[0, sp.Symbol("k2"), 0],
                                            [0, -sp.Symbol("k2"), 0],
                                            [0, 2*sp.Symbol("k2"), 0]])  # type: ignore
        expected_b_smat = sp.Matrix([[0],
                                        [0],
                                        [0]])
        self.assertTrue(result_j2.A_smat.equals(expected_A_smat))
        self.assertTrue(result_j2.b_smat.equals(expected_b_smat))

    def testMakeReactionSymbolicJacobianJ3(self):
        # J3: S2 -> ; k3*S2
        if IGNORE_TEST:
            return
        reaction_dct = self.getReactions()
        reaction = reaction_dct["J3"]
        result_j3 = self.maker._makeReactionLti(reaction)
        expected_A_smat = sp.Matrix([[0, 0, 0],
                                            [0, -sp.Symbol("k3"), 0],
                                            [0, 0, 0]])
        expected_b_smat = sp.Matrix([[0],
                                        [0],
                                        [0]])
        self.assertTrue(result_j3.A_smat.equals(expected_A_smat))
        self.assertTrue(result_j3.b_smat.equals(expected_b_smat))

    def testMakeReactionSymbolicJacobianJ4(self):
        # J4: S3 -> ; k4*S2
        if IGNORE_TEST:
            return
        reaction_dct = self.getReactions()
        reaction = reaction_dct["J4"]
        result_j4 = self.maker._makeReactionLti(reaction)
        expected_A_smat = sp.Matrix([[0, 0, 0],
                                            [0, 0, 0],
                                            [0, -sp.Symbol("k4"), 0]])
        expected_b_smat = sp.Matrix([[0],
                                        [0],
                                        [0]])
        self.assertTrue(result_j4.A_smat.equals(expected_A_smat))
        self.assertTrue(result_j4.b_smat.equals(expected_b_smat))
    
    def testMakeReactionSymbolicJacobianJ5(self):
        # J5: S3 -> 3 S2 + S1; k2*S3
        if IGNORE_TEST:
            return
        reaction_dct = self.getReactions()
        reaction = reaction_dct["J5"]
        result_j5 = self.maker._makeReactionLti(reaction)
        expected_A_smat = sp.Matrix([[0, 0, sp.Symbol("k2")],
                                            [0, 0, 3*sp.Symbol("k2")],  # type: ignore
                                            [0, 0, -sp.Symbol("k2")]])
        expected_b_smat = sp.Matrix([[0],
                                        [0],
                                        [0]])
        self.assertTrue(result_j5.A_smat.equals(expected_A_smat))
        self.assertTrue(result_j5.b_smat.equals(expected_b_smat))

    def checkSymbolicJacobian(self, model: Model, maker: SymbolicJacobianMaker):
        jacobian_smat = maker._makeSymbolicJacobian()[0]
        jacobian_mat = jacobian_smat.subs(maker.kinetic_constant_dct)
        expected_jacobian_mat = maker.jacobian_mat
        expected_species_names = expected_jacobian_mat.colnames
        for row_idx in range(maker.num_species):
            row_species_name = model.species_names[row_idx]
            if not row_species_name in expected_species_names:
                continue
            expected_row_idx = expected_species_names.index(row_species_name)
            for col_idx in range(maker.num_species):
                col_species_name = model.species_names[col_idx]
                if not col_species_name in expected_species_names:
                    continue
                expected_col_idx = expected_species_names.index(col_species_name)
                self.assertAlmostEqual(float(jacobian_mat[row_idx, col_idx]),
                        float(expected_jacobian_mat[expected_row_idx, expected_col_idx]))
    
    def testMakeSymbolicJacobianNonLTI(self):
        if IGNORE_TEST:
            return
        # J2: S2 + 2 S1 -> 2 S3 + S1; k2*S2*S1*S1
        model = Model(MODEL3)
        maker = SymbolicJacobianMaker(model)
        reaction_dct = self.getReactions(maker=maker)
        reaction = reaction_dct["J2"]
        result_j2 = self.maker._makeReactionLti(reaction)
        S1, S2, k2 = sp.symbols("S1 S2 k2")
        expected_A_smat = sp.Matrix([
                [-2*S1*S2*k2, -1*S1**2*k2, 0],
                [-2*S1*S2*k2, -1*S1**2*k2, 0],
                [4*S1*S2*k2, 2*S1**2*k2, 0]])
        expected_b_smat = sp.Matrix([[0],
                                        [0],
                                        [0]])
        self.assertTrue(result_j2.A_smat.equals(expected_A_smat))
        self.assertTrue(result_j2.b_smat.equals(expected_b_smat))
    
    def testMakeSymbolicJacobianSmall(self):
        if IGNORE_TEST:
            return
        for antimony_str in [MODEL1, MODEL2]:
            model = Model(antimony_str)
            maker = SymbolicJacobianMaker(model)
            maker.initialize()
            self.checkSymbolicJacobian(model, maker)



if __name__ == '__main__':
    unittest.main()