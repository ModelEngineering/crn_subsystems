from src.jacobian import Jacobian
from src.model import Model  # type: ignore
from tests.common_test import MODEL1, MODEL2, MODEL3  # type: ignore

import libsbml # type: ignore
import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict, Optional

IGNORE_TEST = False


class TestJacobian(unittest.TestCase):

    def setUp(self):
        self.model = Model(MODEL1)
        self.jacobian = Jacobian(self.model)

    def getReactions(self, maker: Optional[Jacobian]=None)->Dict[str, libsbml.Reaction]:
        # Gets the reactions from the model
        if maker is None:
            maker = self.jacobian
        dct = {}
        for i in range(maker.model.libsbml_model.getNumReactions()):
            reaction = maker.model.libsbml_model.getReaction(i)
            reaction_id = reaction.getId()
            dct[reaction_id] = maker.model.libsbml_model.getReaction(i)
        return dct

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsInstance(self.jacobian, Jacobian)

    def testMakeModel(self):
        if IGNORE_TEST:
            return
        self.assertIsNotNone(self.jacobian.model)
        self.assertEqual(self.jacobian.model.libsbml_model.getNumSpecies(), 3)
        self.assertEqual(self.jacobian.model.libsbml_model.getNumReactions(), 5)

    def testMakeReactionSymbolicJacobianJ1(self):
        # J1: -> S1; k1
        if IGNORE_TEST:
            return
        reaction_dct = self.getReactions()
        reaction = reaction_dct["J1"]
        result_j1 = self.jacobian._makeReactionLti(reaction)
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
        result_j2 = self.jacobian._makeReactionLti(reaction)
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
        result_j3 = self.jacobian._makeReactionLti(reaction)
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
        result_j4 = self.jacobian._makeReactionLti(reaction)
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
        result_j5 = self.jacobian._makeReactionLti(reaction)
        expected_A_smat = sp.Matrix([[0, 0, sp.Symbol("k2")],
                                            [0, 0, 3*sp.Symbol("k2")],  # type: ignore
                                            [0, 0, -sp.Symbol("k2")]])
        expected_b_smat = sp.Matrix([[0],
                                        [0],
                                        [0]])
        self.assertTrue(result_j5.A_smat.equals(expected_A_smat))
        self.assertTrue(result_j5.b_smat.equals(expected_b_smat))

    def checkSymbolicJacobian(self, model: Model, jacobian: Jacobian):
        jacobian_smat = jacobian._makeSymbolicJacobian()[0]
        jacobian_mat = jacobian_smat.subs(model.kinetic_constant_dct)
        expected_jacobian_mat = jacobian.jacobian_df.values
        expected_species_names = list(jacobian.jacobian_df.columns)
        for row_idx in range(model.num_species):
            row_species_name = model.species_names[row_idx]
            if not row_species_name in expected_species_names:
                continue
            expected_row_idx = expected_species_names.index(row_species_name)
            for col_idx in range(model.num_species):
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
        jacobian = Jacobian(model)
        reaction_dct = self.getReactions(maker=jacobian)
        reaction = reaction_dct["J2"]
        result_j2 = self.jacobian._makeReactionLti(reaction)
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
        for antimony_str in [MODEL2, MODEL1]:
            model = Model(antimony_str)
            jacobian = Jacobian(model)
            self.checkSymbolicJacobian(model, jacobian)


if __name__ == '__main__':
    unittest.main()