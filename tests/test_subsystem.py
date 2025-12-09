from src.jacobian import Jacobian
from src.model import Model  # type: ignore
from src.subsystem import Subsystem  # type: ignore
from tests.common_test import MODEL1, MODEL2, MODEL3, MODEL_SEQUENTIAL  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict, Optional

IGNORE_TEST = False


class TestSubsyste(unittest.TestCase):

    def setUp(self):
        self.subsystem = Subsystem(MODEL1)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsInstance(self.subsystem.model, Model)
        self.assertIsInstance(self.subsystem.jacobian, Jacobian)
        for name in ["S1", "S2", "S3"]:
            self.assertIn(name, self.subsystem.model.species_names)
        for name in ["J1", "J2", "J3", "J4", "J5"]:
            self.assertIn(name, self.subsystem.model.reaction_names)

    def testCalculateEigenvalues(self):
        if IGNORE_TEST:
            return
        subsystem = Subsystem(MODEL_SEQUENTIAL)
        eigenvalues = subsystem.calculateEigenvalues()
        self.assertIsInstance(eigenvalues, np.ndarray)
        self.assertEqual(len(eigenvalues), 4)
        trues = [any( np.isclose(e, v) for v in [-1., -2., -3., -4.]) for e in eigenvalues]
        self.assertTrue(all(trues))

    def testCalculateSymbolicDiscs(self):
        #if IGNORE_TEST:
        #    return
        subsystem = Subsystem(MODEL_SEQUENTIAL)
        symbolic_discs = subsystem.calculateSymbolicDiscs()
        self.assertIsInstance(symbolic_discs, tuple)
        self.assertEqual(len(symbolic_discs), 2)
        self.assertIsInstance(symbolic_discs[0], sp.Matrix)
        self.assertIsInstance(symbolic_discs[1], sp.Matrix)
        self.assertEqual(symbolic_discs[0].shape, (1, subsystem.model.num_species))
        self.assertEqual(symbolic_discs[1].shape, (1, subsystem.model.num_species))
        k1, k2, k3 = sp.symbols('k1 k2 k3')
        expected_width = 2.0*sp.Matrix(
                [[0, sp.Abs(k1), sp.Abs(k2), sp.Abs(k3)]])
        width = symbolic_discs[1] - symbolic_discs[0]
        self.assertTrue(width == expected_width)


if __name__ == '__main__':
    unittest.main()